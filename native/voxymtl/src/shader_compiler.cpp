#include "shader_compiler.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <spirv_cross/spirv_msl.hpp>

#include <algorithm>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>

namespace vmtl {

namespace {

std::once_flag g_glslangInit;

EShLanguage toLang(StageKind k) {
    switch (k) {
        case StageKind::Vertex: return EShLangVertex;
        case StageKind::Fragment: return EShLangFragment;
        default: return EShLangCompute;
    }
}

void replaceAll(std::string& s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// Replaces whole-word occurrences only
void replaceWord(std::string& s, const std::string& word, const std::string& to) {
    std::regex re("\\b" + word + "\\b");
    s = std::regex_replace(s, re, to);
}

bool isOpaqueType(const std::string& t) {
    return t.rfind("sampler", 0) == 0 || t.rfind("isampler", 0) == 0 || t.rfind("usampler", 0) == 0 ||
           t.rfind("image", 0) == 0 || t.rfind("iimage", 0) == 0 || t.rfind("uimage", 0) == 0 ||
           t.rfind("texture", 0) == 0 || t.rfind("atomic_uint", 0) == 0;
}

// OpenGL style loose uniforms ("layout(location=N) uniform T name;") do not exist in Vulkan style GLSL.
// They are gathered in one std140 block whose members live at offset location*LOOSE_SLOT_STRIDE, the native
// side keeps a shadow copy of the values written by glUniform* and uploads it with setBytes at draw time.
bool rewriteLooseUniforms(std::string& src, uint32_t& looseBytes, std::string& err) {
    static const std::regex re(
        R"(layout\s*\(\s*location\s*=\s*(\d+)\s*\)\s*uniform\s+(\w+)\s+(\w+)\s*(\[\s*\d+\s*\])?\s*;)");
    std::map<int, std::pair<std::string, std::string>> members; // location -> (type, name)

    std::string out;
    out.reserve(src.size());
    auto begin = std::sregex_iterator(src.begin(), src.end(), re);
    auto end = std::sregex_iterator();
    size_t last = 0;
    for (auto it = begin; it != end; ++it) {
        const auto& m = *it;
        std::string type = m[2].str();
        if (isOpaqueType(type)) continue;
        if (m[4].matched) {
            err = "loose uniform arrays are not supported: " + m[3].str();
            return false;
        }
        int loc = std::stoi(m[1].str());
        if (loc >= MAX_LOOSE_LOCATIONS) {
            err = "uniform location out of range: " + m[1].str();
            return false;
        }
        out.append(src, last, m.position() - last);
        last = m.position() + m.length();
        members[loc] = {type, m[3].str()};
    }
    if (members.empty()) return true;
    out.append(src, last, std::string::npos);

    std::ostringstream block;
    block << "layout(std140, set = 0, binding = 31) uniform VmtlLoose {\n";
    for (auto& [loc, tn] : members) {
        block << "  layout(offset = " << (loc * LOOSE_SLOT_STRIDE) << ") " << tn.first << " " << tn.second << ";\n";
    }
    block << "};\n";
    looseBytes = (uint32_t)((members.rbegin()->first + 1) * LOOSE_SLOT_STRIDE);

    // insert after the last #version / #extension line at the top of the file
    size_t insertAt = 0;
    {
        size_t pos = 0;
        while (pos < out.size()) {
            size_t eol = out.find('\n', pos);
            if (eol == std::string::npos) eol = out.size();
            std::string line = out.substr(pos, eol - pos);
            size_t f = line.find_first_not_of(" \t\r");
            if (f != std::string::npos && line[f] == '#') {
                if (line.find("#version") != std::string::npos || line.find("#extension") != std::string::npos) {
                    insertAt = std::min(eol + 1, out.size());
                }
            } else if (f != std::string::npos && line.compare(f, 2, "//") != 0) {
                break; // first real token
            }
            pos = eol + 1;
        }
    }
    out.insert(insertAt, block.str());
    src = std::move(out);
    return true;
}

// glslang's "Vulkan semantics" map best to MSL, so the OpenGL builtins are translated.
void translateGlBuiltins(std::string& src, StageKind kind) {
    if (kind == StageKind::Vertex) {
        replaceWord(src, "gl_VertexID", "gl_VertexIndex");
        replaceWord(src, "gl_InstanceID", "(gl_InstanceIndex-gl_BaseInstance)");
    }
    // gl_BaseInstance & friends need the draw parameter extension
    size_t v = src.find("#version");
    if (v != std::string::npos) {
        size_t eol = src.find('\n', v);
        src.insert(eol + 1, "#extension GL_ARB_shader_draw_parameters : enable\n");
    }
    // NV barycentrics -> the EXT variant glslang/spirv-cross know about
    replaceAll(src, "GL_NV_fragment_shader_barycentric", "GL_EXT_fragment_shader_barycentric");
    replaceWord(src, "gl_BaryCoordNV", "gl_BaryCoordEXT");
}


// Vulkan style GLSL (and MSL) needs explicit locations on stage inputs / outputs, desktop GL links them by name. Plain
// "in T name;" / "out T name;" declarations get the next free location, in declaration order, which pairs a vertex shader
// with its fragment shader as long as both declare the shared varyings in the same order.
void addMissingLocations(std::string& src) {
    static const std::regex explicitRe(
        R"(layout\s*\([^)]*location\s*=\s*(\d+)[^)]*\)\s*(?:(?:flat|smooth|noperspective|centroid|sample|patch)\s+)*(in|out)\b)");
    static const std::regex plainRe(
        R"(^(\s*)((?:(?:flat|smooth|noperspective|centroid|sample)\s+)*(in|out)\s+\w+\s+\w+\s*;.*)$)");

    std::set<int> used[2];
    for (auto it = std::sregex_iterator(src.begin(), src.end(), explicitRe); it != std::sregex_iterator(); ++it) {
        used[(*it)[2].str() == "in" ? 0 : 1].insert(std::stoi((*it)[1].str()));
    }
    int next[2] = {0, 0};
    std::istringstream in(src);
    std::ostringstream out;
    std::string line;
    while (std::getline(in, line)) {
        std::smatch m;
        if (line.find("layout") == std::string::npos && std::regex_match(line, m, plainRe)) {
            int dir = m[3].str() == "in" ? 0 : 1;
            while (used[dir].count(next[dir])) next[dir]++;
            out << m[1].str() << "layout(location = " << next[dir]++ << ") " << m[2].str() << "\n";
        } else {
            out << line << "\n";
        }
    }
    src = out.str();
}

} // namespace

bool compileStage(StageKind kind, const std::string& glslIn, CompiledStage& out, std::string& log) {
    std::call_once(g_glslangInit, [] { glslang::InitializeProcess(); });

    const EShLanguage lang = toLang(kind);
    const TBuiltInResource* resources = GetDefaultResources();
    const EShMessages msgs = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

    // 1. glslang's preprocessor expands macros (binding numbers etc)
    std::string pre;
    {
        glslang::TShader shader(lang);
        const char* s = glslIn.c_str();
        shader.setStrings(&s, 1);
        shader.setEnvInput(glslang::EShSourceGlsl, lang, glslang::EShClientVulkan, 100);
        shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
        shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);
        glslang::TShader::ForbidIncluder includer;
        if (!shader.preprocess(resources, 460, ECoreProfile, false, false, msgs, &pre, includer)) {
            log += "preprocess failed:\n";
            log += shader.getInfoLog();
            return false;
        }
    }

    // 2. translate GL-isms
    std::string err;
    translateGlBuiltins(pre, kind);
    if (kind != StageKind::Compute) addMissingLocations(pre);
    uint32_t looseBytes = 0;
    if (!rewriteLooseUniforms(pre, looseBytes, err)) {
        log += err;
        return false;
    }

    // 3. glsl -> spirv
    std::vector<uint32_t> spirv;
    {
        glslang::TShader shader(lang);
        const char* s = pre.c_str();
        shader.setStrings(&s, 1);
        shader.setEnvInput(glslang::EShSourceGlsl, lang, glslang::EShClientVulkan, 100);
        shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
        shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);
        shader.setEntryPoint("main");
        if (!shader.parse(resources, 460, ECoreProfile, false, false, msgs)) {
            log += "parse failed:\n";
            log += shader.getInfoLog();
            log += "\n--- preprocessed source ---\n" + pre;
            return false;
        }
        glslang::TProgram program;
        program.addShader(&shader);
        if (!program.link(msgs)) {
            log += "link failed:\n";
            log += program.getInfoLog();
            return false;
        }
        glslang::SpvOptions opts;
        opts.validate = false;
        opts.disableOptimizer = false;
        spv::SpvBuildLogger logger;
        glslang::GlslangToSpv(*program.getIntermediate(lang), spirv, &logger, &opts);
        log += logger.getAllMessages();
    }

    // 4. spirv -> msl
    try {
        spirv_cross::CompilerMSL msl(std::move(spirv));

        auto mopts = msl.get_msl_options();
        mopts.set_msl_version(2, 4);
        mopts.platform = spirv_cross::CompilerMSL::Options::macOS;
        mopts.enable_decoration_binding = true;
        mopts.argument_buffers = false;
        mopts.pad_fragment_output_components = true;
        mopts.texture_buffer_native = true;
        mopts.dynamic_offsets_buffer_index = 29;
        mopts.buffer_size_buffer_index = 28;
        msl.set_msl_options(mopts);

        auto copts = msl.get_common_options();
        // GL clip space z is [-1,1], Metal's is [0,1]
        copts.vertex.fixup_clipspace = true;
        // keep GL's bottom-left orientation of render targets (gl_FragCoord / texel layout stay GL-like)
        copts.vertex.flip_vert_y = true;
        msl.set_common_options(copts);

        const spv::ExecutionModel model = kind == StageKind::Vertex ? spv::ExecutionModelVertex
                                        : kind == StageKind::Fragment ? spv::ExecutionModelFragment
                                        : spv::ExecutionModelGLCompute;

        auto addBinding = [&](uint32_t uniqueBinding, int buffer, int texture, int sampler) {
            spirv_cross::MSLResourceBinding rb{};
            rb.stage = model;
            rb.desc_set = 0;
            rb.binding = uniqueBinding;
            rb.msl_buffer = buffer < 0 ? 0 : buffer;
            rb.msl_texture = texture < 0 ? 0 : texture;
            rb.msl_sampler = sampler < 0 ? 0 : sampler;
            msl.add_msl_resource_binding(rb);
        };

        // GL has one binding namespace per resource type but Vulkan style SPIR-V has a single one, so every
        // resource gets a unique binding (per type base) and an explicit MSL slot
        auto res = msl.get_shader_resources();
        out.resources.clear();
        for (auto& r : res.uniform_buffers) {
            const bool loose = r.name.find("VmtlLoose") != std::string::npos ||
                               msl.get_name(r.base_type_id).find("VmtlLoose") != std::string::npos;
            uint32_t b = msl.get_decoration(r.id, spv::DecorationBinding);
            if (loose) {
                msl.set_decoration(r.id, spv::DecorationBinding, 500);
                msl.set_decoration(r.id, spv::DecorationDescriptorSet, 0);
                addBinding(500, MSL_LOOSE_SLOT, -1, -1);
                out.resources.push_back({ResKind::Loose, 0, MSL_LOOSE_SLOT});
                continue;
            }
            if (b >= 8) { log += "ubo binding too large"; return false; }
            msl.set_decoration(r.id, spv::DecorationBinding, 100 + b);
            msl.set_decoration(r.id, spv::DecorationDescriptorSet, 0);
            addBinding(100 + b, MSL_UBO_BASE + b, -1, -1);
            out.resources.push_back({ResKind::Ubo, (int)b, MSL_UBO_BASE + (int)b});
        }
        for (auto& r : res.storage_buffers) {
            uint32_t b = msl.get_decoration(r.id, spv::DecorationBinding);
            if (b >= 21) { log += "ssbo binding too large"; return false; }
            msl.set_decoration(r.id, spv::DecorationBinding, 200 + b);
            msl.set_decoration(r.id, spv::DecorationDescriptorSet, 0);
            addBinding(200 + b, MSL_SSBO_BASE + b, -1, -1);
            out.resources.push_back({ResKind::Ssbo, (int)b, MSL_SSBO_BASE + (int)b});
        }
        for (auto& r : res.sampled_images) {
            uint32_t b = msl.get_decoration(r.id, spv::DecorationBinding);
            if (b >= 16) { log += "texture binding too large"; return false; }
            msl.set_decoration(r.id, spv::DecorationBinding, 300 + b);
            msl.set_decoration(r.id, spv::DecorationDescriptorSet, 0);
            addBinding(300 + b, -1, (int)b, (int)b);
            out.resources.push_back({ResKind::Texture, (int)b, (int)b});
        }
        for (auto& r : res.storage_images) {
            uint32_t b = msl.get_decoration(r.id, spv::DecorationBinding);
            if (b >= 16) { log += "image binding too large"; return false; }
            msl.set_decoration(r.id, spv::DecorationBinding, 400 + b);
            msl.set_decoration(r.id, spv::DecorationDescriptorSet, 0);
            addBinding(400 + b, -1, MSL_IMAGE_BASE + (int)b, -1);
            out.resources.push_back({ResKind::Image, (int)b, MSL_IMAGE_BASE + (int)b});
        }

        if (kind == StageKind::Compute) {
            auto& ep = msl.get_entry_point("main", model);
            out.localSize[0] = ep.workgroup_size.x ? ep.workgroup_size.x : 1;
            out.localSize[1] = ep.workgroup_size.y ? ep.workgroup_size.y : 1;
            out.localSize[2] = ep.workgroup_size.z ? ep.workgroup_size.z : 1;
        }

        out.looseBytes = looseBytes;
        out.msl = msl.compile();
        out.entry = msl.get_cleansed_entry_point_name("main", model);
    } catch (const std::exception& e) {
        log += std::string("spirv-cross failed: ") + e.what();
        return false;
    }
    return true;
}

} // namespace vmtl
