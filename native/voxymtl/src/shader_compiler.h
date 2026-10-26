// GLSL 4.x (as written for OpenGL) -> SPIR-V (glslang) -> MSL (SPIRV-Cross).
#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace vmtl {

enum class StageKind { Vertex = 0, Fragment = 1, Compute = 2 };
enum class ResKind { Ubo = 0, Ssbo = 1, Texture = 2, Image = 3, Loose = 4 };

struct ResBinding {
    ResKind kind;
    int glBinding;   // binding index as written in the GLSL (0 for the loose uniform block)
    int mslIndex;    // [[buffer(n)]] for Ubo/Ssbo/Loose, [[texture(n)]] (and [[sampler(n)]]) otherwise
};

struct CompiledStage {
    std::string msl;
    std::string entry;
    std::vector<ResBinding> resources;
    uint32_t localSize[3] = {1, 1, 1};
    uint32_t looseBytes = 0;   // bytes of loose-uniform shadow data this stage needs
};

// Metal argument table layout (per stage)
constexpr int MSL_UBO_BASE = 0;     // ubo b   -> buffer(b)           b < 8
constexpr int MSL_SSBO_BASE = 8;    // ssbo b  -> buffer(8 + b)       b < 21
constexpr int MSL_LOOSE_SLOT = 30;  // loose uniforms
constexpr int MSL_IMAGE_BASE = 16;  // image b -> texture(16 + b)
constexpr int LOOSE_SLOT_STRIDE = 64; // bytes reserved per uniform location
constexpr int MAX_LOOSE_LOCATIONS = 64;

bool compileStage(StageKind kind, const std::string& glsl, CompiledStage& out, std::string& log);

} // namespace vmtl
