// Metal implementation of the OpenGL 4.6 subset Voxy needs.
//
// The Java side drives this like a tiny GL: objects are created with ids, state is set, draws / dispatches
// are issued. Everything is recorded into one MTLCommandBuffer at a time, encoders are opened lazily and
// switched (render / compute / blit) as needed.
//
// Conventions:
//  * GL render target orientation is kept: vertex shaders are translated with flip_vert_y and the winding
//    order / viewport is adjusted to match, so textures have the same texel layout as in GL.
//  * GL clip space depth ([-1,1]) is converted to [0,1] in the vertex shader epilogue (fixup_clipspace).
//  * glMultiDrawElementsIndirectCount is emulated with an indirect command buffer that a compute kernel fills
//    from the GPU written draw list and count.
//  * GL_UNSIGNED_BYTE index buffers are widened to 16 bit, GL_TRIANGLE_FAN is converted to indexed triangles.
//  * GL_DEPTH24_STENCIL8 (not available on Apple silicon) is stored as Depth32Float_Stencil8.

#define GL_SILENCE_DEPRECATION 1
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/CGLIOSurface.h>
#import <OpenGL/gl3.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vmtl.h"
#include "shader_compiler.h"
#include "kernels.h"

#ifndef GL_DEPTH_COMPONENT32
#define GL_DEPTH_COMPONENT32 0x81A7
#endif
#ifndef GL_TEXTURE_RECTANGLE
#define GL_TEXTURE_RECTANGLE 0x84F5
#endif
#define GLC_UNIFORM_BUFFER 0x8A11
#define GLC_SHADER_STORAGE_BUFFER 0x90D2
#define GLC_STENCIL_ATTACHMENT 0x8D20
#define GLC_DEPTH_ATTACHMENT 0x8D00
#define GLC_DEPTH_STENCIL_ATTACHMENT 0x821A
#define GLC_COLOR_ATTACHMENT0 0x8CE0
#define GLC_DEPTH32F_STENCIL8 0x8CAD

namespace vmtl {
namespace {

// ---------------------------------------------------------------------------------------------
// objects
// ---------------------------------------------------------------------------------------------

struct Buffer {
    id<MTLBuffer> buf;
    uint64_t size = 0;
    uint64_t version = 0;   // bumped whenever the gpu / api writes into it, used by derived caches
    bool cpuVisible = false;
};

struct SamplerParams {
    uint32_t minFilter = GL_NEAREST_MIPMAP_LINEAR;
    uint32_t magFilter = GL_LINEAR;
    uint32_t wrapS = GL_REPEAT, wrapT = GL_REPEAT;
    float minLod = -1000.0f, maxLod = 1000.0f;
};

struct Texture {
    id<MTLTexture> tex;
    std::vector<id<MTLTexture>> levelViews; // lazily created single level views (image bindings)
    std::unordered_map<uint64_t, id<MTLTexture>> rangeViews; // GL_TEXTURE_BASE_LEVEL / MAX_LEVEL restricted views
    uint32_t baseLevel = 0, maxLevel = 1000;
    uint32_t glFormat = 0;
    MTLPixelFormat pf = MTLPixelFormatInvalid;
    uint32_t w = 0, h = 0, levels = 1;
    bool hasDepth = false, hasStencil = false;
    SamplerParams params;
    IOSurfaceRef surface = nullptr;
    GLuint glName = 0;
};

struct Sampler {
    SamplerParams params;
};

struct Attachment {
    uint32_t tex = 0;
    uint32_t level = 0;
};

struct Framebuffer {
    Attachment colour[8];
    Attachment depth;
    bool stencilAttached = false;
    uint32_t drawMask = 1;
    uint64_t sig = 1;
    // pending clears, applied lazily as load actions
    bool clearColour[8] = {};
    double clearColourValue[8][4] = {};
    bool clearDepth = false, clearStencil = false;
    double clearDepthValue = 1.0;
    uint32_t clearStencilValue = 0;
    bool hasPending() const {
        if (clearDepth || clearStencil) return true;
        for (bool b : clearColour) if (b) return true;
        return false;
    }
};

struct StageInfo {
    id<MTLFunction> fn;
    std::vector<ResBinding> res;
    uint32_t looseBytes = 0;
    uint32_t local[3] = {1, 1, 1};
};

struct Program {
    bool compute = false;
    StageInfo vs, fs, cs;
    id<MTLComputePipelineState> cpso;
    std::unordered_map<std::string, id<MTLRenderPipelineState>> psos;
    uint8_t uniforms[MAX_LOOSE_LOCATIONS * LOOSE_SLOT_STRIDE] = {};
};

struct Fence {
    std::shared_ptr<std::atomic<bool>> done;
};

struct BufBind { uint32_t buf = 0; int64_t off = 0; int64_t size = -1; };
struct ImgBind { uint32_t tex = 0; int level = 0; };

struct State {
    uint32_t prog = 0, fb = 0, elem = 0, indirect = 0, param = 0, dispatch = 0;
    BufBind ubo[8], ssbo[24];
    uint32_t texUnit[16] = {}, sampUnit[16] = {};
    ImgBind img[16];

    bool depthTest = false, stencilTest = false, cull = false, polyLine = false, frontCW = false;
    uint32_t cullFace = GL_BACK;
    uint32_t depthFunc = GL_LESS;
    bool depthMask = true;
    bool colourMask[8][4] = {{true,true,true,true},{true,true,true,true},{true,true,true,true},{true,true,true,true},
                             {true,true,true,true},{true,true,true,true},{true,true,true,true},{true,true,true,true}};
    uint32_t stFunc = GL_ALWAYS;
    int stRef = 0;
    uint32_t stMask = 0xFF, stWriteMask = 0xFF;
    uint32_t sFail = GL_KEEP, dpFail = GL_KEEP, dpPass = GL_KEEP;

    bool blendEn[8] = {};
    uint32_t bSrcRGB[8], bDstRGB[8], bSrcA[8], bDstA[8];
    int vp[4] = {0, 0, 0, 0};
    bool vpSet = false;

    State() {
        for (int i = 0; i < 8; i++) { bSrcRGB[i] = bSrcA[i] = GL_ONE; bDstRGB[i] = bDstA[i] = GL_ZERO; }
    }
};

struct IcbEntry {
    id<MTLIndirectCommandBuffer> icb;
    id<MTLBuffer> argBuf;
    id<MTLBuffer> range;
    uint32_t capacity = 0;
};

struct Ctx {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> cb;
    id<MTLCommandBuffer> lastCb;
    id<MTLRenderCommandEncoder> re;
    id<MTLComputeCommandEncoder> ce;
    id<MTLBlitCommandEncoder> be;
    uint32_t reFb = 0;
    uint64_t reSig = 0;

    id<MTLLibrary> kernels;
    id<MTLFunction> fMdic;
    id<MTLComputePipelineState> kFill32, kWiden, kDepth, kMdic;
    id<MTLArgumentEncoder> mdicArgEnc;
    bool icbSupported = false;
    bool unified = false;

    uint32_t nextId = 1;
    std::unordered_map<uint32_t, Buffer> buffers;
    std::unordered_map<uint32_t, Texture> textures;
    std::unordered_map<uint32_t, Sampler> samplers;
    std::unordered_map<uint32_t, Framebuffer> fbs;
    std::unordered_map<uint32_t, std::unique_ptr<Program>> programs;
    std::unordered_map<uint32_t, Fence> fences;

    std::unordered_map<std::string, id<MTLSamplerState>> samplerCache;
    std::unordered_map<std::string, id<MTLDepthStencilState>> dssCache;
    std::unordered_map<uint32_t, IcbEntry> icbs; // by capacity
    struct Widened { id<MTLBuffer> buf; uint64_t version = ~0ull; };
    std::unordered_map<uint32_t, Widened> widened; // source buffer id -> 16 bit copy
    std::unordered_map<uint32_t, id<MTLBuffer>> fanIndex; // vertex count -> index buffer
    std::unordered_set<uint32_t> pendingClearFbs;

    State st;
};

Ctx* g = nullptr;

inline uint32_t newId() { return HANDLE_TAG | (g->nextId++); }

template <class M>
inline typename M::mapped_type* find(M& m, uint32_t id) {
    auto it = m.find(id);
    return it == m.end() ? nullptr : &it->second;
}
inline Program* findProg(uint32_t id) {
    auto it = g->programs.find(id);
    return it == g->programs.end() ? nullptr : it->second.get();
}

// ---------------------------------------------------------------------------------------------
// enum mapping
// ---------------------------------------------------------------------------------------------

MTLPixelFormat mapFormat(uint32_t f, bool& depth, bool& stencil) {
    depth = stencil = false;
    switch (f) {
        case GL_RGBA8: return MTLPixelFormatRGBA8Unorm;
        case GL_RGBA16F: return MTLPixelFormatRGBA16Float;
        case GL_RGBA32F: return MTLPixelFormatRGBA32Float;
        case GL_RGBA32UI: return MTLPixelFormatRGBA32Uint;
        case GL_RG8: return MTLPixelFormatRG8Unorm;
        case GL_RG16F: return MTLPixelFormatRG16Float;
        case GL_RG32F: return MTLPixelFormatRG32Float;
        case GL_R8: return MTLPixelFormatR8Unorm;
        case GL_R8UI: return MTLPixelFormatR8Uint;
        case GL_R16F: return MTLPixelFormatR16Float;
        case GL_R32F: return MTLPixelFormatR32Float;
        case GL_R32UI: return MTLPixelFormatR32Uint;
        case GL_RGB10_A2: return MTLPixelFormatRGB10A2Unorm;
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F: depth = true; return MTLPixelFormatDepth32Float;
        case GL_DEPTH24_STENCIL8:
        case GLC_DEPTH32F_STENCIL8: depth = stencil = true; return MTLPixelFormatDepth32Float_Stencil8;
        default: return MTLPixelFormatInvalid;
    }
}

MTLCompareFunction mapCompare(uint32_t f) {
    switch (f) {
        case GL_NEVER: return MTLCompareFunctionNever;
        case GL_LESS: return MTLCompareFunctionLess;
        case GL_EQUAL: return MTLCompareFunctionEqual;
        case GL_LEQUAL: return MTLCompareFunctionLessEqual;
        case GL_GREATER: return MTLCompareFunctionGreater;
        case GL_NOTEQUAL: return MTLCompareFunctionNotEqual;
        case GL_GEQUAL: return MTLCompareFunctionGreaterEqual;
        default: return MTLCompareFunctionAlways;
    }
}

MTLStencilOperation mapStencilOp(uint32_t o) {
    switch (o) {
        case GL_ZERO: return MTLStencilOperationZero;
        case GL_REPLACE: return MTLStencilOperationReplace;
        case GL_INCR: return MTLStencilOperationIncrementClamp;
        case GL_DECR: return MTLStencilOperationDecrementClamp;
        case GL_INVERT: return MTLStencilOperationInvert;
        case GL_INCR_WRAP: return MTLStencilOperationIncrementWrap;
        case GL_DECR_WRAP: return MTLStencilOperationDecrementWrap;
        default: return MTLStencilOperationKeep;
    }
}

MTLBlendFactor mapBlend(uint32_t f) {
    switch (f) {
        case GL_ZERO: return MTLBlendFactorZero;
        case GL_ONE: return MTLBlendFactorOne;
        case GL_SRC_COLOR: return MTLBlendFactorSourceColor;
        case GL_ONE_MINUS_SRC_COLOR: return MTLBlendFactorOneMinusSourceColor;
        case GL_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
        case GL_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
        case GL_DST_ALPHA: return MTLBlendFactorDestinationAlpha;
        case GL_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationAlpha;
        case GL_DST_COLOR: return MTLBlendFactorDestinationColor;
        case GL_ONE_MINUS_DST_COLOR: return MTLBlendFactorOneMinusDestinationColor;
        case GL_SRC_ALPHA_SATURATE: return MTLBlendFactorSourceAlphaSaturated;
        default: return MTLBlendFactorOne;
    }
}

MTLPrimitiveType mapPrim(uint32_t m) {
    switch (m) {
        case GL_POINTS: return MTLPrimitiveTypePoint;
        case GL_LINES: return MTLPrimitiveTypeLine;
        case GL_LINE_STRIP: return MTLPrimitiveTypeLineStrip;
        case GL_TRIANGLE_STRIP: return MTLPrimitiveTypeTriangleStrip;
        default: return MTLPrimitiveTypeTriangle;
    }
}

MTLSamplerAddressMode mapWrap(uint32_t w) {
    switch (w) {
        case GL_CLAMP_TO_EDGE: return MTLSamplerAddressModeClampToEdge;
        case GL_MIRRORED_REPEAT: return MTLSamplerAddressModeMirrorRepeat;
        case GL_CLAMP_TO_BORDER: return MTLSamplerAddressModeClampToBorderColor;
        default: return MTLSamplerAddressModeRepeat;
    }
}

void applyParam(SamplerParams& p, uint32_t pname, int64_t valueBits, bool isFloat) {
    float fv = 0;
    int32_t iv = (int32_t)valueBits;
    if (isFloat) { uint32_t b = (uint32_t)valueBits; std::memcpy(&fv, &b, 4); iv = (int32_t)fv; }
    else fv = (float)iv;
    switch (pname) {
        case GL_TEXTURE_MIN_FILTER: p.minFilter = (uint32_t)iv; break;
        case GL_TEXTURE_MAG_FILTER: p.magFilter = (uint32_t)iv; break;
        case GL_TEXTURE_WRAP_S: p.wrapS = (uint32_t)iv; break;
        case GL_TEXTURE_WRAP_T: p.wrapT = (uint32_t)iv; break;
        case GL_TEXTURE_MIN_LOD: p.minLod = fv; break;
        case GL_TEXTURE_MAX_LOD: p.maxLod = fv; break;
        default: break; // depth stencil texture mode, compare mode, ... are not needed
    }
}

// bytes per texel of the *client* data described by (format, type)
uint32_t clientBpp(uint32_t format, uint32_t type) {
    uint32_t comps = 4;
    switch (format) {
        case GL_RED: case GL_RED_INTEGER: case GL_DEPTH_COMPONENT: case GL_STENCIL_INDEX: comps = 1; break;
        case GL_RG: case GL_RG_INTEGER: comps = 2; break;
        case GL_RGB: case GL_RGB_INTEGER: comps = 3; break;
        case GL_DEPTH_STENCIL: return 4;
        default: comps = 4; break;
    }
    uint32_t sz = 4;
    switch (type) {
        case GL_UNSIGNED_BYTE: case GL_BYTE: sz = 1; break;
        case GL_UNSIGNED_SHORT: case GL_SHORT: case GL_HALF_FLOAT: sz = 2; break;
        case GL_UNSIGNED_INT_24_8: case GL_UNSIGNED_INT_8_8_8_8_REV: case GL_UNSIGNED_INT_2_10_10_10_REV: return 4;
        default: sz = 4; break;
    }
    return comps * sz;
}

// ---------------------------------------------------------------------------------------------
// encoder management
// ---------------------------------------------------------------------------------------------

void ensureCB() {
    if (!g->cb) g->cb = [g->queue commandBuffer];
}

void endEncoders() {
    if (g->re) { [g->re endEncoding]; g->re = nil; }
    if (g->ce) { [g->ce endEncoding]; g->ce = nil; }
    if (g->be) { [g->be endEncoding]; g->be = nil; }
    g->reFb = 0;
}

MTLRenderPassDescriptor* makePass(Framebuffer& fb, bool consumeClears) {
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    for (int i = 0; i < 8; i++) {
        auto* t = find(g->textures, fb.colour[i].tex);
        if (!t) continue;
        auto* a = pass.colorAttachments[i];
        a.texture = t->tex;
        a.level = fb.colour[i].level;
        a.storeAction = MTLStoreActionStore;
        if (fb.clearColour[i]) {
            a.loadAction = MTLLoadActionClear;
            a.clearColor = MTLClearColorMake(fb.clearColourValue[i][0], fb.clearColourValue[i][1],
                                             fb.clearColourValue[i][2], fb.clearColourValue[i][3]);
        } else {
            a.loadAction = MTLLoadActionLoad;
        }
    }
    if (auto* t = find(g->textures, fb.depth.tex)) {
        auto* d = pass.depthAttachment;
        d.texture = t->tex;
        d.level = fb.depth.level;
        d.storeAction = MTLStoreActionStore;
        d.loadAction = fb.clearDepth ? MTLLoadActionClear : MTLLoadActionLoad;
        d.clearDepth = fb.clearDepthValue;
        if (fb.stencilAttached && t->hasStencil) {
            auto* s = pass.stencilAttachment;
            s.texture = t->tex;
            s.level = fb.depth.level;
            s.storeAction = MTLStoreActionStore;
            s.loadAction = fb.clearStencil ? MTLLoadActionClear : MTLLoadActionLoad;
            s.clearStencil = fb.clearStencilValue;
        }
    }
    if (consumeClears) {
        for (int i = 0; i < 8; i++) fb.clearColour[i] = false;
        fb.clearDepth = fb.clearStencil = false;
    }
    return pass;
}

// Realise pending clears of every framebuffer except `exclude` with an empty render pass
void flushClears(uint32_t exclude) {
    if (g->pendingClearFbs.empty()) return;
    std::vector<uint32_t> ids(g->pendingClearFbs.begin(), g->pendingClearFbs.end());
    for (uint32_t id : ids) {
        if (id == exclude) continue;
        g->pendingClearFbs.erase(id);
        auto* fb = find(g->fbs, id);
        if (!fb || !fb->hasPending()) continue;
        endEncoders();
        ensureCB();
        auto enc = [g->cb renderCommandEncoderWithDescriptor:makePass(*fb, true)];
        [enc endEncoding];
    }
}

void beginCompute() {
    flushClears(0);
    if (g->ce) return;
    endEncoders();
    ensureCB();
    g->ce = [g->cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
}

void beginBlit() {
    flushClears(0);
    if (g->be) return;
    endEncoders();
    ensureCB();
    g->be = [g->cb blitCommandEncoder];
}

bool beginRender(uint32_t fbId) {
    auto* fb = find(g->fbs, fbId);
    if (!fb) return false;
    flushClears(fbId);
    if (g->re && g->reFb == fbId && g->reSig == fb->sig && !fb->hasPending()) return true;
    endEncoders();
    ensureCB();
    auto pass = makePass(*fb, true);
    g->pendingClearFbs.erase(fbId);
    g->re = [g->cb renderCommandEncoderWithDescriptor:pass];
    g->reFb = fbId;
    g->reSig = fb->sig;
    return true;
}

void flushCB() {
    endEncoders();
    flushClears(0);
    endEncoders();
    if (g->cb) {
        [g->cb commit];
        g->lastCb = g->cb;
        g->cb = nil;
    }
}

void finishCB() {
    flushCB();
    if (g->lastCb) [g->lastCb waitUntilCompleted];
}

// ---------------------------------------------------------------------------------------------
// buffers
// ---------------------------------------------------------------------------------------------

id<MTLBuffer> bufOf(uint32_t id) {
    auto* b = find(g->buffers, id);
    return b ? b->buf : nil;
}

void touch(uint32_t id) {
    if (auto* b = find(g->buffers, id)) b->version++;
}

void dispatch1D(id<MTLComputePipelineState> pso, uint32_t n) {
    NSUInteger tg = MIN((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
    [g->ce dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void bufferFill32(uint32_t id, uint64_t off, uint64_t size, uint32_t value) {
    auto* b = find(g->buffers, id);
    if (!b) return;
    if ((int64_t)size < 0) size = b->size - off; // whole buffer
    touch(id);
    if (value == 0 || (value & 0xFFFFFFFFu) == ((value & 0xFF) * 0x01010101u)) {
        beginBlit();
        [g->be fillBuffer:b->buf range:NSMakeRange(off, size) value:(uint8_t)(value & 0xFF)];
        return;
    }
    beginCompute();
    [g->ce setComputePipelineState:g->kFill32];
    [g->ce setBuffer:b->buf offset:0 atIndex:0];
    uint32_t p[3] = {(uint32_t)(off / 4), (uint32_t)(size / 4), value};
    [g->ce setBytes:p length:sizeof(p) atIndex:1];
    dispatch1D(g->kFill32, p[1]);
}

void bufferCopy(uint32_t src, uint32_t dst, uint64_t so, uint64_t dof, uint64_t size) {
    auto* s = find(g->buffers, src);
    auto* d = find(g->buffers, dst);
    if (!s || !d) return;
    touch(dst);
    beginBlit();
    [g->be copyFromBuffer:s->buf sourceOffset:so toBuffer:d->buf destinationOffset:dof size:size];
}

void bufferRead(uint32_t id, uint64_t off, uint64_t size, void* dst) {
    auto* b = find(g->buffers, id);
    if (!b) return;
    if (b->cpuVisible) {
        finishCB();
        std::memcpy(dst, (uint8_t*)b->buf.contents + off, size);
        return;
    }
    id<MTLBuffer> stage = [g->dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    beginBlit();
    [g->be copyFromBuffer:b->buf sourceOffset:off toBuffer:stage destinationOffset:0 size:size];
    finishCB();
    std::memcpy(dst, stage.contents, size);
}

void bufferSubData(uint32_t id, uint64_t off, uint64_t size, const void* src) {
    auto* b = find(g->buffers, id);
    if (!b) return;
    touch(id);
    if (b->cpuVisible) {
        std::memcpy((uint8_t*)b->buf.contents + off, src, size);
        return;
    }
    id<MTLBuffer> stage = [g->dev newBufferWithBytes:src length:size options:MTLResourceStorageModeShared];
    beginBlit();
    [g->be copyFromBuffer:stage sourceOffset:0 toBuffer:b->buf destinationOffset:off size:size];
}

// ---------------------------------------------------------------------------------------------
// textures
// ---------------------------------------------------------------------------------------------

id<MTLTexture> viewOfLevel(Texture& t, uint32_t level) {
    if (t.levels <= 1 && level == 0) return t.tex;
    if (t.levelViews.size() < t.levels) t.levelViews.resize(t.levels);
    if (!t.levelViews[level]) {
        t.levelViews[level] = [t.tex newTextureViewWithPixelFormat:t.pf
                                                       textureType:MTLTextureType2D
                                                            levels:NSMakeRange(level, 1)
                                                            slices:NSMakeRange(0, 1)];
    }
    return t.levelViews[level];
}

// the texture as seen through GL_TEXTURE_BASE_LEVEL / GL_TEXTURE_MAX_LEVEL
id<MTLTexture> sampledView(Texture& t) {
    uint32_t base = MIN(t.baseLevel, t.levels - 1);
    uint32_t last = MIN(t.maxLevel, t.levels - 1);
    if (last < base) last = base;
    if (base == 0 && last == t.levels - 1) return t.tex;
    uint64_t key = ((uint64_t)base << 32) | last;
    auto it = t.rangeViews.find(key);
    if (it != t.rangeViews.end()) return it->second;
    id<MTLTexture> v = [t.tex newTextureViewWithPixelFormat:t.pf textureType:MTLTextureType2D
                                                     levels:NSMakeRange(base, last - base + 1) slices:NSMakeRange(0, 1)];
    t.rangeViews[key] = v;
    return v;
}

uint32_t createSharedTexture(uint32_t w, uint32_t h, uint32_t glFormat, uint32_t* glNameOut) {
    uint32_t bpe = 4;
    OSType osFmt;
    MTLPixelFormat mpf;
    GLenum ifmt, fmt, type;
    if (glFormat == GL_R32F) {
        osFmt = 'L00f'; mpf = MTLPixelFormatR32Float; ifmt = GL_R32F; fmt = GL_RED; type = GL_FLOAT;
    } else {
        osFmt = 'BGRA'; mpf = MTLPixelFormatBGRA8Unorm; ifmt = GL_RGBA; fmt = GL_BGRA; type = GL_UNSIGNED_INT_8_8_8_8_REV;
    }
    NSDictionary* props = @{
        (id)kIOSurfaceWidth: @(w),
        (id)kIOSurfaceHeight: @(h),
        (id)kIOSurfaceBytesPerElement: @(bpe),
        (id)kIOSurfacePixelFormat: @(osFmt),
    };
    IOSurfaceRef surf = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    if (!surf) return 0;
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:mpf width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> mt = [g->dev newTextureWithDescriptor:d iosurface:surf plane:0];
    if (!mt) { CFRelease(surf); return 0; }

    GLuint name = 0;
    glGenTextures(1, &name);
    glBindTexture(GL_TEXTURE_RECTANGLE, name);
    CGLError err = CGLTexImageIOSurface2D(CGLGetCurrentContext(), GL_TEXTURE_RECTANGLE, ifmt, (GLsizei)w, (GLsizei)h, fmt, type, surf, 0);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
    if (err != kCGLNoError) {
        glDeleteTextures(1, &name);
        CFRelease(surf);
        return 0;
    }

    uint32_t id = newId();
    Texture t;
    t.tex = mt;
    t.glFormat = glFormat;
    t.pf = mpf;
    t.w = w; t.h = h; t.levels = 1;
    t.surface = surf;
    t.glName = name;
    g->textures.emplace(id, std::move(t));
    if (glNameOut) *glNameOut = name;
    return id;
}

uint32_t createTexture(uint32_t w, uint32_t h, uint32_t levels, uint32_t glFormat, bool shared, uint32_t* glNameOut) {
    if (shared) return createSharedTexture(w, h, glFormat, glNameOut);
    Texture t;
    t.pf = mapFormat(glFormat, t.hasDepth, t.hasStencil);
    if (t.pf == MTLPixelFormatInvalid) return 0;
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pf width:w height:h mipmapped:levels > 1];
    d.mipmapLevelCount = levels;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
    if (!t.hasDepth) d.usage |= MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModePrivate;
    t.tex = [g->dev newTextureWithDescriptor:d];
    if (!t.tex) return 0;
    t.glFormat = glFormat;
    t.w = w; t.h = h; t.levels = levels;
    uint32_t id = newId();
    g->textures.emplace(id, std::move(t));
    return id;
}

void textureSubImage(uint32_t id, uint32_t lvl, uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                     uint32_t glFormat, uint32_t glType, const void* data) {
    auto* t = find(g->textures, id);
    if (!t) return;
    uint32_t bpp = clientBpp(glFormat, glType);
    uint64_t rowBytes = (uint64_t)w * bpp;
    id<MTLBuffer> stage = [g->dev newBufferWithBytes:data length:rowBytes * h options:MTLResourceStorageModeShared];
    beginBlit();
    [g->be copyFromBuffer:stage
             sourceOffset:0
        sourceBytesPerRow:rowBytes
      sourceBytesPerImage:rowBytes * h
               sourceSize:MTLSizeMake(w, h, 1)
                toTexture:t->tex
         destinationSlice:0
         destinationLevel:lvl
        destinationOrigin:MTLOriginMake(x, y, 0)];
}

void textureClear(uint32_t id, uint32_t lvl, uint32_t glFormat, uint32_t glType, const void* value) {
    auto* t = find(g->textures, id);
    if (!t) return;
    endEncoders();
    ensureCB();
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    if (t->hasDepth) {
        double dv = 0.0;
        uint32_t sv = 0;
        if (value) {
            if (glType == GL_FLOAT) { float f; std::memcpy(&f, value, 4); dv = f; }
            else if (glType == GL_UNSIGNED_INT_24_8) { uint32_t u; std::memcpy(&u, value, 4); dv = (u >> 8) / 16777215.0; sv = u & 0xFF; }
            else if (glType == GL_FLOAT_32_UNSIGNED_INT_24_8_REV) { float f; std::memcpy(&f, value, 4); dv = f; uint32_t u; std::memcpy(&u, (const uint8_t*)value + 4, 4); sv = u & 0xFF; }
        }
        auto* d = pass.depthAttachment;
        d.texture = t->tex; d.level = lvl; d.loadAction = MTLLoadActionClear; d.storeAction = MTLStoreActionStore; d.clearDepth = dv;
        if (t->hasStencil) {
            auto* s = pass.stencilAttachment;
            s.texture = t->tex; s.level = lvl; s.loadAction = MTLLoadActionClear; s.storeAction = MTLStoreActionStore; s.clearStencil = sv;
        }
    } else {
        double c[4] = {0, 0, 0, 0};
        if (value) {
            if (glType == GL_FLOAT) { float f[4]; std::memcpy(f, value, 4 * sizeof(float)); for (int i = 0; i < 4; i++) c[i] = f[i]; }
            else if (glType == GL_UNSIGNED_INT) { uint32_t u; std::memcpy(&u, value, 4); c[0] = u; }
            else if (glType == GL_INT) { int32_t u; std::memcpy(&u, value, 4); c[0] = u; }
            else if (glType == GL_UNSIGNED_BYTE) { const uint8_t* b = (const uint8_t*)value; for (int i = 0; i < 4; i++) c[i] = b[i] / 255.0; }
        }
        auto* a = pass.colorAttachments[0];
        a.texture = t->tex; a.level = lvl; a.loadAction = MTLLoadActionClear; a.storeAction = MTLStoreActionStore;
        a.clearColor = MTLClearColorMake(c[0], c[1], c[2], c[3]);
    }
    auto enc = [g->cb renderCommandEncoderWithDescriptor:pass];
    [enc endEncoding];
}

void textureRead(uint32_t id, uint32_t lvl, uint32_t glFormat, uint32_t glType, void* dst) {
    auto* t = find(g->textures, id);
    if (!t) return;
    uint32_t w = MAX(t->w >> lvl, 1u), h = MAX(t->h >> lvl, 1u);
    uint32_t bpp = clientBpp(glFormat, glType);
    uint64_t rowBytes = (uint64_t)w * bpp;
    id<MTLBuffer> stage = [g->dev newBufferWithLength:rowBytes * h options:MTLResourceStorageModeShared];
    beginBlit();
    [g->be copyFromTexture:t->tex sourceSlice:0 sourceLevel:lvl sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                  toBuffer:stage destinationOffset:0 destinationBytesPerRow:rowBytes destinationBytesPerImage:rowBytes * h];
    finishCB();
    std::memcpy(dst, stage.contents, rowBytes * h);
}

// ---------------------------------------------------------------------------------------------
// samplers / pipelines / depth stencil
// ---------------------------------------------------------------------------------------------

id<MTLSamplerState> samplerState(const SamplerParams& p) {
    std::string key((const char*)&p, sizeof(SamplerParams));
    auto it = g->samplerCache.find(key);
    if (it != g->samplerCache.end()) return it->second;

    MTLSamplerDescriptor* d = [MTLSamplerDescriptor new];
    bool minLinear = p.minFilter == GL_LINEAR || p.minFilter == GL_LINEAR_MIPMAP_LINEAR || p.minFilter == GL_LINEAR_MIPMAP_NEAREST;
    bool mipmapped = p.minFilter != GL_NEAREST && p.minFilter != GL_LINEAR;
    bool mipLinear = p.minFilter == GL_NEAREST_MIPMAP_LINEAR || p.minFilter == GL_LINEAR_MIPMAP_LINEAR;
    d.minFilter = minLinear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.magFilter = p.magFilter == GL_LINEAR ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.mipFilter = !mipmapped ? MTLSamplerMipFilterNotMipmapped : (mipLinear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest);
    d.sAddressMode = mapWrap(p.wrapS);
    d.tAddressMode = mapWrap(p.wrapT);
    d.lodMinClamp = MAX(p.minLod, 0.0f);
    d.lodMaxClamp = mipmapped ? MIN(p.maxLod, 1000.0f) : 0.0f;
    d.supportArgumentBuffers = NO;
    auto s = [g->dev newSamplerStateWithDescriptor:d];
    g->samplerCache[key] = s;
    return s;
}

id<MTLDepthStencilState> depthStencilState() {
    const State& s = g->st;
    struct K { uint32_t depthTest, func, mask, stencil, sfunc, smask, swrite, sfail, dpfail, dppass; } k{};
    k.depthTest = s.depthTest; k.func = s.depthFunc; k.mask = s.depthMask;
    k.stencil = s.stencilTest; k.sfunc = s.stFunc; k.smask = s.stMask; k.swrite = s.stWriteMask;
    k.sfail = s.sFail; k.dpfail = s.dpFail; k.dppass = s.dpPass;
    std::string key((const char*)&k, sizeof(k));
    auto it = g->dssCache.find(key);
    if (it != g->dssCache.end()) return it->second;

    MTLDepthStencilDescriptor* d = [MTLDepthStencilDescriptor new];
    d.depthCompareFunction = s.depthTest ? mapCompare(s.depthFunc) : MTLCompareFunctionAlways;
    // GL writes depth only when the depth test is enabled
    d.depthWriteEnabled = s.depthTest && s.depthMask;
    if (s.stencilTest) {
        MTLStencilDescriptor* sd = [MTLStencilDescriptor new];
        sd.stencilCompareFunction = mapCompare(s.stFunc);
        sd.stencilFailureOperation = mapStencilOp(s.sFail);
        sd.depthFailureOperation = mapStencilOp(s.dpFail);
        sd.depthStencilPassOperation = mapStencilOp(s.dpPass);
        sd.readMask = s.stMask;
        sd.writeMask = s.stWriteMask;
        d.frontFaceStencil = sd;
        d.backFaceStencil = sd;
    }
    auto st = [g->dev newDepthStencilStateWithDescriptor:d];
    g->dssCache[key] = st;
    return st;
}

id<MTLRenderPipelineState> pipelineFor(Program& p, uint32_t progId, Framebuffer& fb) {
    struct Key {
        uint32_t prog;
        uint32_t cf[8];
        uint32_t df, sf;
        uint8_t blend[8];
        uint32_t f[8][4];
        uint8_t mask[8];
    } k;
    std::memset(&k, 0, sizeof(k));
    k.prog = progId;
    for (int i = 0; i < 8; i++) {
        auto* t = find(g->textures, fb.colour[i].tex);
        k.cf[i] = t ? (uint32_t)t->pf : 0;
        k.blend[i] = g->st.blendEn[i];
        k.f[i][0] = g->st.bSrcRGB[i]; k.f[i][1] = g->st.bDstRGB[i];
        k.f[i][2] = g->st.bSrcA[i];   k.f[i][3] = g->st.bDstA[i];
        uint8_t m = 0;
        if ((fb.drawMask >> i) & 1) {
            for (int c = 0; c < 4; c++) if (g->st.colourMask[i][c]) m |= (1 << c);
        }
        k.mask[i] = m;
    }
    uint32_t depthPf = 0;
    if (auto* t = find(g->textures, fb.depth.tex)) {
        depthPf = (uint32_t)t->pf;
        k.df = depthPf;
        k.sf = (fb.stencilAttached && t->hasStencil) ? (uint32_t)MTLPixelFormatDepth32Float_Stencil8 : 0;
    }
    std::string key((const char*)&k, sizeof(k));
    auto it = p.psos.find(key);
    if (it != p.psos.end()) return it->second;

    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = p.vs.fn;
    d.fragmentFunction = p.fs.fn;
    for (int i = 0; i < 8; i++) {
        if (!k.cf[i]) continue;
        auto* a = d.colorAttachments[i];
        a.pixelFormat = (MTLPixelFormat)k.cf[i];
        MTLColorWriteMask wm = MTLColorWriteMaskNone;
        if (k.mask[i] & 1) wm |= MTLColorWriteMaskRed;
        if (k.mask[i] & 2) wm |= MTLColorWriteMaskGreen;
        if (k.mask[i] & 4) wm |= MTLColorWriteMaskBlue;
        if (k.mask[i] & 8) wm |= MTLColorWriteMaskAlpha;
        a.writeMask = wm;
        if (k.blend[i]) {
            a.blendingEnabled = YES;
            a.sourceRGBBlendFactor = mapBlend(k.f[i][0]);
            a.destinationRGBBlendFactor = mapBlend(k.f[i][1]);
            a.sourceAlphaBlendFactor = mapBlend(k.f[i][2]);
            a.destinationAlphaBlendFactor = mapBlend(k.f[i][3]);
        }
    }
    d.depthAttachmentPixelFormat = (MTLPixelFormat)k.df;
    d.stencilAttachmentPixelFormat = (MTLPixelFormat)k.sf;
    NSError* err = nil;
    auto pso = [g->dev newRenderPipelineStateWithDescriptor:d error:&err];
    if (!pso) {
        NSLog(@"[voxymtl] render pipeline creation failed: %@", err);
        return nil;
    }
    p.psos[key] = pso;
    return pso;
}

// ---------------------------------------------------------------------------------------------
// resource binding
// ---------------------------------------------------------------------------------------------

enum class Stage { Vertex, Fragment, Compute };

void bindBuffer(Stage s, id<MTLBuffer> b, uint64_t off, int idx) {
    switch (s) {
        case Stage::Vertex: [g->re setVertexBuffer:b offset:off atIndex:idx]; break;
        case Stage::Fragment: [g->re setFragmentBuffer:b offset:off atIndex:idx]; break;
        case Stage::Compute: [g->ce setBuffer:b offset:off atIndex:idx]; break;
    }
}
void bindBytes(Stage s, const void* d, uint32_t n, int idx) {
    switch (s) {
        case Stage::Vertex: [g->re setVertexBytes:d length:n atIndex:idx]; break;
        case Stage::Fragment: [g->re setFragmentBytes:d length:n atIndex:idx]; break;
        case Stage::Compute: [g->ce setBytes:d length:n atIndex:idx]; break;
    }
}
void bindTexture(Stage s, id<MTLTexture> t, int idx) {
    switch (s) {
        case Stage::Vertex: [g->re setVertexTexture:t atIndex:idx]; break;
        case Stage::Fragment: [g->re setFragmentTexture:t atIndex:idx]; break;
        case Stage::Compute: [g->ce setTexture:t atIndex:idx]; break;
    }
}
void bindSampler(Stage s, id<MTLSamplerState> t, int idx) {
    switch (s) {
        case Stage::Vertex: [g->re setVertexSamplerState:t atIndex:idx]; break;
        case Stage::Fragment: [g->re setFragmentSamplerState:t atIndex:idx]; break;
        case Stage::Compute: [g->ce setSamplerState:t atIndex:idx]; break;
    }
}

void bindStage(Stage s, const StageInfo& info, Program& prog) {
    for (const auto& r : info.res) {
        switch (r.kind) {
            case ResKind::Ubo:
            case ResKind::Ssbo: {
                const BufBind& bb = r.kind == ResKind::Ubo ? g->st.ubo[r.glBinding] : g->st.ssbo[r.glBinding];
                bindBuffer(s, bufOf(bb.buf), (uint64_t)bb.off, r.mslIndex);
                break;
            }
            case ResKind::Loose: {
                uint32_t n = (info.looseBytes + 15u) & ~15u;
                bindBytes(s, prog.uniforms, MIN(n, (uint32_t)sizeof(prog.uniforms)), r.mslIndex);
                break;
            }
            case ResKind::Texture: {
                uint32_t unit = (uint32_t)r.glBinding;
                auto* t = find(g->textures, g->st.texUnit[unit]);
                bindTexture(s, t ? sampledView(*t) : nil, r.mslIndex);
                SamplerParams params = t ? t->params : SamplerParams();
                if (auto* so = find(g->samplers, g->st.sampUnit[unit])) params = so->params;
                bindSampler(s, samplerState(params), r.mslIndex);
                break;
            }
            case ResKind::Image: {
                const ImgBind& ib = g->st.img[r.glBinding];
                auto* t = find(g->textures, ib.tex);
                bindTexture(s, t ? viewOfLevel(*t, (uint32_t)ib.level) : nil, r.mslIndex);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// draws
// ---------------------------------------------------------------------------------------------

// 16 bit version of an 8 bit index buffer (kept up to date lazily)
id<MTLBuffer> widenedIndices(uint32_t srcId) {
    auto* sb = find(g->buffers, srcId);
    if (!sb) return nil;
    auto& w = g->widened[srcId];
    if (w.buf && w.version == sb->version) return w.buf;
    if (!w.buf) w.buf = [g->dev newBufferWithLength:sb->size * 2 options:MTLResourceStorageModePrivate];
    beginCompute();
    [g->ce setComputePipelineState:g->kWiden];
    [g->ce setBuffer:sb->buf offset:0 atIndex:0];
    [g->ce setBuffer:w.buf offset:0 atIndex:1];
    uint32_t n = (uint32_t)sb->size;
    [g->ce setBytes:&n length:4 atIndex:2];
    dispatch1D(g->kWiden, n);
    w.version = sb->version;
    return w.buf;
}

id<MTLBuffer> fanIndices(uint32_t vertexCount) {
    auto it = g->fanIndex.find(vertexCount);
    if (it != g->fanIndex.end()) return it->second;
    std::vector<uint16_t> idx;
    for (uint32_t i = 1; i + 1 < vertexCount; i++) { idx.push_back(0); idx.push_back(i); idx.push_back(i + 1); }
    auto b = [g->dev newBufferWithBytes:idx.data() length:idx.size() * 2 options:MTLResourceStorageModeShared];
    g->fanIndex[vertexCount] = b;
    return b;
}

// Opens the render encoder and applies all pipeline state / resource bindings
bool setupGraphicsDraw() {
    uint32_t progId = g->st.prog;
    Program* p = findProg(progId);
    if (!p || p->compute) return false;
    auto* fb = find(g->fbs, g->st.fb);
    if (!fb) return false;
    if (!beginRender(g->st.fb)) return false;

    auto pso = pipelineFor(*p, progId, *fb);
    if (!pso) return false;
    [g->re setRenderPipelineState:pso];
    [g->re setDepthStencilState:depthStencilState()];
    [g->re setStencilReferenceValue:(uint32_t)g->st.stRef];
    MTLCullMode cm = MTLCullModeNone;
    if (g->st.cull) cm = g->st.cullFace == GL_FRONT ? MTLCullModeFront : MTLCullModeBack;
    [g->re setCullMode:cm];
    // the vertex y flip inverts the winding
    [g->re setFrontFacingWinding:g->st.frontCW ? MTLWindingCounterClockwise : MTLWindingClockwise];
    [g->re setTriangleFillMode:g->st.polyLine ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];

    // render target size
    uint32_t rtW = 0, rtH = 0;
    {
        Attachment* a = nullptr;
        for (int i = 0; i < 8 && !a; i++) if (fb->colour[i].tex) a = &fb->colour[i];
        if (!a && fb->depth.tex) a = &fb->depth;
        if (a) if (auto* t = find(g->textures, a->tex)) { rtW = MAX(t->w >> a->level, 1u); rtH = MAX(t->h >> a->level, 1u); }
    }
    int vx = g->st.vp[0], vy = g->st.vp[1], vw = g->st.vp[2], vh = g->st.vp[3];
    if (!g->st.vpSet || vx < 0 || vy < 0 || vx + vw > (int)rtW || vy + vh > (int)rtH || vw <= 0 || vh <= 0) {
        vx = vy = 0; vw = (int)rtW; vh = (int)rtH;
    }
    MTLViewport v = {(double)vx, (double)vy, (double)vw, (double)vh, 0.0, 1.0};
    [g->re setViewport:v];
    MTLScissorRect sr = {0, 0, rtW, rtH};
    [g->re setScissorRect:sr];

    bindStage(Stage::Vertex, p->vs, *p);
    bindStage(Stage::Fragment, p->fs, *p);
    return true;
}

bool resolveIndex(uint32_t glType, uint32_t& byteMul, MTLIndexType& it, id<MTLBuffer>& buf) {
    auto* eb = find(g->buffers, g->st.elem);
    if (!eb) return false;
    if (glType == GL_UNSIGNED_BYTE) {
        buf = widenedIndices(g->st.elem);
        byteMul = 2; it = MTLIndexTypeUInt16;
    } else {
        buf = eb->buf;
        byteMul = 1; it = glType == GL_UNSIGNED_INT ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
    }
    return buf != nil;
}

void drawArrays(uint32_t mode, uint32_t first, uint32_t count, uint32_t instances) {
    if (mode == GL_TRIANGLE_FAN) {
        id<MTLBuffer> idx = fanIndices(count);
        if (!setupGraphicsDraw()) return;
        [g->re drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:(count - 2) * 3 indexType:MTLIndexTypeUInt16
                         indexBuffer:idx indexBufferOffset:0 instanceCount:instances baseVertex:first baseInstance:0];
        return;
    }
    if (!setupGraphicsDraw()) return;
    [g->re drawPrimitives:mapPrim(mode) vertexStart:first vertexCount:count instanceCount:instances baseInstance:0];
}

void drawElements(uint32_t mode, uint32_t count, uint32_t type, uint64_t byteOffset, uint32_t instances, uint32_t baseInstance) {
    uint32_t mul; MTLIndexType it; id<MTLBuffer> ib;
    if (!resolveIndex(type, mul, it, ib)) return; // may open a compute encoder, so do before the render setup
    if (!setupGraphicsDraw()) return;
    [g->re drawIndexedPrimitives:mapPrim(mode) indexCount:count indexType:it indexBuffer:ib
               indexBufferOffset:byteOffset * mul instanceCount:instances baseVertex:0 baseInstance:baseInstance];
}

void drawIndirect(uint32_t mode, uint32_t type, uint64_t byteOffset) {
    uint32_t mul; MTLIndexType it; id<MTLBuffer> ib;
    if (!resolveIndex(type, mul, it, ib)) return;
    id<MTLBuffer> ind = bufOf(g->st.indirect);
    if (!ind) return;
    if (!setupGraphicsDraw()) return;
    // DrawElementsIndirectCommand has the same layout as MTLDrawIndexedPrimitivesIndirectArguments
    [g->re drawIndexedPrimitives:mapPrim(mode) indexType:it indexBuffer:ib indexBufferOffset:0
                  indirectBuffer:ind indirectBufferOffset:byteOffset];
}

IcbEntry* icbFor(uint32_t maxCount) {
    uint32_t cap = 1024;
    while (cap < maxCount) cap <<= 1;
    auto it = g->icbs.find(cap);
    if (it != g->icbs.end()) return &it->second;
    MTLIndirectCommandBufferDescriptor* d = [MTLIndirectCommandBufferDescriptor new];
    d.commandTypes = MTLIndirectCommandTypeDrawIndexed;
    d.inheritBuffers = YES;
    d.inheritPipelineState = YES;
    d.maxVertexBufferBindCount = 31;
    d.maxFragmentBufferBindCount = 31;
    IcbEntry e;
    e.capacity = cap;
    e.icb = [g->dev newIndirectCommandBufferWithDescriptor:d maxCommandCount:cap options:MTLResourceStorageModePrivate];
    e.argBuf = [g->dev newBufferWithLength:g->mdicArgEnc.encodedLength options:MTLResourceStorageModeShared];
    [g->mdicArgEnc setArgumentBuffer:e.argBuf offset:0];
    [g->mdicArgEnc setIndirectCommandBuffer:e.icb atIndex:0];
    e.range = [g->dev newBufferWithLength:16 options:MTLResourceStorageModePrivate];
    return &(g->icbs[cap] = e);
}

void multiDrawIndirectCount(uint32_t mode, uint32_t type, uint64_t indirectOffset, uint64_t countOffset,
                            uint32_t maxCount, uint32_t stride) {
    if (maxCount == 0) return;
    uint32_t mul; MTLIndexType it; id<MTLBuffer> ib;
    if (!resolveIndex(type, mul, it, ib) || it != MTLIndexTypeUInt16) return;
    id<MTLBuffer> ind = bufOf(g->st.indirect);
    id<MTLBuffer> cnt = bufOf(g->st.param);
    if (!ind || !cnt) return;
    uint32_t strideWords = stride ? stride / 4 : 5;

    if (g->icbSupported) {
        IcbEntry* e = icbFor(maxCount);
        // 1. compute pass translates the GPU draw list into an indirect command buffer
        beginCompute();
        [g->ce setComputePipelineState:g->kMdic];
        [g->ce setBuffer:ind offset:0 atIndex:0];
        [g->ce setBuffer:cnt offset:0 atIndex:1];
        uint32_t params[4] = {(uint32_t)(indirectOffset / 4), strideWords, (uint32_t)(countOffset / 4), maxCount};
        [g->ce setBytes:params length:sizeof(params) atIndex:2];
        [g->ce setBuffer:ib offset:0 atIndex:3];
        [g->ce setBuffer:e->argBuf offset:0 atIndex:4];
        [g->ce setBuffer:e->range offset:0 atIndex:5];
        [g->ce useResource:e->icb usage:MTLResourceUsageWrite];
        dispatch1D(g->kMdic, maxCount);

        // 2. the render pass executes it, pipeline state and buffer bindings are inherited
        if (!setupGraphicsDraw()) return;
        [g->re useResource:ib usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
        [g->re useResource:e->icb usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
        [g->re executeCommandsInBuffer:e->icb indirectBuffer:e->range indirectBufferOffset:0];
        return;
    }

    // fallback: read the count back on the CPU and issue one indirect draw per command
    uint32_t count = 0;
    bufferRead(g->st.param, countOffset, 4, &count);
    count = MIN(count, maxCount);
    if (!setupGraphicsDraw()) return;
    for (uint32_t i = 0; i < count; i++) {
        [g->re drawIndexedPrimitives:mapPrim(mode) indexType:it indexBuffer:ib indexBufferOffset:0
                      indirectBuffer:ind indirectBufferOffset:indirectOffset + (uint64_t)i * strideWords * 4];
    }
}

bool setupCompute() {
    Program* p = findProg(g->st.prog);
    if (!p || !p->compute) return false;
    beginCompute();
    [g->ce setComputePipelineState:p->cpso];
    bindStage(Stage::Compute, p->cs, *p);
    return true;
}

void dispatchCompute(uint32_t x, uint32_t y, uint32_t z) {
    if (!setupCompute()) return;
    Program* p = findProg(g->st.prog);
    [g->ce dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(p->cs.local[0], p->cs.local[1], p->cs.local[2])];
}

void dispatchComputeIndirect(uint64_t off) {
    id<MTLBuffer> ind = bufOf(g->st.dispatch);
    if (!ind || !setupCompute()) return;
    Program* p = findProg(g->st.prog);
    [g->ce dispatchThreadgroupsWithIndirectBuffer:ind indirectBufferOffset:off
                            threadsPerThreadgroup:MTLSizeMake(p->cs.local[0], p->cs.local[1], p->cs.local[2])];
}

// ---------------------------------------------------------------------------------------------
// programs
// ---------------------------------------------------------------------------------------------

struct StageSrcArg { int64_t type; int64_t src; int64_t len; };

void writeLog(int64_t logPtr, int64_t cap, const std::string& s) {
    if (!logPtr || cap <= 0) return;
    size_t n = MIN((size_t)(cap - 1), s.size());
    std::memcpy((void*)logPtr, s.data(), n);
    ((char*)logPtr)[n] = 0;
}

bool buildStage(StageKind kind, const std::string& glsl, StageInfo& info, std::string& log) {
    CompiledStage cs;
    if (!compileStage(kind, glsl, cs, log)) return false;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion2_4;
    NSError* err = nil;
    id<MTLLibrary> lib = [g->dev newLibraryWithSource:[NSString stringWithUTF8String:cs.msl.c_str()] options:opts error:&err];
    if (!lib) {
        log += std::string("MSL compile failed: ") + (err ? err.localizedDescription.UTF8String : "?") + "\n" + cs.msl;
        return false;
    }
    info.fn = [lib newFunctionWithName:[NSString stringWithUTF8String:cs.entry.c_str()]];
    if (!info.fn) { log += "entry point missing: " + cs.entry; return false; }
    info.res = cs.resources;
    info.looseBytes = cs.looseBytes;
    for (int i = 0; i < 3; i++) info.local[i] = cs.localSize[i];
    return true;
}

uint32_t createProgram(int64_t n, const StageSrcArg* stages, int64_t logPtr, int64_t logCap) {
    auto prog = std::make_unique<Program>();
    std::string log;
    bool ok = true;
    for (int64_t i = 0; i < n && ok; i++) {
        std::string src((const char*)stages[i].src, (size_t)stages[i].len);
        switch (stages[i].type) {
            case 0: ok = buildStage(StageKind::Vertex, src, prog->vs, log); break;
            case 1: ok = buildStage(StageKind::Fragment, src, prog->fs, log); break;
            case 2: ok = buildStage(StageKind::Compute, src, prog->cs, log); prog->compute = true; break;
            default: log += "unsupported shader stage"; ok = false;
        }
    }
    if (ok && prog->compute) {
        NSError* err = nil;
        MTLComputePipelineDescriptor* cd = [MTLComputePipelineDescriptor new];
        cd.computeFunction = prog->cs.fn;
        cd.label = @"voxy compute";
        prog->cpso = [g->dev newComputePipelineStateWithDescriptor:cd options:MTLPipelineOptionNone reflection:nil error:&err];
        if (!prog->cpso) { log += std::string("compute pipeline failed: ") + (err ? err.localizedDescription.UTF8String : "?"); ok = false; }
    } else if (ok && !prog->vs.fn) {
        log += "graphics program has no vertex shader";
        ok = false;
    }
    writeLog(logPtr, logCap, log);
    if (!ok) return 0;
    uint32_t id = newId();
    g->programs[id] = std::move(prog);
    return id;
}

// ---------------------------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------------------------

bool init(int64_t* caps) {
    if (g) return true;
    g = new Ctx();
    g->dev = MTLCreateSystemDefaultDevice();
    if (!g->dev) { delete g; g = nullptr; return false; }
    g->queue = [g->dev newCommandQueue];
    g->unified = g->dev.hasUnifiedMemory;
    g->icbSupported = [g->dev supportsFamily:MTLGPUFamilyApple4] || [g->dev supportsFamily:MTLGPUFamilyMac2];

    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion2_4;
    g->kernels = [g->dev newLibraryWithSource:[NSString stringWithUTF8String:KERNEL_SOURCE] options:opts error:&err];
    if (!g->kernels) {
        NSLog(@"[voxymtl] kernel library failed: %@", err);
        delete g; g = nullptr;
        return false;
    }
    auto mk = [&](NSString* name) {
        id<MTLFunction> f = [g->kernels newFunctionWithName:name];
        NSError* e = nil;
        return [g->dev newComputePipelineStateWithFunction:f error:&e];
    };
    g->kFill32 = mk(@"vmtl_fill32");
    g->kWiden = mk(@"vmtl_widen_u8");
    g->kDepth = mk(@"vmtl_depth_to_r32f");
    if (g->icbSupported) {
        g->fMdic = [g->kernels newFunctionWithName:@"vmtl_mdic_encode"];
        NSError* e = nil;
        g->kMdic = [g->dev newComputePipelineStateWithFunction:g->fMdic error:&e];
        g->mdicArgEnc = [g->fMdic newArgumentEncoderWithBufferIndex:4];
        if (!g->kMdic || !g->mdicArgEnc) g->icbSupported = false;
    }

    if (caps) {
        caps[0] = 1;
        caps[1] = (int64_t)g->dev.maxBufferLength;
        caps[2] = 256; // buffer binding alignment (constant address space on macOS)
        caps[3] = [g->dev supportsFamily:MTLGPUFamilyApple7]; // simd width of 32, required by the subgroup shaders
        caps[4] = 1;
        caps[5] = g->icbSupported;
        caps[6] = [g->dev supportsFamily:MTLGPUFamilyApple1];
        caps[7] = g->unified;
        caps[8] = (int64_t)g->dev.recommendedMaxWorkingSetSize;
        caps[9] = g->dev.lowPower;
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------------------------

extern "C" __attribute__((visibility("default"))) int64_t vmtl_call(int64_t* a) {
    @autoreleasepool {
        const int64_t op = a[0];
        if (op == OP_INIT) return init((int64_t*)a[1]) ? 1 : 0;
        if (!g) return 0;
        State& st = g->st;

        switch (op) {
            case OP_SHUTDOWN: {
                finishCB();
                delete g; g = nullptr;
                return 1;
            }

            // ---- buffers
            case OP_BUF_CREATE: {
                bool mapped = a[2] != 0;
                Buffer b;
                b.size = (uint64_t)a[1];
                b.cpuVisible = mapped || g->unified;
                MTLResourceOptions o = b.cpuVisible ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
                b.buf = [g->dev newBufferWithLength:MAX(b.size, (uint64_t)16) options:o];
                if (!b.buf) return 0;
                if (b.cpuVisible) std::memset(b.buf.contents, 0, b.buf.length);
                uint32_t id = newId();
                g->buffers.emplace(id, std::move(b));
                return id;
            }
            case OP_BUF_DELETE: g->buffers.erase((uint32_t)a[1]); g->widened.erase((uint32_t)a[1]); return 1;
            case OP_BUF_PTR: {
                auto* b = find(g->buffers, (uint32_t)a[1]);
                return b && b->cpuVisible ? (int64_t)b->buf.contents : 0;
            }
            case OP_BUF_COPY: bufferCopy((uint32_t)a[1], (uint32_t)a[2], a[3], a[4], a[5]); return 1;
            case OP_BUF_FILL32: bufferFill32((uint32_t)a[1], a[2], a[3], (uint32_t)a[4]); return 1;
            case OP_BUF_READ: bufferRead((uint32_t)a[1], a[2], a[3], (void*)a[4]); return 1;
            case OP_BUF_SUBDATA: bufferSubData((uint32_t)a[1], a[2], a[3], (const void*)a[4]); return 1;

            // ---- textures
            case OP_TEX_CREATE: return createTexture((uint32_t)a[1], (uint32_t)a[2], (uint32_t)a[3], (uint32_t)a[4], a[5] != 0, (uint32_t*)a[6]);
            case OP_TEX_DELETE: {
                uint32_t id = (uint32_t)a[1];
                if (auto* t = find(g->textures, id)) {
                    if (t->glName) { GLuint n = t->glName; glDeleteTextures(1, &n); }
                    if (t->surface) CFRelease(t->surface);
                }
                g->textures.erase(id);
                return 1;
            }
            case OP_TEX_SUBIMAGE: textureSubImage((uint32_t)a[1], (uint32_t)a[2], (uint32_t)a[3], (uint32_t)a[4], (uint32_t)a[5], (uint32_t)a[6], (uint32_t)a[7], (uint32_t)a[8], (const void*)a[9]); return 1;
            case OP_TEX_CLEAR: textureClear((uint32_t)a[1], (uint32_t)a[2], (uint32_t)a[3], (uint32_t)a[4], (const void*)a[5]); return 1;
            case OP_TEX_GENMIPS: {
                auto* t = find(g->textures, (uint32_t)a[1]);
                if (t) { beginBlit(); [g->be generateMipmapsForTexture:t->tex]; }
                return 1;
            }
            case OP_TEX_PARAM: {
                if (auto* t = find(g->textures, (uint32_t)a[1])) {
                    uint32_t pname = (uint32_t)a[2];
                    if (pname == 0x813C) t->baseLevel = (uint32_t)MAX((int64_t)0, a[3]); // GL_TEXTURE_BASE_LEVEL
                    else if (pname == 0x813D) t->maxLevel = (uint32_t)MAX((int64_t)0, a[3]); // GL_TEXTURE_MAX_LEVEL
                    else applyParam(t->params, pname, a[3], a[4] != 0);
                }
                return 1;
            }
            case OP_TEX_READ: textureRead((uint32_t)a[1], (uint32_t)a[2], (uint32_t)a[3], (uint32_t)a[4], (void*)a[5]); return 1;
            case OP_TEX_VIEW: {
                auto* src = find(g->textures, (uint32_t)a[1]);
                if (!src) return 0;
                Texture t;
                t.pf = mapFormat((uint32_t)a[2], t.hasDepth, t.hasStencil);
                if (t.pf == MTLPixelFormatInvalid) return 0;
                uint32_t base = (uint32_t)a[3], n = (uint32_t)a[4];
                t.tex = [src->tex newTextureViewWithPixelFormat:t.pf textureType:MTLTextureType2D
                                                         levels:NSMakeRange(base, n) slices:NSMakeRange(0, 1)];
                if (!t.tex) return 0;
                t.glFormat = (uint32_t)a[2];
                t.w = MAX(src->w >> base, 1u); t.h = MAX(src->h >> base, 1u); t.levels = n;
                t.params = src->params;
                uint32_t id = newId();
                g->textures.emplace(id, std::move(t));
                return id;
            }
            case OP_TEX_DEPTH_TO_R32F: {
                auto* s = find(g->textures, (uint32_t)a[1]);
                auto* d = find(g->textures, (uint32_t)a[2]);
                if (!s || !d) return 0;
                beginCompute();
                [g->ce setComputePipelineState:g->kDepth];
                [g->ce setTexture:s->tex atIndex:0];
                [g->ce setTexture:d->tex atIndex:1];
                [g->ce dispatchThreads:MTLSizeMake(d->w, d->h, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                return 1;
            }
            case OP_TEX_COPY: {
                auto* s = find(g->textures, (uint32_t)a[1]);
                auto* d = find(g->textures, (uint32_t)a[2]);
                if (!s || !d) return 0;
                uint32_t lvl = (uint32_t)a[3];
                beginBlit();
                [g->be copyFromTexture:s->tex sourceSlice:0 sourceLevel:lvl sourceOrigin:MTLOriginMake(0, 0, 0)
                            sourceSize:MTLSizeMake(MAX(s->w >> lvl, 1u), MAX(s->h >> lvl, 1u), 1)
                             toTexture:d->tex destinationSlice:0 destinationLevel:lvl destinationOrigin:MTLOriginMake(0, 0, 0)];
                return 1;
            }

            // ---- samplers
            case OP_SAMP_CREATE: { uint32_t id = newId(); g->samplers.emplace(id, Sampler{}); return id; }
            case OP_SAMP_DELETE: g->samplers.erase((uint32_t)a[1]); return 1;
            case OP_SAMP_PARAM: {
                if (auto* s = find(g->samplers, (uint32_t)a[1])) applyParam(s->params, (uint32_t)a[2], a[3], a[4] != 0);
                return 1;
            }

            // ---- programs
            case OP_PROG_CREATE: return createProgram(a[1], (const StageSrcArg*)a[2], a[3], a[4]);
            case OP_PROG_DELETE: g->programs.erase((uint32_t)a[1]); return 1;
            case OP_UNIFORM: {
                if (auto* p = findProg((uint32_t)a[1])) {
                    int64_t loc = a[2], n = a[4];
                    if (loc >= 0 && loc < MAX_LOOSE_LOCATIONS && n <= LOOSE_SLOT_STRIDE)
                        std::memcpy(p->uniforms + loc * LOOSE_SLOT_STRIDE, (const void*)a[3], (size_t)n);
                }
                return 1;
            }

            // ---- bindings
            case OP_USE_PROGRAM: st.prog = (uint32_t)a[1]; return 1;
            case OP_BIND_BASE: {
                BufBind bb{(uint32_t)a[3], a[4], a[5]};
                int idx = (int)a[2];
                if (a[1] == GLC_UNIFORM_BUFFER && idx < 8) st.ubo[idx] = bb;
                else if (a[1] == GLC_SHADER_STORAGE_BUFFER && idx < 24) st.ssbo[idx] = bb;
                return 1;
            }
            case OP_BIND_TEXUNIT: if (a[1] < 16) st.texUnit[a[1]] = (uint32_t)a[2]; return 1;
            case OP_BIND_SAMPLER: if (a[1] < 16) st.sampUnit[a[1]] = (uint32_t)a[2]; return 1;
            case OP_BIND_IMAGE: if (a[1] < 16) st.img[a[1]] = {(uint32_t)a[2], (int)a[3]}; return 1;
            case OP_BIND_FB: st.fb = (uint32_t)a[1]; return 1;
            case OP_BIND_ELEMENT: st.elem = (uint32_t)a[1]; return 1;
            case OP_BIND_INDIRECT: st.indirect = (uint32_t)a[1]; return 1;
            case OP_BIND_PARAMETER: st.param = (uint32_t)a[1]; return 1;
            case OP_BIND_DISPATCH: st.dispatch = (uint32_t)a[1]; return 1;

            // ---- framebuffers
            case OP_FB_CREATE: { uint32_t id = newId(); g->fbs.emplace(id, Framebuffer{}); return id; }
            case OP_FB_DELETE: g->fbs.erase((uint32_t)a[1]); g->pendingClearFbs.erase((uint32_t)a[1]); return 1;
            case OP_FB_ATTACH: {
                auto* fb = find(g->fbs, (uint32_t)a[1]);
                if (!fb) return 0;
                uint32_t att = (uint32_t)a[2];
                Attachment at{(uint32_t)a[3], (uint32_t)a[4]};
                if (att >= GLC_COLOR_ATTACHMENT0 && att < GLC_COLOR_ATTACHMENT0 + 8) fb->colour[att - GLC_COLOR_ATTACHMENT0] = at;
                else if (att == GLC_DEPTH_ATTACHMENT) { fb->depth = at; fb->stencilAttached = false; }
                else if (att == GLC_DEPTH_STENCIL_ATTACHMENT) { fb->depth = at; fb->stencilAttached = at.tex != 0; }
                else if (att == GLC_STENCIL_ATTACHMENT) { fb->depth = at; fb->stencilAttached = at.tex != 0; }
                fb->sig++;
                return 1;
            }
            case OP_FB_DRAWBUFFERS: {
                if (auto* fb = find(g->fbs, (uint32_t)a[1])) { fb->drawMask = (uint32_t)a[2]; fb->sig++; }
                return 1;
            }
            case OP_FB_CLEAR: {
                uint32_t id = (uint32_t)a[1];
                auto* fb = find(g->fbs, id);
                if (!fb) return 0;
                auto f = [&](int i) { uint32_t b = (uint32_t)a[4 + i]; float v; std::memcpy(&v, &b, 4); return (double)v; };
                int kind = (int)a[2];
                int buf = (int)a[3];
                if (kind == 0 && buf >= 0 && buf < 8) {
                    fb->clearColour[buf] = true;
                    for (int i = 0; i < 4; i++) fb->clearColourValue[buf][i] = f(i);
                } else {
                    if (kind == 1 || kind == 3) { fb->clearDepth = true; fb->clearDepthValue = f(0); }
                    if (kind == 2) { fb->clearStencil = true; fb->clearStencilValue = (uint32_t)a[4]; }
                    if (kind == 3) { fb->clearStencil = true; fb->clearStencilValue = (uint32_t)a[5]; }
                }
                // an already open pass on this fb is ended so the clear is applied as a load action
                if (g->re && g->reFb == id) endEncoders();
                g->pendingClearFbs.insert(id);
                return 1;
            }

            // ---- fixed function state
            case OP_SET_CAP: {
                bool on = a[2] != 0;
                switch (a[1]) {
                    case GL_DEPTH_TEST: st.depthTest = on; break;
                    case GL_STENCIL_TEST: st.stencilTest = on; break;
                    case GL_CULL_FACE: st.cull = on; break;
                    case GL_BLEND: for (int i = 0; i < 8; i++) st.blendEn[i] = on; break;
                    default: break;
                }
                return 1;
            }
            case OP_DEPTH_FUNC: st.depthFunc = (uint32_t)a[1]; return 1;
            case OP_DEPTH_MASK: st.depthMask = a[1] != 0; return 1;
            case OP_COLOR_MASK: {
                int from = a[1] < 0 ? 0 : (int)a[1], to = a[1] < 0 ? 8 : (int)a[1] + 1;
                for (int i = from; i < to && i < 8; i++) for (int c = 0; c < 4; c++) st.colourMask[i][c] = a[2 + c] != 0;
                return 1;
            }
            case OP_STENCIL_FUNC: st.stFunc = (uint32_t)a[1]; st.stRef = (int)a[2]; st.stMask = (uint32_t)a[3]; return 1;
            case OP_STENCIL_OP: st.sFail = (uint32_t)a[1]; st.dpFail = (uint32_t)a[2]; st.dpPass = (uint32_t)a[3]; return 1;
            case OP_STENCIL_MASK: st.stWriteMask = (uint32_t)a[1]; return 1;
            case OP_BLEND_SEP: {
                int from = a[1] < 0 ? 0 : (int)a[1], to = a[1] < 0 ? 8 : (int)a[1] + 1;
                for (int i = from; i < to && i < 8; i++) {
                    st.bSrcRGB[i] = (uint32_t)a[2]; st.bDstRGB[i] = (uint32_t)a[3];
                    st.bSrcA[i] = (uint32_t)a[4]; st.bDstA[i] = (uint32_t)a[5];
                }
                return 1;
            }
            case OP_VIEWPORT: st.vp[0] = (int)a[1]; st.vp[1] = (int)a[2]; st.vp[2] = (int)a[3]; st.vp[3] = (int)a[4]; st.vpSet = true; return 1;
            case OP_FRONT_FACE: st.frontCW = a[1] == GL_CW; return 1;
            case OP_POLYGON_MODE: st.polyLine = a[1] == GL_LINE; return 1;
            case OP_CULL_FACE: st.cullFace = (uint32_t)a[1]; return 1;

            // ---- draws
            case OP_DRAW_ARRAYS: drawArrays((uint32_t)a[1], (uint32_t)a[2], (uint32_t)a[3], MAX((uint32_t)a[4], 1u)); return 1;
            case OP_DRAW_ELEMENTS: drawElements((uint32_t)a[1], (uint32_t)a[2], (uint32_t)a[3], (uint64_t)a[4], (uint32_t)a[5], (uint32_t)a[6]); return 1;
            case OP_DRAW_INDIRECT: drawIndirect((uint32_t)a[1], (uint32_t)a[2], (uint64_t)a[3]); return 1;
            case OP_MDIC: multiDrawIndirectCount((uint32_t)a[1], (uint32_t)a[2], (uint64_t)a[3], (uint64_t)a[4], (uint32_t)a[5], (uint32_t)a[6]); return 1;
            case OP_DISPATCH: dispatchCompute((uint32_t)a[1], (uint32_t)a[2], (uint32_t)a[3]); return 1;
            case OP_DISPATCH_INDIRECT: dispatchComputeIndirect((uint64_t)a[1]); return 1;

            // ---- sync
            case OP_FLUSH: flushCB(); return 1;
            case OP_FINISH: finishCB(); return 1;
            case OP_FENCE_CREATE: {
                Fence f;
                f.done = std::make_shared<std::atomic<bool>>(false);
                endEncoders();
                flushClears(0);
                endEncoders();
                // handlers can only be added before commit, an empty command buffer is ordered after all previous ones
                ensureCB();
                auto flag = f.done;
                [g->cb addCompletedHandler:^(id<MTLCommandBuffer>) { flag->store(true); }];
                [g->cb commit];
                g->lastCb = g->cb;
                g->cb = nil;
                uint32_t id = newId();
                g->fences.emplace(id, std::move(f));
                return id;
            }
            case OP_FENCE_STATUS: {
                auto* f = find(g->fences, (uint32_t)a[1]);
                return !f || f->done->load() ? 1 : 0;
            }
            case OP_FENCE_DELETE: g->fences.erase((uint32_t)a[1]); return 1;
        }
    }
    return 0;
}

} // namespace vmtl
