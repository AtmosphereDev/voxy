// Shared ABI between the Java side (MetalNative.java) and the native library.
// Every call goes through the single exported entry point
//     int64_t vmtl_call(int64_t* args);
// where args[0] is an Op and the remaining slots are operands (pointers are passed as int64).
#pragma once
#include <cstdint>

namespace vmtl {

enum Op : int64_t {
    OP_INIT = 0,            // (capsOut[16]) -> 1 on success
    OP_SHUTDOWN = 1,

    OP_BUF_CREATE = 10,     // (size, cpuMapped) -> id
    OP_BUF_DELETE = 11,     // (id)
    OP_BUF_PTR = 12,        // (id) -> cpu address
    OP_BUF_COPY = 13,       // (src, dst, srcOff, dstOff, size)
    OP_BUF_FILL32 = 14,     // (id, off, size, value)   value==0 uses blit fill
    OP_BUF_READ = 15,       // (id, off, size, dstPtr)  flushes and waits
    OP_BUF_SUBDATA = 16,    // (id, off, size, srcPtr)

    OP_TEX_CREATE = 20,     // (w, h, levels, glFormat, shared, glNameOut) -> id
    OP_TEX_DELETE = 21,
    OP_TEX_SUBIMAGE = 22,   // (id, lvl, x, y, w, h, glFormat, glType, ptr)
    OP_TEX_CLEAR = 23,      // (id, lvl, glFormat, glType, ptr or 0)
    OP_TEX_GENMIPS = 24,
    OP_TEX_PARAM = 25,      // (id, pname, valueBits, isFloat)
    OP_TEX_READ = 26,       // (id, lvl, glFormat, glType, dstPtr)
    OP_TEX_VIEW = 27,       // (src, glFormat, baseLevel, levels) -> id
    OP_TEX_DEPTH_TO_R32F = 28, // (srcDepth, dstR32F)

    OP_SAMP_CREATE = 30,
    OP_SAMP_DELETE = 31,
    OP_SAMP_PARAM = 32,     // (id, pname, valueBits, isFloat)

    OP_PROG_CREATE = 40,    // (nStages, stagesPtr, logPtr, logCap) -> id (0 on failure)
    OP_PROG_DELETE = 41,
    OP_UNIFORM = 42,        // (prog, location, ptr, nbytes)

    OP_USE_PROGRAM = 50,
    OP_BIND_BASE = 51,      // (target, index, buf, offset, size or -1)
    OP_BIND_TEXUNIT = 52,   // (unit, tex)
    OP_BIND_SAMPLER = 53,   // (unit, sampler)
    OP_BIND_IMAGE = 54,     // (unit, tex, level, access, glFormat)
    OP_BIND_FB = 55,        // (id)
    OP_BIND_ELEMENT = 56,   // (buf)
    OP_BIND_INDIRECT = 57,  // (buf)
    OP_BIND_PARAMETER = 58, // (buf)
    OP_BIND_DISPATCH = 59,  // (buf)

    OP_FB_CREATE = 60,
    OP_FB_DELETE = 61,
    OP_FB_ATTACH = 62,      // (fb, glAttachment, tex, level)
    OP_FB_DRAWBUFFERS = 63, // (fb, colourMask)
    OP_FB_CLEAR = 64,       // (fb, kind, drawBuffer, f0, f1, f2, f3)  kind: 0 colour, 1 depth, 2 stencil, 3 depth+stencil

    OP_SET_CAP = 70,        // (glCap, enabled)
    OP_DEPTH_FUNC = 71,     // (glFunc)
    OP_DEPTH_MASK = 72,     // (bool)
    OP_COLOR_MASK = 73,     // (buffer or -1, r, g, b, a)
    OP_STENCIL_FUNC = 74,   // (glFunc, ref, mask)
    OP_STENCIL_OP = 75,     // (sfail, dpfail, dppass)
    OP_STENCIL_MASK = 76,   // (mask)
    OP_BLEND_SEP = 77,      // (buffer or -1, srcRGB, dstRGB, srcA, dstA)
    OP_VIEWPORT = 78,       // (x, y, w, h)
    OP_FRONT_FACE = 79,     // (glFace)
    OP_POLYGON_MODE = 80,   // (glMode)
    OP_CULL_FACE = 81,      // (glFace)

    OP_DRAW_ARRAYS = 90,    // (mode, first, count, instances)
    OP_DRAW_ELEMENTS = 91,  // (mode, count, type, byteOffset, instances, baseInstance)
    OP_DRAW_INDIRECT = 92,  // (mode, type, byteOffset)
    OP_MDIC = 93,           // (mode, type, indirectOffset, countOffset, maxCount, stride)
    OP_DISPATCH = 94,       // (x, y, z)
    OP_DISPATCH_INDIRECT = 95, // (byteOffset)

    OP_FLUSH = 100,
    OP_FINISH = 101,
    OP_FENCE_CREATE = 102,  // -> id
    OP_FENCE_STATUS = 103,  // (id) -> 1 if signalled
    OP_FENCE_DELETE = 104,

    OP_TEX_COPY = 110,      // (src, dst, level) same format whole level copy
};

// Handle ids returned by the library carry this tag so that the Java side can
// tell them apart from real OpenGL object names.
constexpr uint32_t HANDLE_TAG = 0x40000000u;

} // namespace vmtl
