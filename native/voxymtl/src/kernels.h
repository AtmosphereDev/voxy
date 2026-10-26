// Built in Metal kernels used to emulate GL features Metal lacks natively.
#pragma once

namespace vmtl {

static const char* KERNEL_SOURCE = R"MSL(
#include <metal_stdlib>
using namespace metal;

// glClearNamedBufferData with a 32 bit pattern
kernel void vmtl_fill32(device uint* dst [[buffer(0)]],
                        constant uint3& p [[buffer(1)]],   // x = word offset, y = word count, z = value
                        uint i [[thread_position_in_grid]]) {
    if (i < p.y) dst[p.x + i] = p.z;
}

// Metal has no 8 bit index type, GL_UNSIGNED_BYTE index buffers are widened to 16 bit
kernel void vmtl_widen_u8(device const uchar* src [[buffer(0)]],
                          device ushort* dst [[buffer(1)]],
                          constant uint& n [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = (ushort)src[i];
}

// Depth (d32 / d32s8) -> r32f so that GL can sample it through an IOSurface
kernel void vmtl_depth_to_r32f(depth2d<float, access::read> src [[texture(0)]],
                               texture2d<float, access::write> dst [[texture(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    dst.write(float4(src.read(gid), 0.0, 0.0, 1.0), gid);
}

// glMultiDrawElementsIndirectCount: the GPU written draw list + count is converted into an
// indirect command buffer plus an execution range, executed by the render encoder afterwards
struct MdicParams {
    uint cmdBaseWords;      // first command (in 32 bit words) relative to the bound indirect buffer
    uint cmdStrideWords;    // distance between commands in words
    uint countWord;         // index of the count word in the count buffer
    uint maxCount;
};
struct ExecRange { uint location; uint length; };
struct ICBContainer { command_buffer cmdBuf [[id(0)]]; };

kernel void vmtl_mdic_encode(device const uint* cmds [[buffer(0)]],
                             device const uint* countBuf [[buffer(1)]],
                             constant MdicParams& p [[buffer(2)]],
                             device ushort* indexBuf [[buffer(3)]],
                             device ICBContainer* container [[buffer(4)]],
                             device ExecRange* range [[buffer(5)]],
                             uint tid [[thread_position_in_grid]]) {
    uint n = min(countBuf[p.countWord], p.maxCount);
    if (tid == 0) {
        range->location = 0;
        range->length = n;
    }
    if (tid >= n) return;
    device const uint* c = cmds + p.cmdBaseWords + tid * p.cmdStrideWords;
    render_command cmd(container->cmdBuf, tid);
    cmd.draw_indexed_primitives(primitive_type::triangle, c[0], indexBuf + c[2], c[1], (int)c[3], c[4]);
}
)MSL";

} // namespace vmtl
