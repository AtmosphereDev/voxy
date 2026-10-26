# voxymtl – Metal backend for Voxy on macOS

macOS only exposes OpenGL 4.1 (no compute shaders, SSBOs, multi draw indirect, persistent buffers...), Voxy needs 4.6.
Instead of running Voxy's GL calls, Voxy's own resources live in Metal and the calls are translated by this library.
Minecraft keeps rendering with its normal OpenGL 4.1 context; only a few things cross between the two (see `MacInterop.java`).

## Build

```
./native/build.sh          # on a Mac, needs xmake + xcode command line tools
./gradlew build            # packs native/dist/** into the jar under /natives
```
Or run the `build-natives-macos` GitHub workflow, it builds both architectures and a jar that bundles them.
`-Dvoxy.metal.lib=/path/libvoxymtl.dylib` loads a library from disk instead (development).

## How it works

| GL feature Voxy uses | Metal implementation |
| --- | --- |
| GLSL 4.6 shaders | glslang -> SPIR-V -> SPIRV-Cross -> MSL, at shader creation (`shader_compiler.cpp`) |
| `layout(location=N) uniform ...` | packed in one generated std140 block, shadowed on the CPU, uploaded with `setBytes` |
| separate binding namespaces per resource type | every resource gets a unique SPIR-V binding and a fixed MSL slot |
| GL clip space / bottom-left origin | `fixup_clipspace` + `flip_vert_y` in the vertex stage, inverted winding |
| SSBOs, compute, indirect dispatch | `MTLBuffer`, compute encoder, `dispatchThreadgroupsWithIndirectBuffer` |
| `glMultiDrawElementsIndirectCount` | compute kernel writes an indirect command buffer + execution range from the GPU draw list/count (fallback: CPU readback) |
| `glDrawElementsIndirect`, base instance | `drawIndexedPrimitives:...indirectBuffer` |
| `GL_UNSIGNED_BYTE` indices, `GL_TRIANGLE_FAN` | widened to 16 bit on the GPU / converted to indexed triangles |
| `GL_DEPTH24_STENCIL8` | `Depth32Float_Stencil8` |
| persistent mapped buffers + fences | shared storage buffers, command buffer completion handlers |
| `glMemoryBarrier` | no-op (Metal tracks hazards) |
| `GL_TEXTURE_BASE_LEVEL/MAX_LEVEL` | texture views |

The ABI (`vmtl_call(int64_t* args)`, one entry point, op codes in `src/vmtl.h`) is called from Java through LWJGL's
`JNI.callPP`, so there is no JNI glue. `VGL.java` routes Voxy's GL calls: Metal handles carry a tag bit, anything else
goes to the real OpenGL.

## Interop with Minecraft's OpenGL context (`MacInterop.java`)

* depth of the main framebuffer: GL pass -> IOSurface backed `r32f` rect texture -> read by Metal
* light map: copied on the CPU (16x16)
* result: Metal renders colour into an IOSurface texture and depth is converted to `r32f` by a kernel, a GL pass
  (`shaders/post/macos/composite.*`) blends both into the main framebuffer. Metal is waited on (`glFinish`) before.

## Known limitations / not done

* Iris shader pack pipeline is disabled on macOS (the normal pipeline is used).
* GPU timing queries, `GL_NV_*` / sparse buffers are not translated.
* The CPU waits for the GPU once per frame for the composite.
* Buffers are `Shared` on unified memory GPUs; on discrete GPUs only CPU mapped buffers are `Shared`.
