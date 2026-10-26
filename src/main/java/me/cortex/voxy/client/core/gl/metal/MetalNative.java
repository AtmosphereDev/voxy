package me.cortex.voxy.client.core.gl.metal;

import me.cortex.voxy.common.Logger;
import org.lwjgl.system.JNI;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.system.Platform;
import org.lwjgl.system.macosx.DynamicLinkLoader;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Bridge to the native voxymtl library (see /native/voxymtl), which implements the subset of OpenGL 4.6 Voxy needs on top of Metal.
 * The library exposes a single entry point, vmtl_call(int64_t* args), that is invoked through LWJGL's JNI helper, so no
 * custom JNI glue is required. The operand layout of every op is documented in native/voxymtl/src/vmtl.h.
 */
public final class MetalNative {
    //Mirrors vmtl.h
    public static final int OP_INIT = 0, OP_SHUTDOWN = 1;
    public static final int OP_BUF_CREATE = 10, OP_BUF_DELETE = 11, OP_BUF_PTR = 12, OP_BUF_COPY = 13, OP_BUF_FILL32 = 14, OP_BUF_READ = 15, OP_BUF_SUBDATA = 16;
    public static final int OP_TEX_CREATE = 20, OP_TEX_DELETE = 21, OP_TEX_SUBIMAGE = 22, OP_TEX_CLEAR = 23, OP_TEX_GENMIPS = 24, OP_TEX_PARAM = 25, OP_TEX_READ = 26, OP_TEX_VIEW = 27, OP_TEX_DEPTH_TO_R32F = 28;
    public static final int OP_SAMP_CREATE = 30, OP_SAMP_DELETE = 31, OP_SAMP_PARAM = 32;
    public static final int OP_PROG_CREATE = 40, OP_PROG_DELETE = 41, OP_UNIFORM = 42;
    public static final int OP_USE_PROGRAM = 50, OP_BIND_BASE = 51, OP_BIND_TEXUNIT = 52, OP_BIND_SAMPLER = 53, OP_BIND_IMAGE = 54, OP_BIND_FB = 55, OP_BIND_ELEMENT = 56, OP_BIND_INDIRECT = 57, OP_BIND_PARAMETER = 58, OP_BIND_DISPATCH = 59;
    public static final int OP_FB_CREATE = 60, OP_FB_DELETE = 61, OP_FB_ATTACH = 62, OP_FB_DRAWBUFFERS = 63, OP_FB_CLEAR = 64;
    public static final int OP_SET_CAP = 70, OP_DEPTH_FUNC = 71, OP_DEPTH_MASK = 72, OP_COLOR_MASK = 73, OP_STENCIL_FUNC = 74, OP_STENCIL_OP = 75, OP_STENCIL_MASK = 76, OP_BLEND_SEP = 77, OP_VIEWPORT = 78, OP_FRONT_FACE = 79, OP_POLYGON_MODE = 80, OP_CULL_FACE = 81;
    public static final int OP_DRAW_ARRAYS = 90, OP_DRAW_ELEMENTS = 91, OP_DRAW_INDIRECT = 92, OP_MDIC = 93, OP_DISPATCH = 94, OP_DISPATCH_INDIRECT = 95;
    public static final int OP_FLUSH = 100, OP_FINISH = 101, OP_FENCE_CREATE = 102, OP_FENCE_STATUS = 103, OP_FENCE_DELETE = 104;
    public static final int OP_TEX_COPY = 110;

    /** Handles returned by the library carry this bit, real OpenGL object names never do */
    public static final int HANDLE_TAG = 0x40000000;

    public static final boolean AVAILABLE;

    //caps filled by OP_INIT
    public static long maxBufferLength;
    public static boolean supportsSubgroup;
    public static boolean supportsIcb;
    public static boolean unifiedMemory;
    public static boolean appleGpu;
    public static long recommendedWorkingSet;

    private static final long ENTRY;
    private static final long ARGS = MemoryUtil.nmemAllocChecked(32 * 8);

    static {
        long entry = 0;
        boolean ok = false;
        if (Platform.get() == Platform.MACOSX) {
            try {
                String override = System.getProperty("voxy.metal.lib");
                Path lib = override != null ? Path.of(override) : extractLibrary();
                long handle = DynamicLinkLoader.dlopen(lib.toAbsolutePath().toString(), DynamicLinkLoader.RTLD_NOW);
                if (handle == 0) {
                    throw new IllegalStateException("dlopen failed for " + lib + ": " + DynamicLinkLoader.dlerror());
                }
                entry = DynamicLinkLoader.dlsym(handle, "vmtl_call");
                if (entry == 0) {
                    throw new IllegalStateException("vmtl_call not found in " + lib);
                }
                ok = true;
            } catch (Throwable t) {
                Logger.error("Failed to load the voxymtl native library, macOS rendering is unavailable", t);
            }
        }
        ENTRY = entry;
        boolean initOk = false;
        if (ok) {
            long caps = MemoryUtil.nmemCallocChecked(16, 8);
            long[] a = {OP_INIT, caps};
            initOk = callRaw(a) == 1;
            if (initOk) {
                maxBufferLength = MemoryUtil.memGetLong(caps + 8);
                supportsSubgroup = MemoryUtil.memGetLong(caps + 24) != 0;
                supportsIcb = MemoryUtil.memGetLong(caps + 40) != 0;
                appleGpu = MemoryUtil.memGetLong(caps + 48) != 0;
                unifiedMemory = MemoryUtil.memGetLong(caps + 56) != 0;
                recommendedWorkingSet = MemoryUtil.memGetLong(caps + 64);
            } else {
                Logger.error("voxymtl failed to create a Metal device");
            }
            MemoryUtil.nmemFree(caps);
        }
        AVAILABLE = initOk;
    }

    private static long callRaw(long[] a) {
        for (int i = 0; i < a.length; i++) {
            MemoryUtil.memPutLong(ARGS + 8L * i, a[i]);
        }
        return JNI.callPP(ARGS, ENTRY);
    }

    private static Path extractLibrary() throws IOException {
        String arch = switch (Platform.getArchitecture()) {
            case ARM64 -> "arm64";
            case X64 -> "x64";
            default -> throw new IllegalStateException("Unsupported macOS architecture " + Platform.getArchitecture());
        };
        String res = "/natives/macos-" + arch + "/libvoxymtl.dylib";
        try (InputStream in = MetalNative.class.getResourceAsStream(res)) {
            if (in == null) {
                throw new IllegalStateException("Missing native resource " + res + " (build native/voxymtl with xmake and place it in src/main/resources)");
            }
            byte[] data = in.readAllBytes();
            Path dir = Path.of(System.getProperty("java.io.tmpdir"), "voxymtl-" + Integer.toHexString(java.util.Arrays.hashCode(data)));
            Files.createDirectories(dir);
            Path out = dir.resolve("libvoxymtl.dylib");
            if (!Files.exists(out) || Files.size(out) != data.length) {
                Path tmp = Files.createTempFile(dir, "lib", ".tmp");
                Files.write(tmp, data);
                Files.move(tmp, out, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            }
            return out;
        }
    }

    public static boolean isHandle(int id) {
        return (id & HANDLE_TAG) != 0;
    }

    //Single threaded: everything is invoked from the render thread
    public static long call(int op) {
        MemoryUtil.memPutLong(ARGS, op);
        return JNI.callPP(ARGS, ENTRY);
    }

    public static long call(int op, long a1) {
        MemoryUtil.memPutLong(ARGS, op);
        MemoryUtil.memPutLong(ARGS + 8, a1);
        return JNI.callPP(ARGS, ENTRY);
    }

    public static long call(int op, long a1, long a2) {
        MemoryUtil.memPutLong(ARGS, op);
        MemoryUtil.memPutLong(ARGS + 8, a1);
        MemoryUtil.memPutLong(ARGS + 16, a2);
        return JNI.callPP(ARGS, ENTRY);
    }

    public static long call(int op, long a1, long a2, long a3) {
        MemoryUtil.memPutLong(ARGS, op);
        MemoryUtil.memPutLong(ARGS + 8, a1);
        MemoryUtil.memPutLong(ARGS + 16, a2);
        MemoryUtil.memPutLong(ARGS + 24, a3);
        return JNI.callPP(ARGS, ENTRY);
    }

    public static long call(int op, long a1, long a2, long a3, long a4) {
        MemoryUtil.memPutLong(ARGS, op);
        MemoryUtil.memPutLong(ARGS + 8, a1);
        MemoryUtil.memPutLong(ARGS + 16, a2);
        MemoryUtil.memPutLong(ARGS + 24, a3);
        MemoryUtil.memPutLong(ARGS + 32, a4);
        return JNI.callPP(ARGS, ENTRY);
    }

    public static long call(int op, long... a) {
        MemoryUtil.memPutLong(ARGS, op);
        for (int i = 0; i < a.length; i++) {
            MemoryUtil.memPutLong(ARGS + 8L * (i + 1), a[i]);
        }
        return JNI.callPP(ARGS, ENTRY);
    }

    //Helpers for the more common operations
    private static final it.unimi.dsi.fastutil.ints.Int2LongOpenHashMap BUFFER_SIZES = new it.unimi.dsi.fastutil.ints.Int2LongOpenHashMap();

    public static int createBuffer(long size, boolean cpuMapped) {
        int id = (int) call(OP_BUF_CREATE, size, cpuMapped ? 1 : 0);
        if (id == 0) {
            throw new IllegalStateException("Failed to allocate Metal buffer of size " + size);
        }
        BUFFER_SIZES.put(id, size);
        return id;
    }

    public static void deleteBuffer(int id) {
        call(OP_BUF_DELETE, id);
        BUFFER_SIZES.remove(id);
    }

    public static long bufferSize(int id) {
        return BUFFER_SIZES.get(id);
    }
}
