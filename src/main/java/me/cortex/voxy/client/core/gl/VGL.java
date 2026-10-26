package me.cortex.voxy.client.core.gl;

import me.cortex.voxy.client.core.gl.metal.MetalNative;
import org.lwjgl.opengl.*;
import org.lwjgl.system.MemoryUtil;

import static me.cortex.voxy.client.core.gl.metal.MetalNative.*;

/**
 * Thin facade over the OpenGL calls Voxy issues against its own resources.
 *
 * On every platform except macOS this forwards 1:1 to LWJGL. On macOS (OpenGL 4.1, no compute shaders, SSBOs, indirect
 * counts, buffer storage...) Voxy's resources live in Metal (see native/voxymtl), objects are identified by handle ids
 * carrying {@link MetalNative#HANDLE_TAG}. Resource calls are routed by the tag of the object, fixed function state
 * calls go to Metal only, Minecraft's own OpenGL state is never touched by them.
 */
public final class VGL {
    public static final boolean METAL = MetalNative.AVAILABLE;

    private static final int GL_ELEMENT_ARRAY_BUFFER = 0x8893;
    private static final int GL_DRAW_INDIRECT_BUFFER = 0x8F3F;
    private static final int GL_PARAMETER_BUFFER_ARB = 0x80EE;
    private static final int GL_DISPATCH_INDIRECT_BUFFER = 0x90EE;

    private static final long SCRATCH = MemoryUtil.nmemCallocChecked(1, 64);
    private static int metalProgram;
    private static int metalFramebuffer;
    private static final float[] clearColour = new float[4];

    private VGL() {}

    private static boolean m(int id) {
        return METAL && (id == 0 || MetalNative.isHandle(id));
    }

    //The framebuffer currently bound through this facade if it is a Metal one, 0 otherwise
    public static int boundMetalFramebuffer() {
        return metalFramebuffer;
    }

    // ---------------------------------------------------------------- buffers

    public static void glBindBufferBase(int target, int index, int buffer) {
        if (m(buffer)) call(OP_BIND_BASE, target, index, buffer, 0, -1);
        else GL30C.glBindBufferBase(target, index, buffer);
    }

    public static void glBindBufferRange(int target, int index, int buffer, long offset, long size) {
        if (m(buffer)) call(OP_BIND_BASE, target, index, buffer, offset, size);
        else GL30C.glBindBufferRange(target, index, buffer, offset, size);
    }

    public static void glBindBuffer(int target, int buffer) {
        if (METAL) {
            switch (target) {
                case GL_ELEMENT_ARRAY_BUFFER -> call(OP_BIND_ELEMENT, buffer);
                case GL_DRAW_INDIRECT_BUFFER -> call(OP_BIND_INDIRECT, buffer);
                case GL_PARAMETER_BUFFER_ARB -> call(OP_BIND_PARAMETER, buffer);
                case GL_DISPATCH_INDIRECT_BUFFER -> call(OP_BIND_DISPATCH, buffer);
                default -> {}
            }
        } else {
            GL15C.glBindBuffer(target, buffer);
        }
    }

    public static void glCopyNamedBufferSubData(int src, int dst, long srcOffset, long dstOffset, long size) {
        if (m(src)) call(OP_BUF_COPY, src, dst, srcOffset, dstOffset, size);
        else GL45C.glCopyNamedBufferSubData(src, dst, srcOffset, dstOffset, size);
    }

    public static void glFlushMappedNamedBufferRange(int buffer, long offset, long length) {
        if (!METAL) GL45C.glFlushMappedNamedBufferRange(buffer, offset, length);//Metal shared storage is coherent
    }

    public static void nglClearNamedBufferSubData(int buffer, int internalFormat, long offset, long size, int format, int type, long data) {
        if (m(buffer)) {
            int value = 0;
            if (data != 0) {
                value = internalFormat == GL30C.GL_R8UI ? MemoryUtil.memGetByte(data) & 0xFF : MemoryUtil.memGetInt(data);
            }
            call(OP_BUF_FILL32, buffer, offset, size, value);
        } else {
            GL45C.nglClearNamedBufferSubData(buffer, internalFormat, offset, size, format, type, data);
        }
    }

    public static void nglClearNamedBufferData(int buffer, int internalFormat, int format, int type, long data) {
        if (m(buffer)) {
            int value = 0;
            if (data != 0) {
                value = internalFormat == GL30C.GL_R8UI ? MemoryUtil.memGetByte(data) & 0xFF : MemoryUtil.memGetInt(data);
            }
            //size is resolved natively from the whole buffer (size -1)
            call(OP_BUF_FILL32, buffer, 0, -1, value);
        } else {
            GL45C.nglClearNamedBufferData(buffer, internalFormat, format, type, data);
        }
    }

    //Clears with a repeating pattern of ints (e.g. GL_RGBA32UI)
    private static void fillPattern(int buffer, long offset, long size, int[] pattern) {
        long data = MemoryUtil.nmemAllocChecked(size);
        try {
            for (long i = 0; i < size / 4; i++) {
                MemoryUtil.memPutInt(data + i * 4, pattern[(int) (i % pattern.length)]);
            }
            call(OP_BUF_SUBDATA, buffer, offset, size, data);
        } finally {
            MemoryUtil.nmemFree(data);
        }
    }

    public static void glClearNamedBufferData(int buffer, int internalFormat, int format, int type, int[] data) {
        if (m(buffer)) fillPattern(buffer, 0, MetalNative.bufferSize(buffer), data);
        else GL45C.glClearNamedBufferData(buffer, internalFormat, format, type, data);
    }

    public static void glClearNamedBufferSubData(int buffer, int internalFormat, long offset, long size, int format, int type, int[] data) {
        if (m(buffer)) fillPattern(buffer, offset, size, data);
        else GL45C.glClearNamedBufferSubData(buffer, internalFormat, offset, size, format, type, data);
    }

    public static void nglGetNamedBufferSubData(int buffer, long offset, long size, long data) {
        if (m(buffer)) call(OP_BUF_READ, buffer, offset, size, data);
        else GL45C.nglGetNamedBufferSubData(buffer, offset, size, data);
    }

    // ---------------------------------------------------------------- textures / samplers

    public static void nglTextureSubImage2D(int texture, int level, int x, int y, int width, int height, int format, int type, long pixels) {
        if (m(texture)) call(OP_TEX_SUBIMAGE, texture, level, x, y, width, height, format, type, pixels);
        else GL45C.nglTextureSubImage2D(texture, level, x, y, width, height, format, type, pixels);
    }

    public static void glBindTextureUnit(int unit, int texture) {
        if (METAL) call(OP_BIND_TEXUNIT, unit, MetalNative.isHandle(texture) ? texture : 0);
        else GL45C.glBindTextureUnit(unit, texture);
    }

    public static void glTextureParameteri(int texture, int pname, int value) {
        if (m(texture)) call(OP_TEX_PARAM, texture, pname, value, 0);
        else GL45C.glTextureParameteri(texture, pname, value);
    }

    public static void glTextureParameterf(int texture, int pname, float value) {
        if (m(texture)) call(OP_TEX_PARAM, texture, pname, Float.floatToRawIntBits(value), 1);
        else GL45C.glTextureParameterf(texture, pname, value);
    }

    public static int glGenSamplers() {
        return METAL ? (int) call(OP_SAMP_CREATE) : GL33C.glGenSamplers();
    }

    public static int glCreateSamplers() {
        return METAL ? (int) call(OP_SAMP_CREATE) : GL45C.glCreateSamplers();
    }

    public static void glDeleteSamplers(int sampler) {
        if (m(sampler)) call(OP_SAMP_DELETE, sampler);
        else GL33C.glDeleteSamplers(sampler);
    }

    public static void glSamplerParameteri(int sampler, int pname, int value) {
        if (m(sampler)) call(OP_SAMP_PARAM, sampler, pname, value, 0);
        else GL33C.glSamplerParameteri(sampler, pname, value);
    }

    public static void glBindSampler(int unit, int sampler) {
        if (METAL) call(OP_BIND_SAMPLER, unit, MetalNative.isHandle(sampler) ? sampler : 0);
        else GL33C.glBindSampler(unit, sampler);
    }

    public static void glBindImageTexture(int unit, int texture, int level, boolean layered, int layer, int access, int format) {
        if (METAL) call(OP_BIND_IMAGE, unit, MetalNative.isHandle(texture) ? texture : 0, level, access, format);
        else GL42C.glBindImageTexture(unit, texture, level, layered, layer, access, format);
    }

    // ---------------------------------------------------------------- framebuffers

    public static void glBindFramebuffer(int target, int framebuffer) {
        if (METAL && MetalNative.isHandle(framebuffer)) {
            metalFramebuffer = framebuffer;
            call(OP_BIND_FB, framebuffer);
        } else {
            metalFramebuffer = 0;
            GL30C.glBindFramebuffer(target, framebuffer);
        }
    }

    public static void glNamedFramebufferTexture(int framebuffer, int attachment, int texture, int level) {
        if (m(framebuffer)) call(OP_FB_ATTACH, framebuffer, attachment, texture, level);
        else GL45C.glNamedFramebufferTexture(framebuffer, attachment, texture, level);
    }

    public static void glNamedFramebufferDrawBuffer(int framebuffer, int buf) {
        if (m(framebuffer)) {
            int mask = 0;
            if (buf >= GL30C.GL_COLOR_ATTACHMENT0 && buf < GL30C.GL_COLOR_ATTACHMENT0 + 8) mask = 1 << (buf - GL30C.GL_COLOR_ATTACHMENT0);
            call(OP_FB_DRAWBUFFERS, framebuffer, mask);
        } else {
            GL45C.glNamedFramebufferDrawBuffer(framebuffer, buf);
        }
    }

    public static void glNamedFramebufferDrawBuffers(int framebuffer, int[] bufs) {
        if (m(framebuffer)) {
            int mask = 0;
            for (int buf : bufs) {
                if (buf >= GL30C.GL_COLOR_ATTACHMENT0 && buf < GL30C.GL_COLOR_ATTACHMENT0 + 8) mask |= 1 << (buf - GL30C.GL_COLOR_ATTACHMENT0);
            }
            call(OP_FB_DRAWBUFFERS, framebuffer, mask);
        } else {
            GL45C.glNamedFramebufferDrawBuffers(framebuffer, bufs);
        }
    }

    public static int glCheckNamedFramebufferStatus(int framebuffer, int target) {
        if (m(framebuffer)) return GL30C.GL_FRAMEBUFFER_COMPLETE;
        return GL45C.glCheckNamedFramebufferStatus(framebuffer, target);
    }

    public static void glClearNamedFramebufferfi(int framebuffer, int buffer, int drawBuffer, float depth, int stencil) {
        if (m(framebuffer)) call(OP_FB_CLEAR, framebuffer, 3, 0, Float.floatToRawIntBits(depth), stencil);
        else GL45C.glClearNamedFramebufferfi(framebuffer, buffer, drawBuffer, depth, stencil);
    }

    public static void nglClearNamedFramebufferfv(int framebuffer, int buffer, int drawBuffer, long value) {
        if (m(framebuffer)) {
            if (buffer == GL11C.GL_DEPTH) {
                call(OP_FB_CLEAR, framebuffer, 1, 0, MemoryUtil.memGetInt(value));
            } else {
                call(OP_FB_CLEAR, framebuffer, 0, drawBuffer, MemoryUtil.memGetInt(value), MemoryUtil.memGetInt(value + 4), MemoryUtil.memGetInt(value + 8), MemoryUtil.memGetInt(value + 12));
            }
        } else {
            GL45C.nglClearNamedFramebufferfv(framebuffer, buffer, drawBuffer, value);
        }
    }

    public static void nglClearNamedFramebufferiv(int framebuffer, int buffer, int drawBuffer, long value) {
        if (m(framebuffer)) {
            if (buffer == GL11C.GL_STENCIL) {
                call(OP_FB_CLEAR, framebuffer, 2, 0, MemoryUtil.memGetInt(value));
            }
        } else {
            GL45C.nglClearNamedFramebufferiv(framebuffer, buffer, drawBuffer, value);
        }
    }

    public static void glClearColor(float r, float g, float b, float a) {
        if (METAL) {
            clearColour[0] = r; clearColour[1] = g; clearColour[2] = b; clearColour[3] = a;
        } else {
            GL11C.glClearColor(r, g, b, a);
        }
    }

    public static void glClear(int mask) {
        if (METAL && metalFramebuffer != 0) {
            if ((mask & GL11C.GL_COLOR_BUFFER_BIT) != 0) {
                call(OP_FB_CLEAR, metalFramebuffer, 0, 0, Float.floatToRawIntBits(clearColour[0]), Float.floatToRawIntBits(clearColour[1]), Float.floatToRawIntBits(clearColour[2]), Float.floatToRawIntBits(clearColour[3]));
            }
            if ((mask & GL11C.GL_DEPTH_BUFFER_BIT) != 0) {
                call(OP_FB_CLEAR, metalFramebuffer, 1, 0, Float.floatToRawIntBits(1.0f));
            }
            if ((mask & GL11C.GL_STENCIL_BUFFER_BIT) != 0) {
                call(OP_FB_CLEAR, metalFramebuffer, 2, 0, 0);
            }
        } else {
            GL11C.glClear(mask);
        }
    }

    public static int glGetInteger(int pname) {
        if (METAL && metalFramebuffer != 0 && (pname == GL30C.GL_DRAW_FRAMEBUFFER_BINDING || pname == GL30C.GL_FRAMEBUFFER_BINDING)) {
            return metalFramebuffer;
        }
        return GL11C.glGetInteger(pname);
    }

    // ---------------------------------------------------------------- programs / uniforms

    public static void glUseProgram(int program) {
        if (METAL && MetalNative.isHandle(program)) {
            metalProgram = program;
            call(OP_USE_PROGRAM, program);
        } else {
            GL20C.glUseProgram(program);
        }
    }

    public static void glUniform1i(int location, int v) {
        if (METAL) {
            MemoryUtil.memPutInt(SCRATCH, v);
            call(OP_UNIFORM, metalProgram, location, SCRATCH, 4);
        } else {
            GL20C.glUniform1i(location, v);
        }
    }

    public static void glUniform1ui(int location, int v) {
        if (METAL) {
            MemoryUtil.memPutInt(SCRATCH, v);
            call(OP_UNIFORM, metalProgram, location, SCRATCH, 4);
        } else {
            GL30C.glUniform1ui(location, v);
        }
    }

    public static void glUniform1f(int location, float v) {
        if (METAL) {
            MemoryUtil.memPutFloat(SCRATCH, v);
            call(OP_UNIFORM, metalProgram, location, SCRATCH, 4);
        } else {
            GL20C.glUniform1f(location, v);
        }
    }

    public static void glUniform2f(int location, float x, float y) {
        if (METAL) {
            MemoryUtil.memPutFloat(SCRATCH, x);
            MemoryUtil.memPutFloat(SCRATCH + 4, y);
            call(OP_UNIFORM, metalProgram, location, SCRATCH, 8);
        } else {
            GL20C.glUniform2f(location, x, y);
        }
    }

    public static void glUniform4f(int location, float x, float y, float z, float w) {
        if (METAL) {
            MemoryUtil.memPutFloat(SCRATCH, x);
            MemoryUtil.memPutFloat(SCRATCH + 4, y);
            MemoryUtil.memPutFloat(SCRATCH + 8, z);
            MemoryUtil.memPutFloat(SCRATCH + 12, w);
            call(OP_UNIFORM, metalProgram, location, SCRATCH, 16);
        } else {
            GL20C.glUniform4f(location, x, y, z, w);
        }
    }

    public static void nglUniformMatrix4fv(int location, int count, boolean transpose, long value) {
        if (METAL) call(OP_UNIFORM, metalProgram, location, value, 64L * count);
        else GL20C.nglUniformMatrix4fv(location, count, transpose, value);
    }

    // ---------------------------------------------------------------- state

    public static void glEnable(int cap) {
        if (METAL) call(OP_SET_CAP, cap, 1);
        else GL11C.glEnable(cap);
    }

    public static void glDisable(int cap) {
        if (METAL) call(OP_SET_CAP, cap, 0);
        else GL11C.glDisable(cap);
    }

    public static void glEnablei(int cap, int index) {
        if (METAL) call(OP_SET_CAP, cap, 1);
        else GL30C.glEnablei(cap, index);
    }

    public static void glDisablei(int cap, int index) {
        if (METAL) call(OP_SET_CAP, cap, 0);
        else GL30C.glDisablei(cap, index);
    }

    public static void glDepthFunc(int func) {
        if (METAL) call(OP_DEPTH_FUNC, func);
        else GL11C.glDepthFunc(func);
    }

    public static void glDepthMask(boolean mask) {
        if (METAL) call(OP_DEPTH_MASK, mask ? 1 : 0);
        else GL11C.glDepthMask(mask);
    }

    public static void glColorMask(boolean r, boolean g, boolean b, boolean a) {
        if (METAL) call(OP_COLOR_MASK, -1, r ? 1 : 0, g ? 1 : 0, b ? 1 : 0, a ? 1 : 0);
        else GL11C.glColorMask(r, g, b, a);
    }

    public static void glStencilFunc(int func, int ref, int mask) {
        if (METAL) call(OP_STENCIL_FUNC, func, ref, mask);
        else GL11C.glStencilFunc(func, ref, mask);
    }

    public static void glStencilOp(int sfail, int dpfail, int dppass) {
        if (METAL) call(OP_STENCIL_OP, sfail, dpfail, dppass);
        else GL11C.glStencilOp(sfail, dpfail, dppass);
    }

    public static void glStencilMask(int mask) {
        if (METAL) call(OP_STENCIL_MASK, mask);
        else GL11C.glStencilMask(mask);
    }

    public static void glBlendFuncSeparate(int srcRGB, int dstRGB, int srcAlpha, int dstAlpha) {
        if (METAL) call(OP_BLEND_SEP, -1, srcRGB, dstRGB, srcAlpha, dstAlpha);
        else GL14C.glBlendFuncSeparate(srcRGB, dstRGB, srcAlpha, dstAlpha);
    }

    public static void glBlendFuncSeparateiARB(int buf, int srcRGB, int dstRGB, int srcAlpha, int dstAlpha) {
        if (METAL) call(OP_BLEND_SEP, buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
        else ARBDrawBuffersBlend.glBlendFuncSeparateiARB(buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
    }

    public static void glViewport(int x, int y, int w, int h) {
        if (METAL) call(OP_VIEWPORT, x, y, w, h);
        else GL11C.glViewport(x, y, w, h);
    }

    public static void glFrontFace(int mode) {
        if (METAL) call(OP_FRONT_FACE, mode);
        else GL11C.glFrontFace(mode);
    }

    public static void glCullFace(int mode) {
        if (METAL) call(OP_CULL_FACE, mode);
        else GL11C.glCullFace(mode);
    }

    public static void glPolygonMode(int face, int mode) {
        if (METAL) call(OP_POLYGON_MODE, mode);
        else GL11C.glPolygonMode(face, mode);
    }

    public static void glProvokingVertex(int mode) {
        //Metal always uses the first vertex, which is what Voxy requests
        if (!METAL) GL32C.glProvokingVertex(mode);
    }

    // ---------------------------------------------------------------- draws / dispatches / sync

    public static void glDrawArrays(int mode, int first, int count) {
        if (METAL) call(OP_DRAW_ARRAYS, mode, first, count, 1);
        else GL11C.glDrawArrays(mode, first, count);
    }

    public static void glDrawElements(int mode, int count, int type, long offset) {
        if (METAL) call(OP_DRAW_ELEMENTS, mode, count, type, offset, 1, 0);
        else GL11C.glDrawElements(mode, count, type, offset);
    }

    public static void glDrawElementsInstanced(int mode, int count, int type, long offset, int instances) {
        if (METAL) call(OP_DRAW_ELEMENTS, mode, count, type, offset, instances, 0);
        else GL31C.glDrawElementsInstanced(mode, count, type, offset, instances);
    }

    public static void glDrawElementsInstancedBaseInstance(int mode, int count, int type, long offset, int instances, int baseInstance) {
        if (METAL) call(OP_DRAW_ELEMENTS, mode, count, type, offset, instances, baseInstance);
        else GL42C.glDrawElementsInstancedBaseInstance(mode, count, type, offset, instances, baseInstance);
    }

    public static void glDrawElementsIndirect(int mode, int type, long indirect) {
        if (METAL) call(OP_DRAW_INDIRECT, mode, type, indirect);
        else GL40C.glDrawElementsIndirect(mode, type, indirect);
    }

    public static void glMultiDrawElementsIndirectCountARB(int mode, int type, long indirect, long drawCount, int maxDrawCount, int stride) {
        if (METAL) call(OP_MDIC, mode, type, indirect, drawCount, maxDrawCount, stride);
        else ARBIndirectParameters.glMultiDrawElementsIndirectCountARB(mode, type, indirect, drawCount, maxDrawCount, stride);
    }

    public static void glDispatchCompute(int x, int y, int z) {
        if (METAL) call(OP_DISPATCH, x, y, z);
        else GL43C.glDispatchCompute(x, y, z);
    }

    public static void glDispatchComputeIndirect(long indirect) {
        if (METAL) call(OP_DISPATCH_INDIRECT, indirect);
        else GL43C.glDispatchComputeIndirect(indirect);
    }

    //Metal tracks resource hazards itself
    public static void glMemoryBarrier(int barriers) {
        if (!METAL) GL42C.glMemoryBarrier(barriers);
    }

    public static void glTextureBarrier() {
        if (!METAL) GL45C.glTextureBarrier();
    }

    public static void glFinish() {
        if (METAL) call(OP_FINISH);
        else GL11C.glFinish();
    }

    public static void glFlush() {
        if (METAL) call(OP_FLUSH);
        else GL11C.glFlush();
    }

    public static int glGetError() {
        return METAL ? GL11C.GL_NO_ERROR : GL11C.glGetError();
    }

    //Vertex array objects are meaningless for Metal (Voxy only uses attribute-less rendering)
    public static void glBindVertexArray(int array) {
        if (!METAL) GL30C.glBindVertexArray(array);
    }

    public static int glGenVertexArrays() {
        return METAL ? 0 : GL30C.glGenVertexArrays();
    }

    public static int glCreateVertexArrays() {
        return METAL ? 0 : GL45C.glCreateVertexArrays();
    }

    public static void glDeleteVertexArrays(int array) {
        if (!METAL) GL30C.glDeleteVertexArrays(array);
    }
}
