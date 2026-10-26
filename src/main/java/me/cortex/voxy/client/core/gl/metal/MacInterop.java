package me.cortex.voxy.client.core.gl.metal;

import me.cortex.voxy.client.core.RenderProperties;
import me.cortex.voxy.client.core.gl.GlTexture;
import me.cortex.voxy.client.core.gl.VGL;
import me.cortex.voxy.client.core.gl.shader.ShaderLoader;
import me.cortex.voxy.common.Logger;
import net.minecraft.client.Minecraft;
import org.joml.Matrix4f;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import static org.lwjgl.opengl.GL11C.*;
import static org.lwjgl.opengl.GL13C.GL_TEXTURE0;
import static org.lwjgl.opengl.GL13C.glActiveTexture;
import static org.lwjgl.opengl.GL14C.glBlendFuncSeparate;
import static org.lwjgl.opengl.GL15C.GL_ARRAY_BUFFER_BINDING;
import static org.lwjgl.opengl.GL20C.*;
import static org.lwjgl.opengl.GL30C.*;
import static org.lwjgl.opengl.GL31C.GL_TEXTURE_RECTANGLE;
import static org.lwjgl.opengl.GL33C.GL_SAMPLER_BINDING;
import static org.lwjgl.opengl.GL33C.glBindSampler;

/**
 * macOS only. Voxy renders with Metal while Minecraft owns the (OpenGL 4.1) window framebuffer, this class moves the few
 * things that have to cross between the two:
 * <ul>
 *     <li>Minecraft's depth buffer  GL -> Metal (a GL pass writes it into an IOSurface backed r32f texture)</li>
 *     <li>the light map GL -> Metal (16x16 texture, copied on the CPU)</li>
 *     <li>the finished colour / depth Metal -> GL (IOSurface backed textures sampled by a GL pass that blends into the main framebuffer)</li>
 * </ul>
 * Everything here talks to the real OpenGL context directly (not through VGL) and restores the GL state it changes.
 */
public final class MacInterop {
    private MacInterop() {}

    private static final String FULLSCREEN_VERT = """
            #version 410 core
            void main() {
                vec2 corner = vec2(gl_VertexID&1, (gl_VertexID>>1)&1);
                gl_Position = vec4(corner * 2.0f - 1.0f, 0.0f, 1.0f);
            }
            """;
    private static final String DEPTH_IMPORT_FRAG = """
            #version 410 core
            uniform sampler2D depthTex;
            out vec4 outColour;
            void main() {
                outColour = vec4(texelFetch(depthTex, ivec2(gl_FragCoord.xy), 0).r, 0.0f, 0.0f, 1.0f);
            }
            """;

    private static int vao;
    private static int importProgram;
    private static int importFbo;
    private static GlTexture importedDepth;
    private static int importedSrcW, importedSrcH;

    private static GlTexture lightmap;
    private static boolean lightmapDirty = true;

    private static int compositeProgram;
    private static int compositeFbo;
    private static boolean compositeFog;
    private static int uInvProj, uProj, uEndParams, uFogColour, uDepth, uColour;

    /** Called at the start of every voxy frame */
    public static void beginFrame() {
        lightmapDirty = true;
    }

    // ------------------------------------------------------------------ GL state handling

    private static final class SavedState {
        int drawFbo, readFbo, program, vao, activeTexture, tex2d, texRect, sampler, arrayBuffer;
        final int[] viewport = new int[4];
        boolean depthTest, blend, cull, stencil, scissor;
        boolean depthMask;
        final boolean[] colourMask = new boolean[4];
        int depthFunc;
        final int[] blendFunc = new int[4];

        void save() {
            this.drawFbo = glGetInteger(GL_DRAW_FRAMEBUFFER_BINDING);
            this.readFbo = glGetInteger(GL_READ_FRAMEBUFFER_BINDING);
            this.program = glGetInteger(GL_CURRENT_PROGRAM);
            this.vao = glGetInteger(GL_VERTEX_ARRAY_BINDING);
            this.arrayBuffer = glGetInteger(GL_ARRAY_BUFFER_BINDING);
            glGetIntegerv(GL_VIEWPORT, this.viewport);
            this.activeTexture = glGetInteger(GL_ACTIVE_TEXTURE);
            glActiveTexture(GL_TEXTURE0);
            this.tex2d = glGetInteger(GL_TEXTURE_BINDING_2D);
            this.texRect = glGetInteger(org.lwjgl.opengl.GL31C.GL_TEXTURE_BINDING_RECTANGLE);
            this.sampler = glGetInteger(GL_SAMPLER_BINDING);
            this.depthTest = glIsEnabled(GL_DEPTH_TEST);
            this.blend = glIsEnabled(GL_BLEND);
            this.cull = glIsEnabled(GL_CULL_FACE);
            this.stencil = glIsEnabled(GL_STENCIL_TEST);
            this.scissor = glIsEnabled(GL_SCISSOR_TEST);
            this.depthMask = glGetBoolean(GL_DEPTH_WRITEMASK);
            try (var stack = MemoryStack.stackPush()) {
                var b = stack.malloc(4);
                org.lwjgl.opengl.GL11.glGetBooleanv(GL_COLOR_WRITEMASK, b);
                for (int i = 0; i < 4; i++) this.colourMask[i] = b.get(i) != 0;
            }
            this.depthFunc = glGetInteger(GL_DEPTH_FUNC);
            this.blendFunc[0] = glGetInteger(org.lwjgl.opengl.GL14C.GL_BLEND_SRC_RGB);
            this.blendFunc[1] = glGetInteger(org.lwjgl.opengl.GL14C.GL_BLEND_DST_RGB);
            this.blendFunc[2] = glGetInteger(org.lwjgl.opengl.GL14C.GL_BLEND_SRC_ALPHA);
            this.blendFunc[3] = glGetInteger(org.lwjgl.opengl.GL14C.GL_BLEND_DST_ALPHA);
        }

        void restore() {
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, this.drawFbo);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, this.readFbo);
            glUseProgram(this.program);
            glBindVertexArray(this.vao);
            glBindBuffer(org.lwjgl.opengl.GL15C.GL_ARRAY_BUFFER, this.arrayBuffer);
            glViewport(this.viewport[0], this.viewport[1], this.viewport[2], this.viewport[3]);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, this.tex2d);
            glBindTexture(GL_TEXTURE_RECTANGLE, this.texRect);
            glBindSampler(0, this.sampler);
            glActiveTexture(this.activeTexture);
            setCap(GL_DEPTH_TEST, this.depthTest);
            setCap(GL_BLEND, this.blend);
            setCap(GL_CULL_FACE, this.cull);
            setCap(GL_STENCIL_TEST, this.stencil);
            setCap(GL_SCISSOR_TEST, this.scissor);
            glDepthMask(this.depthMask);
            glColorMask(this.colourMask[0], this.colourMask[1], this.colourMask[2], this.colourMask[3]);
            glDepthFunc(this.depthFunc);
            glBlendFuncSeparate(this.blendFunc[0], this.blendFunc[1], this.blendFunc[2], this.blendFunc[3]);
        }
    }

    private static void setCap(int cap, boolean on) {
        if (on) glEnable(cap); else glDisable(cap);
    }

    private static int compileProgram(String vert, String frag, String name) {
        int program = glCreateProgram();
        int[] shaders = {glCreateShader(GL_VERTEX_SHADER), glCreateShader(GL_FRAGMENT_SHADER)};
        String[] sources = {vert, frag};
        for (int i = 0; i < 2; i++) {
            glShaderSource(shaders[i], sources[i]);
            glCompileShader(shaders[i]);
            if (glGetShaderi(shaders[i], GL_COMPILE_STATUS) != GL_TRUE) {
                throw new IllegalStateException("macOS interop shader '" + name + "' failed to compile: " + glGetShaderInfoLog(shaders[i]));
            }
            glAttachShader(program, shaders[i]);
        }
        glLinkProgram(program);
        if (glGetProgrami(program, GL_LINK_STATUS) != GL_TRUE) {
            throw new IllegalStateException("macOS interop program '" + name + "' failed to link: " + glGetProgramInfoLog(program));
        }
        for (int s : shaders) {
            glDetachShader(program, s);
            glDeleteShader(s);
        }
        return program;
    }

    private static void ensureCommon() {
        if (vao == 0) {
            vao = glGenVertexArrays();
        }
    }

    // ------------------------------------------------------------------ GL -> Metal: depth

    /** The OpenGL name of the depth attachment of a (real GL) framebuffer */
    public static int getDepthAttachmentGlName(int framebuffer) {
        int previous = glGetInteger(GL_READ_FRAMEBUFFER_BINDING);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
        int name = glGetFramebufferAttachmentParameteri(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, previous);
        return name;
    }

    /**
     * Copies the depth buffer of the given GL framebuffer into a Metal texture (r32f) and returns the Metal texture id.
     * The returned texture stays valid until the next call.
     */
    public static int importDepth(int sourceFramebuffer, int width, int height) {
        ensureCommon();
        if (importProgram == 0) {
            importProgram = compileProgram(FULLSCREEN_VERT, DEPTH_IMPORT_FRAG, "depth import");
            importFbo = glGenFramebuffers();
        }
        if (importedDepth == null || importedSrcW != width || importedSrcH != height) {
            if (importedDepth != null) {
                importedDepth.free();
            }
            importedDepth = new GlTexture().storeShared(GL_R32F, width, height);
            importedSrcW = width;
            importedSrcH = height;
            int prev = glGetInteger(GL_DRAW_FRAMEBUFFER_BINDING);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, importFbo);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, importedDepth.glShareId, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev);
        }

        int depthName = getDepthAttachmentGlName(sourceFramebuffer);

        var state = new SavedState();
        state.save();
        try {
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, importFbo);
            glViewport(0, 0, width, height);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDisable(GL_STENCIL_TEST);
            glDisable(GL_SCISSOR_TEST);
            glColorMask(true, true, true, true);
            glUseProgram(importProgram);
            glBindVertexArray(vao);
            glActiveTexture(GL_TEXTURE0);
            glBindSampler(0, 0);
            glBindTexture(GL_TEXTURE_2D, depthName);
            glUniform1i(glGetUniformLocation(importProgram, "depthTex"), 0);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        } finally {
            state.restore();
        }
        //Make sure the GL writes are visible to Metal before it starts reading the surface
        glFinish();
        return importedDepth.id;
    }

    /** The Metal texture id of the last imported depth buffer */
    public static int lastImportedDepth() {
        return importedDepth == null ? 0 : importedDepth.id;
    }

    // ------------------------------------------------------------------ GL -> Metal: light map

    /** Returns a Metal texture mirroring Minecraft's light map */
    public static int lightmapTexture() {
        int glId = ((com.mojang.blaze3d.opengl.GlTexture) (Minecraft.getInstance().gameRenderer.lightTexture().getTextureView().texture())).glId();
        if (lightmap == null) {
            lightmap = new GlTexture().store(GL_RGBA8, 1, 16, 16);
            VGL.glTextureParameteri(lightmap.id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            VGL.glTextureParameteri(lightmap.id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            VGL.glTextureParameteri(lightmap.id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            VGL.glTextureParameteri(lightmap.id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            lightmapDirty = true;
        }
        if (lightmapDirty) {
            lightmapDirty = false;
            long data = MemoryUtil.nmemAllocChecked(16 * 16 * 4);
            try {
                glActiveTexture(GL_TEXTURE0);
                int previous = glGetInteger(GL_TEXTURE_BINDING_2D);
                glBindTexture(GL_TEXTURE_2D, glId);
                if (glGetTexLevelParameteri(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 16) {
                    nglGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
                    MetalNative.call(MetalNative.OP_TEX_SUBIMAGE, lightmap.id, 0, 0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, data);
                } else {
                    Logger.error("Unexpected light map size");
                }
                glBindTexture(GL_TEXTURE_2D, previous);
            } finally {
                MemoryUtil.nmemFree(data);
            }
        }
        return lightmap.id;
    }

    // ------------------------------------------------------------------ Metal -> GL: composite

    /**
     * Blends the voxy colour / depth (both IOSurface backed, rectangle textures on the GL side) into the given GL framebuffer.
     * Metal work is waited on first.
     */
    public static void composite(GlTexture colour, GlTexture depthR32, int destinationFramebuffer, int width, int height,
                                 Matrix4f invProjection, Matrix4f projection, RenderProperties properties,
                                 float[] fogEnd, float[] fogColour, boolean blend) {
        ensureCommon();
        boolean fog = fogEnd != null;
        if (compositeProgram == 0 || compositeFog != fog) {
            if (compositeProgram != 0) {
                glDeleteProgram(compositeProgram);
            }
            String defs = (properties.isZero2One() ? "#define USE_ZERO_ONE_DEPTH\n" : "") +
                    (properties.isReverseZ() ? "#define USE_REVERSE_Z\n" : "") +
                    (fog ? "#define USE_ENV_FOG\n" : "");
            compositeProgram = compileProgram(glVersion410(ShaderLoader.parse("voxy:post/macos/composite.vert"), defs),
                    glVersion410(ShaderLoader.parse("voxy:post/macos/composite.frag"), defs), "composite");
            compositeFog = fog;
            uInvProj = glGetUniformLocation(compositeProgram, "invProjMat");
            uProj = glGetUniformLocation(compositeProgram, "projMat");
            uEndParams = glGetUniformLocation(compositeProgram, "endParams");
            uFogColour = glGetUniformLocation(compositeProgram, "fogColour");
            uDepth = glGetUniformLocation(compositeProgram, "depthTex");
            uColour = glGetUniformLocation(compositeProgram, "colourTex");
        }

        //Metal must be completely done with the surfaces
        VGL.glFinish();

        var state = new SavedState();
        state.save();
        try {
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, destinationFramebuffer);
            glViewport(0, 0, width, height);
            glDisable(GL_STENCIL_TEST);
            glDisable(GL_CULL_FACE);
            glDisable(GL_SCISSOR_TEST);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(properties.closerEqualDepthCompare());
            glDepthMask(true);
            glColorMask(true, true, true, true);
            if (blend) {
                glEnable(GL_BLEND);
                glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
            } else {
                glDisable(GL_BLEND);
            }

            glUseProgram(compositeProgram);
            glBindVertexArray(vao);

            glActiveTexture(GL_TEXTURE0);
            glBindSampler(0, 0);
            glBindTexture(GL_TEXTURE_RECTANGLE, depthR32.glShareId);
            glUniform1i(uDepth, 0);
            glActiveTexture(GL_TEXTURE0 + 1);
            int prevUnit1 = glGetInteger(org.lwjgl.opengl.GL31C.GL_TEXTURE_BINDING_RECTANGLE);
            int prevSampler1 = glGetInteger(GL_SAMPLER_BINDING);
            glBindSampler(1, 0);
            glBindTexture(GL_TEXTURE_RECTANGLE, colour.glShareId);
            glUniform1i(uColour, 1);

            try (var stack = MemoryStack.stackPush()) {
                long ptr = stack.nmalloc(64);
                invProjection.getToAddress(ptr);
                nglUniformMatrix4fv(uInvProj, 1, false, ptr);
                projection.getToAddress(ptr);
                nglUniformMatrix4fv(uProj, 1, false, ptr);
            }
            if (fog) {
                glUniform4f(uEndParams, fogEnd[0], fogEnd[1], fogEnd[2], fogEnd[3]);
                glUniform4f(uFogColour, fogColour[0], fogColour[1], fogColour[2], fogColour[3]);
            }

            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            glBindTexture(GL_TEXTURE_RECTANGLE, prevUnit1);
            glBindSampler(1, prevSampler1);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_RECTANGLE, 0);
        } finally {
            state.restore();
        }
    }

    //Shader sources are loaded with a 4.6 header, rewrite it and inject the defines
    private static String glVersion410(String src, String defs) {
        int eol = src.indexOf('\n');
        return "#version 410 core\n" + defs + src.substring(eol + 1);
    }

    public static void free() {
        if (importedDepth != null) { importedDepth.free(); importedDepth = null; }
        if (lightmap != null) { lightmap.free(); lightmap = null; }
        if (importProgram != 0) { glDeleteProgram(importProgram); importProgram = 0; }
        if (compositeProgram != 0) { glDeleteProgram(compositeProgram); compositeProgram = 0; }
        if (importFbo != 0) { glDeleteFramebuffers(importFbo); importFbo = 0; }
        if (vao != 0) { glDeleteVertexArrays(vao); vao = 0; }
    }
}
