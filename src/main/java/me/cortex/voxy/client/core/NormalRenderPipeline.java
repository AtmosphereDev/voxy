package me.cortex.voxy.client.core;

import me.cortex.voxy.client.config.VoxyConfig;
import me.cortex.voxy.client.core.gl.GlFramebuffer;
import me.cortex.voxy.client.core.gl.GlTexture;
import me.cortex.voxy.client.core.gl.VGL;
import me.cortex.voxy.client.core.gl.metal.MacInterop;
import me.cortex.voxy.client.core.gl.metal.MetalNative;
import me.cortex.voxy.client.core.gl.shader.Shader;
import me.cortex.voxy.client.core.gl.shader.ShaderType;
import me.cortex.voxy.client.core.rendering.Viewport;
import me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager;
import me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser;
import me.cortex.voxy.client.core.rendering.hierachical.NodeCleaner;
import me.cortex.voxy.client.core.rendering.post.FullscreenBlit;
import me.cortex.voxy.client.core.rendering.util.DepthFramebuffer;
import me.cortex.voxy.client.core.util.GPUTiming;
import net.minecraft.client.Minecraft;
import org.joml.Matrix4f;
import org.lwjgl.system.MemoryStack;

import java.util.List;
import java.util.function.BooleanSupplier;

import static org.lwjgl.opengl.GL30.GL_DEPTH_ATTACHMENT;
import static org.lwjgl.opengl.GL30.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME;
import static org.lwjgl.opengl.GL30C.*;
import static org.lwjgl.opengl.GL33C.*;
import static org.lwjgl.opengl.GL43.GL_DEPTH_STENCIL_TEXTURE_MODE;
import static org.lwjgl.opengl.GL45.glGetNamedFramebufferAttachmentParameteri;
import static me.cortex.voxy.client.core.gl.VGL.glBindFramebuffer;
import static me.cortex.voxy.client.core.gl.VGL.glBindTextureUnit;
import static me.cortex.voxy.client.core.gl.VGL.glBlendFuncSeparate;
import static me.cortex.voxy.client.core.gl.VGL.glDisable;
import static me.cortex.voxy.client.core.gl.VGL.glEnable;
import static me.cortex.voxy.client.core.gl.VGL.glTextureParameterf;
import static me.cortex.voxy.client.core.gl.VGL.glUniform4f;

public class NormalRenderPipeline extends AbstractRenderPipeline {
    private GlTexture colourTex;
    private GlTexture colourSSAOTex;
    private GlTexture depthShared;//macOS: the depth of voxy as an r32f texture OpenGL can read
    private final GlFramebuffer fbSSAO = new GlFramebuffer();

    private final boolean useEnvFog;
    private final FullscreenBlit finalBlit;

    private final SSAO ssao;

    protected NormalRenderPipeline(RenderProperties properties, AsyncNodeManager nodeManager, NodeCleaner nodeCleaner, HierarchicalOcclusionTraverser traversal, BooleanSupplier frexSupplier) {
        super(properties, nodeManager, nodeCleaner, traversal, frexSupplier, false);
        this.useEnvFog = VoxyConfig.CONFIG.useEnvironmentalFog;
        //On macOS the final composite is done by OpenGL (MacInterop), the metal translated blit is not needed
        this.finalBlit = VGL.METAL ? null : new FullscreenBlit(properties, "voxy:post/blit_texture_depth_cutout.frag",
                a->a.defineIf("USE_ENV_FOG", this.useEnvFog).define("EMIT_COLOUR"));


        this.ssao = SSAO.createSSAO(properties, VoxyConfig.CONFIG.getSSAOMode());
    }

    @Override
    protected int setup(Viewport<?> viewport, int sourceFB, int srcWidth, int srcHeight) {
        if (this.colourTex == null || this.colourTex.getHeight() != viewport.height || this.colourTex.getWidth() != viewport.width) {
            if (this.colourTex != null) {
                this.colourTex.free();
                this.colourSSAOTex.free();
                if (this.depthShared != null) {
                    this.depthShared.free();
                    this.depthShared = null;
                }
            }
            this.fb.resize(viewport.width, viewport.height);

            this.colourTex = new GlTexture().store(GL_RGBA8, 1, viewport.width, viewport.height);
            if (VGL.METAL) {
                //These two are shared with OpenGL for the final composite
                this.colourSSAOTex = new GlTexture().storeShared(GL_RGBA8, viewport.width, viewport.height);
                this.depthShared = new GlTexture().storeShared(GL_R32F, viewport.width, viewport.height);
            } else {
                this.colourSSAOTex = new GlTexture().store(GL_RGBA8, 1, viewport.width, viewport.height);
            }

            this.fb.framebuffer.bind(GL_COLOR_ATTACHMENT0, this.colourTex).verify();
            this.fbSSAO.bind(this.fb.getDepthAttachmentType(), this.fb.getDepthTex()).bind(GL_COLOR_ATTACHMENT0, this.colourSSAOTex).verify();


            glTextureParameterf(this.colourTex.id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTextureParameterf(this.colourTex.id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTextureParameterf(this.colourSSAOTex.id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTextureParameterf(this.colourSSAOTex.id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTextureParameterf(this.fb.getDepthTex().id, GL_DEPTH_STENCIL_TEXTURE_MODE, GL_DEPTH_COMPONENT);
        }

        this.initDepthStencil(sourceFB, this.fb.framebuffer.id, viewport.width, viewport.height, viewport.width, viewport.height);

        return this.fb.getDepthTex().id;
    }

    @Override
    protected void postOpaquePreTranslucent(Viewport<?> viewport, int sourceFrameBuffer) {
        GPUTiming.INSTANCE.marker("ao");
        this.ssao.computeSSAO(viewport, this.colourSSAOTex, this.colourTex, this.fb.getDepthTex(), sourceFrameBuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, this.fbSSAO.id);
    }

    @Override
    protected void finish(Viewport<?> viewport, int sourceFrameBuffer, int srcWidth, int srcHeight) {
        if (VGL.METAL) {
            this.finishMetal(viewport, sourceFrameBuffer, srcWidth, srcHeight);
            return;
        }
        this.finalBlit.bind();

        boolean fogCoversAllRendering = viewport.fogParameters.environmentalEnd()<Minecraft.getInstance().gameRenderer.getRenderDistance();

        if (this.useEnvFog) {
            float start = viewport.fogParameters.environmentalStart();
            float end = viewport.fogParameters.environmentalEnd();
            if (Math.abs(end-start)>1) {
                float invEndFogDelta = 1f / (end - start);
                float endDistance = Math.max(Minecraft.getInstance().gameRenderer.getRenderDistance(), 20*16);//TODO: make this constant a config option
                endDistance *= (float)Math.sqrt(3);
                float startDelta = -start * invEndFogDelta;
                glUniform4f(4, invEndFogDelta, startDelta, Math.clamp(endDistance*invEndFogDelta+startDelta, 0, 1),0);//
                glUniform4f(5, viewport.fogParameters.red(), viewport.fogParameters.green(), viewport.fogParameters.blue(), viewport.fogParameters.alpha());
            } else {
                glUniform4f(4, 0, 0, 0, 0);
                glUniform4f(5, 0, 0, 0, 0);
            }
        }

        glBindTextureUnit(3, this.colourSSAOTex.id);

        //Do alpha blending
        //Unbelievably jank hack, only blit out to the framebuffer if we are rendering fog
        if (!fogCoversAllRendering) {
            glEnable(GL_BLEND);
            glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
            AbstractRenderPipeline.transformBlitDepth(this.finalBlit, this.fb.getDepthTex().id, sourceFrameBuffer, viewport, new Matrix4f(viewport.vanillaProjection).mul(viewport.modelView));
            glDisable(GL_BLEND);
        } else {
            glDisable(GL_STENCIL_TEST);
            glDisable(GL_DEPTH_TEST);
        }
        //glBlitNamedFramebuffer(this.fbSSAO.id, sourceFrameBuffer, 0,0, viewport.width, viewport.height, 0,0, viewport.width, viewport.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    //macOS: Metal has produced colour + depth, blend it into the framebuffer of minecraft with OpenGL
    private void finishMetal(Viewport<?> viewport, int sourceFrameBuffer, int srcWidth, int srcHeight) {
        boolean fogCoversAllRendering = viewport.fogParameters.environmentalEnd()<Minecraft.getInstance().gameRenderer.getRenderDistance();
        if (fogCoversAllRendering) {
            return;
        }
        float[] fogEnd = null;
        float[] fogColour = null;
        if (this.useEnvFog) {
            fogEnd = new float[4];
            fogColour = new float[4];
            float start = viewport.fogParameters.environmentalStart();
            float end = viewport.fogParameters.environmentalEnd();
            if (Math.abs(end-start)>1) {
                float invEndFogDelta = 1f / (end - start);
                float endDistance = Math.max(Minecraft.getInstance().gameRenderer.getRenderDistance(), 20*16);
                endDistance *= (float)Math.sqrt(3);
                float startDelta = -start * invEndFogDelta;
                fogEnd[0] = invEndFogDelta;
                fogEnd[1] = startDelta;
                fogEnd[2] = Math.clamp(endDistance*invEndFogDelta+startDelta, 0, 1);
                fogColour[0] = viewport.fogParameters.red();
                fogColour[1] = viewport.fogParameters.green();
                fogColour[2] = viewport.fogParameters.blue();
                fogColour[3] = viewport.fogParameters.alpha();
            }
        }
        MetalNative.call(MetalNative.OP_TEX_DEPTH_TO_R32F, this.fb.getDepthTex().id, this.depthShared.id);
        MacInterop.composite(this.colourSSAOTex, this.depthShared, sourceFrameBuffer, srcWidth, srcHeight,
                new Matrix4f(viewport.MVP).invert(),
                new Matrix4f(viewport.vanillaProjection).mul(viewport.modelView),
                this.properties, fogEnd, fogColour, true);
    }

    @Override
    public void setupAndBindOpaque(Viewport<?> viewport) {
        this.fb.bind();
    }

    @Override
    public void setupAndBindTranslucent(Viewport<?> viewport) {
        glBindFramebuffer(GL_FRAMEBUFFER, this.fbSSAO.id);
    }

    @Override
    public void free() {
        if (this.finalBlit != null) {
            this.finalBlit.delete();
        }
        if (this.depthShared != null) {
            this.depthShared.free();
        }
        this.ssao.free();
        this.fbSSAO.free();
        if (this.colourTex != null) {
            this.colourTex.free();
            this.colourSSAOTex.free();
        }
        super.free0();
    }

    @Override
    public void addDebug(List<String> debug) {
        super.addDebug(debug);
        this.ssao.addDebugInfo(debug);
    }
}
