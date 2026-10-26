package me.cortex.voxy.client.core.rendering.util;


import net.minecraft.client.Minecraft;
import static me.cortex.voxy.client.core.gl.VGL.glBindSampler;
import static me.cortex.voxy.client.core.gl.VGL.glBindTextureUnit;

public class LightMapHelper {
    public static void bind(int lightingIndex) {
        glBindSampler(lightingIndex, 0);
        glBindTextureUnit(lightingIndex, me.cortex.voxy.client.core.gl.VGL.METAL ? me.cortex.voxy.client.core.gl.metal.MacInterop.lightmapTexture() : getLightmapTextureId());
    }

    public static int getLightmapTextureId() {
        return ((com.mojang.blaze3d.opengl.GlTexture)(Minecraft.getInstance().gameRenderer.lightTexture().getTextureView().texture())).glId();
    }
}