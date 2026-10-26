package me.cortex.voxy.client.core.gl;

import me.cortex.voxy.client.core.gl.metal.MetalNative;
import me.cortex.voxy.common.util.TrackedObject;

import static org.lwjgl.opengl.GL11.GL_RGBA8;
import static org.lwjgl.opengl.GL11C.*;
import static org.lwjgl.opengl.GL30.GL_DEPTH24_STENCIL8;
import static org.lwjgl.opengl.GL45C.*;

public class GlTexture extends TrackedObject {
    //On Metal the id is only known once the storage has been allocated (store)
    public int id;
    //The OpenGL rectangle texture name sharing the memory of this (IOSurface backed) texture, 0 if not shared
    public int glShareId;
    private final int type;
    private int format;
    private int width;
    private int height;
    private int levels;
    private boolean hasAllocated;

    private static int COUNT;
    private static long ESTIMATED_TOTAL_SIZE;

    public GlTexture() {
        this(GL_TEXTURE_2D);
    }

    public GlTexture(int type) {
        this.id = VGL.METAL ? 0 : glCreateTextures(type);
        this.type = type;
        COUNT++;
    }

    private GlTexture(int type, boolean useGenTypes) {
        if (VGL.METAL) {
            this.id = 0;
        } else if (useGenTypes) {
            this.id = glGenTextures();
        } else {
            this.id = glCreateTextures(type);
        }
        this.type = type;
        COUNT++;
    }

    public GlTexture store(int format, int levels, int width, int height) {
        if (this.hasAllocated) {
            throw new IllegalStateException("Texture already allocated");
        }
        this.hasAllocated = true;

        this.format = format;
        if (this.type == GL_TEXTURE_2D) {
            if (VGL.METAL) {
                this.id = (int) MetalNative.call(MetalNative.OP_TEX_CREATE, width, height, levels, format, 0, 0);
                if (this.id == 0) {
                    throw new IllegalStateException("Failed to create Metal texture of format " + format);
                }
            } else {
                glTextureStorage2D(this.id, levels, format, width, height);
            }
            this.width = width;
            this.height = height;
            this.levels = levels;
        } else {
            throw new IllegalStateException("Unknown texture type");
        }
        ESTIMATED_TOTAL_SIZE += this.getEstimatedSize();
        return this;
    }

    /**
     * Metal only: allocates the storage as an IOSurface backed texture that OpenGL can sample through glShareId
     * (a GL_TEXTURE_RECTANGLE). Only GL_RGBA8 and GL_R32F are supported.
     */
    public GlTexture storeShared(int format, int width, int height) {
        if (!VGL.METAL) throw new IllegalStateException("Shared textures only exist on Metal");
        if (this.hasAllocated) {
            throw new IllegalStateException("Texture already allocated");
        }
        this.hasAllocated = true;
        this.format = format;
        long scratch = org.lwjgl.system.MemoryUtil.nmemCallocChecked(1, 8);
        try {
            this.id = (int) MetalNative.call(MetalNative.OP_TEX_CREATE, width, height, 1, format, 1, scratch);
            if (this.id == 0) {
                throw new IllegalStateException("Failed to create shared Metal texture");
            }
            this.glShareId = org.lwjgl.system.MemoryUtil.memGetInt(scratch);
        } finally {
            org.lwjgl.system.MemoryUtil.nmemFree(scratch);
        }
        this.width = width;
        this.height = height;
        this.levels = 1;
        ESTIMATED_TOTAL_SIZE += this.getEstimatedSize();
        return this;
    }

    public GlTexture createView() {
        this.assertAllocated();
        var view = new GlTexture(this.type, true);
        if (VGL.METAL) {
            view.id = (int) MetalNative.call(MetalNative.OP_TEX_VIEW, this.id, this.format, 0, 1);
        } else {
            glTextureView(view.id, this.type, this.id, this.format, 0, 1, 0, 1);
        }
        return view;
    }

    @Override
    public void free() {
        if (this.hasAllocated) {
            ESTIMATED_TOTAL_SIZE -= this.getEstimatedSize();
        }
        COUNT--;
        this.hasAllocated = false;
        super.free0();
        if (VGL.METAL) {
            if (this.glShareId != 0 || this.id != 0) MetalNative.call(MetalNative.OP_TEX_DELETE, this.id);
        } else {
            glDeleteTextures(this.id);
        }
    }

    public GlTexture name(String name) {
        this.assertAllocated();
        return GlDebug.name(name, this);
    }

    public int getWidth() {
        this.assertAllocated();
        return this.width;
    }

    public int getHeight() {
        this.assertAllocated();
        return this.height;
    }

    public int getLevels() {
        this.assertAllocated();
        return this.levels;
    }

    public int getFormat() {
        this.assertAllocated();
        return this.format;
    }

    public int getPixelTransferFormat() {
        this.assertAllocated();
        return switch (this.format) {
            case GL_RGBA8 -> GL_RGBA;
            case GL_RG16F -> GL_RG;
            case GL_R32UI -> GL_RED_INTEGER;
            case GL_R32F -> GL_RED;
            case GL_DEPTH_COMPONENT24,GL_DEPTH_COMPONENT32F,GL_DEPTH_COMPONENT32 -> GL_DEPTH_COMPONENT;
            case GL_DEPTH24_STENCIL8 -> GL_DEPTH_STENCIL;
            default -> throw new IllegalStateException("Unknown format");
        };
    }

    private long getEstimatedSize() {
        this.assertAllocated();
        long elemSize = switch (this.format) {
            case GL_R32UI, GL_RGBA8, GL_DEPTH24_STENCIL8, GL_R32F, GL_RG16F -> 4;
            case GL_DEPTH_COMPONENT24 -> 4;//TODO: check this is right????
            case GL_DEPTH_COMPONENT32F -> 4;
            case GL_DEPTH_COMPONENT32 -> 4;

            default -> throw new IllegalStateException("Unknown element size");
        };

        long size = 0;
        for (int lvl = 0; lvl < this.levels; lvl++) {
            size += Math.max((((long)this.width)>>lvl), 1) * Math.max((((long)this.height)>>lvl), 1) * elemSize;
        }
        return size;
    }

    public void assertAllocated() {
        if (!this.hasAllocated) {
            throw new IllegalStateException("Texture not yet allocated");
        }
    }

    public GlTexture zero() {
        this.assertAllocated();
        int type = switch (this.format) {
            case GL_R32UI -> GL_UNSIGNED_INT;
            case GL_RGBA8 -> GL_INT;
            case GL_R32F,GL_DEPTH_COMPONENT24,GL_DEPTH_COMPONENT32F,GL_DEPTH_COMPONENT32, GL_RG16F -> GL_FLOAT;
            case GL_DEPTH24_STENCIL8 -> GL_UNSIGNED_INT_24_8;
            case GL_DEPTH32F_STENCIL8 -> GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
            default -> throw new IllegalStateException("Unknown format");
        };
        for (int lvl = 0; lvl < this.levels; lvl++) {
            if (VGL.METAL) {
                MetalNative.call(MetalNative.OP_TEX_CLEAR, this.id, lvl, this.getPixelTransferFormat(), type, 0);
            } else {
                nglClearTexImage(this.id, lvl, this.getPixelTransferFormat(), type, 0);
            }
        }
        return this;
    }

    public static int getCount() {
        return COUNT;
    }

    public static long getEstimatedTotalSize() {
        return ESTIMATED_TOTAL_SIZE;
    }
}
