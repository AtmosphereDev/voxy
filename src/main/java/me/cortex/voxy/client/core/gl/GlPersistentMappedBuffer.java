package me.cortex.voxy.client.core.gl;

import me.cortex.voxy.client.core.gl.metal.MetalNative;
import me.cortex.voxy.common.util.TrackedObject;

import static org.lwjgl.opengl.ARBMapBufferRange.GL_MAP_WRITE_BIT;
import static org.lwjgl.opengl.GL15.glDeleteBuffers;
import static org.lwjgl.opengl.GL45C.*;

public class GlPersistentMappedBuffer extends TrackedObject {
    public final int id;
    private final long size;
    private final long addr;
    public GlPersistentMappedBuffer(long size, int flags) {
        this.size = size;
        if (VGL.METAL) {
            //Metal shared storage buffers are persistently mapped and coherent by nature
            this.id = MetalNative.createBuffer(size, true);
            this.addr = MetalNative.call(MetalNative.OP_BUF_PTR, this.id);
        } else {
            this.id = glCreateBuffers();
            glNamedBufferStorage(this.id, size, GL_MAP_PERSISTENT_BIT|(flags&(GL_MAP_COHERENT_BIT|GL_MAP_WRITE_BIT|GL_MAP_READ_BIT|GL_CLIENT_STORAGE_BIT)));
            this.addr = nglMapNamedBufferRange(this.id, 0, size, (flags&(GL_MAP_WRITE_BIT|GL_MAP_READ_BIT|GL_MAP_UNSYNCHRONIZED_BIT|GL_MAP_FLUSH_EXPLICIT_BIT))|GL_MAP_PERSISTENT_BIT);
        }
    }

    @Override
    public void free() {
        this.free0();
        if (VGL.METAL) {
            MetalNative.deleteBuffer(this.id);
        } else {
            glUnmapNamedBuffer(this.id);
            glDeleteBuffers(this.id);
        }
    }

    public long size() {
        return this.size;
    }

    public long addr() {
        return this.addr;
    }

    public GlPersistentMappedBuffer name(String name) {
        return GlDebug.name(name, this);
    }
}
