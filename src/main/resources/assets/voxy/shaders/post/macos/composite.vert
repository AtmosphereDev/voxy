#version 460 core
//macOS composite pass, runs on Minecraft's real OpenGL 4.1 context (the loader rewrites the version)
#import <voxy:util/depthutils.glsl>

out vec2 UV;
void main() {
    vec2 corner = vec2(gl_VertexID&1, (gl_VertexID>>1)&1);
    gl_Position = vec4(corner * 2.0f - 1.0f, 0.0f, 1.0f);
    UV = corner;
}
