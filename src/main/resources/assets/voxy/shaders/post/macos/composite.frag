#version 460 core
//macOS composite pass, runs on Minecraft's real OpenGL 4.1 context (the loader rewrites the version).
//The voxy colour/depth results are rendered by Metal and shared with OpenGL through IOSurface backed rectangle textures.

uniform sampler2DRect depthTex;
uniform sampler2DRect colourTex;
uniform mat4 invProjMat;
uniform mat4 projMat;
#ifdef USE_ENV_FOG
uniform vec4 endParams;
uniform vec4 fogColour;
#endif

#import <voxy:util/depthutils.glsl>

out vec4 colour;
in vec2 UV;

vec3 rev3d(vec3 clip) {
    vec4 view = invProjMat * vec4(SCREEN2NDC(clip),1.0f);
    return view.xyz/view.w;
}

float projDepth(vec3 pos) {
    vec4 view = projMat * vec4(pos, 1);
    return view.z/view.w;
}

void main() {
    ivec2 texel = ivec2(UV * vec2(textureSize(depthTex)));
    float depth = texelFetch(depthTex, texel).r;
    if (depth == 0.0f || depth == 1.0f) {
        discard;
    }

    vec3 point = rev3d(vec3(UV.xy, depth));
    depth = projDepth(point);
    depth = REDUCTION2(FAR+CLOSER_SIGN*(2.0f/((1<<24)-1)), depth);
    depth = NDC2SCREEN_DEPTH(depth);

    gl_FragDepth = depth;

    colour = texelFetch(colourTex, ivec2(UV * vec2(textureSize(colourTex))));
    if (colour.a == 0.0) {
        discard;
    }
    #ifdef USE_ENV_FOG
    if (fogColour.a>0.0){
        float fogLerp = clamp(fma(length(point.xyz),endParams.x,endParams.y),0,endParams.z);
        colour.rgb = mix(colour.rgb, fogColour.rgb, fogLerp*fogColour.a);
    }
    #endif
}
