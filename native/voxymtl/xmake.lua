-- voxymtl: Metal backend that translates the subset of OpenGL 4.6 used by Voxy to Metal.
--
-- Build (on a Mac):
--   xmake f -p macosx -a arm64  -m release && xmake    (Apple silicon)
--   xmake f -p macosx -a x86_64 -m release && xmake    (Intel)
-- Output: build/macosx/<arch>/release/libvoxymtl.dylib
set_project("voxymtl")
set_version("0.1.0")
set_languages("c++20", "c17")
add_rules("mode.release", "mode.debug")

add_requires("glslang", "spirv-cross")

if is_plat("macosx") then
    set_policy("build.across_targets_in_parallel", false)
    add_mxxflags("-fobjc-arc", "-fno-objc-exceptions")
    set_toolset("cxx", "clang++")
    set_toolset("mxx", "clang++")

    target("voxymtl")
        set_kind("shared")
        set_basename("voxymtl")
        add_files("src/*.mm", "src/*.cpp")
        add_headerfiles("src/*.h")
        add_packages("glslang", "spirv-cross")
        add_frameworks("Metal", "Foundation", "QuartzCore", "IOSurface", "OpenGL", "CoreFoundation")
        add_cxflags("-fvisibility=hidden", {force = true})
        set_symbols("hidden")
        set_strip("all")
        -- minimum macOS version: 12 (MSL 2.4)
        add_cxflags("-mmacosx-version-min=12.0")
        add_ldflags("-mmacosx-version-min=12.0")
        add_shflags("-install_name @rpath/libvoxymtl.dylib")
else
    target("voxymtl")
        set_kind("phony")
        on_build(function () print("voxymtl only builds on macOS (-p macosx)") end)
end
