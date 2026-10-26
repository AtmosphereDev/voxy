#!/usr/bin/env bash
# Builds libvoxymtl.dylib for Apple silicon and Intel and places them where gradle packs them into the mod jar
# (native/dist/macos-<arch>/libvoxymtl.dylib). Needs a Mac with xcode command line tools and xmake.
set -euo pipefail
cd "$(dirname "$0")/voxymtl"
for pair in "arm64:arm64" "x86_64:x64"; do
    xarch="${pair%%:*}"
    jarch="${pair##*:}"
    xmake f -y -p macosx -a "$xarch" -m release
    xmake -y
    out="../dist/macos-$jarch"
    mkdir -p "$out"
    cp "build/macosx/$xarch/release/libvoxymtl.dylib" "$out/libvoxymtl.dylib"
done
echo "natives written to native/dist"
