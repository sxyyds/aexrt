#!/usr/bin/env python3
"""AEXRT-M build driver.

  py mobile/tools/build.py shaders              # .comp -> .spv -> *_spv.h
  py mobile/tools/build.py android [--api 35]   # NDK cross-build for the phone
  py mobile/tools/build.py host                 # Windows/MSVC build for numeric tests
  py mobile/tools/build.py all

Every artifact stays under build/mobile. The script is stdlib-only so it works
from a plain Windows shell without the Vulkan SDK installed (NDK shader-tools
provide glslc, and a Vulkan import library is generated from the system DLL).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOBILE = ROOT / "mobile"
KERNELS = MOBILE / "kernels"
BUILD = ROOT / "build" / "mobile"

KERNEL_SOURCES = ["conv_i8.comp", "conv_quad.comp", "conv_pair.comp", "conv_f16.comp", "conv_f16_oc4.comp", "conv_f16_oc1.comp", "conv_stem.comp", "conv_wino.comp", "conv_wino2.comp",
                  "conv_i8q.comp", "conv_stem_i8q.comp", "elementwise_i8.comp", "decode_i8.comp",
                  "elementwise.comp", "decode.comp"]


def run(cmd, cwd=None, env=None):
    print("+ " + " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, encoding="mbcs",
                       errors="replace", stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    if p.stdout:
        print(p.stdout.rstrip())
    if p.returncode != 0:
        raise SystemExit("command failed (%d): %s" % (p.returncode, cmd[0]))
    return p.stdout


def find_ndk():
    for key in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "NDK_ROOT"):
        if os.environ.get(key):
            return Path(os.environ[key])
    sdk = Path(os.environ.get("ANDROID_SDK_ROOT", Path.home() / "AppData/Local/Android/Sdk"))
    candidates = sorted((sdk / "ndk").glob("*"), reverse=True)
    for c in candidates:
        if (c / "toolchains/llvm/prebuilt").exists():
            return c
    return None


def find_glslc(ndk):
    if ndk:
        p = ndk / "shader-tools/windows-x86_64/glslc.exe"
        if p.exists():
            return p
    p = shutil.which("glslc") or shutil.which("glslc.exe")
    return Path(p) if p else None


def emit_header(spv_path: Path, out_path: Path):
    data = spv_path.read_bytes()
    if len(data) % 4:
        raise SystemExit(f"{spv_path}: SPIR-V size is not a multiple of 4")
    words = ["0x%08x" % int.from_bytes(data[i:i + 4], "little")
             for i in range(0, len(data), 4)]
    name = spv_path.stem + "_spv"
    lines = [f"/* auto-generated from {spv_path.name} by mobile/tools/build.py, do not edit */",
             "#include <stdint.h>",
             f"static const uint32_t {name}[{len(words)}] = {{"]
    for i in range(0, len(words), 8):
        lines.append("    " + ", ".join(words[i:i + 8]) + ",")
    lines.append("};")
    lines.append("")
    out_path.write_text("\n".join(lines), newline="\n")
    print(f"  {out_path.relative_to(ROOT)}: {len(data)} bytes, {len(words)} words")


def cmd_shaders(args):
    ndk = find_ndk()
    glslc = Path(args.glslc) if args.glslc else find_glslc(ndk)
    if not glslc or not Path(glslc).exists():
        raise SystemExit("glslc not found: install the Vulkan SDK or the Android NDK, "
                         "or pass --glslc PATH")
    BUILD.mkdir(parents=True, exist_ok=True)
    for name in KERNEL_SOURCES:
        src = KERNELS / name
        if not src.exists():
            print(f"  skip {name} (missing)")
            continue
        spv = KERNELS / (src.stem + ".spv")
        run([str(glslc), "-fshader-stage=compute", "--target-env=vulkan1.2",
             "-O", "-o", str(spv), str(src)])
        emit_header(spv, KERNELS / (src.stem + "_spv.h"))


def cmd_api_android(args):
    """Cross-compile the in-process library + example client for Android."""
    ndk = find_ndk()
    if not ndk:
        raise SystemExit("Android NDK not found (set ANDROID_NDK_HOME)")
    binroot = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin"
    clang = binroot / f"aarch64-linux-android{args.api}-clang.cmd"
    if not clang.exists():
        clang = binroot / f"aarch64-linux-android{args.api}-clang.exe"
    if not clang.exists():
        raise SystemExit(f"NDK clang for API {args.api} not found under {binroot}")
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / "example_api_android"
    runtime = MOBILE / "runtime" / "yolo_flat.c"
    example = MOBILE / "runtime" / "example_api.c"
    run([str(clang), "-O2", "-std=c11", "-DANDROID", "-DAEXRT_M_NO_MAIN",
         "-fPIE", "-pie", "-Wall", "-Wextra", "-Wno-unused-parameter",
         "-Wno-missing-field-initializers", "-Wno-unused-function", "-Wno-unused-variable",
         "-I", str(ndk / "toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include"),
         "-I", str(MOBILE / "runtime"),
         str(runtime), str(example), "-o", str(out),
         "-lvulkan", "-landroid", "-llog"])
    print(f"android api example: {out} ({out.stat().st_size} bytes)")


def cmd_android(args):
    ndk = find_ndk()
    if not ndk:
        raise SystemExit("Android NDK not found (set ANDROID_NDK_HOME)")
    binroot = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin"
    clang = binroot / f"aarch64-linux-android{args.api}-clang.cmd"
    if not clang.exists():
        clang = binroot / f"aarch64-linux-android{args.api}-clang.exe"
    if not clang.exists():
        raise SystemExit(f"NDK clang for API {args.api} not found under {binroot}")
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / "yolo_flat"
    src = MOBILE / "runtime" / "yolo_flat.c"
    run([str(clang), "-O2", "-std=c11", "-DANDROID", "-fPIE", "-pie",
         "-Wall", "-Wextra", "-Wno-unused-parameter",
         "-Wno-missing-field-initializers",
         "-I", str(ndk / "toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include"),
         str(src), "-o", str(out), "-lvulkan", "-landroid", "-llog"])
    # adb push convenience name used by earlier phases
    shutil.copy2(out, BUILD / "yolo_flat.exe")
    print(f"android binary: {out} ({out.stat().st_size} bytes)")


def find_vs():
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        return None
    out = subprocess.run([str(vswhere), "-latest", "-products", "*",
                          "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                          "-property", "installationPath"],
                         text=True, stdout=subprocess.PIPE).stdout.strip()
    return Path(out) if out else None


def make_vulkan_lib(vs: Path):
    """Create an import library for the system vulkan-1.dll (no SDK required)."""
    outdir = BUILD / "vk"
    outdir.mkdir(parents=True, exist_ok=True)
    lib = outdir / "vulkan-1.lib"
    if lib.exists():
        return lib
    dll = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/vulkan-1.dll"
    if not dll.exists():
        raise SystemExit("vulkan-1.dll not found; install a Vulkan-capable driver")
    bat = outdir / "mk_vk_lib.bat"
    bat.write_text(
        "@echo off\n"
        f'call "{vs}\\Common7\\Tools\\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul\n'
        f'dumpbin /nologo /exports "{dll}" > "{outdir / "exports.txt"}"\n'
        "if errorlevel 1 exit /b 1\n"
        f'py -3 "{MOBILE / "tools" / "mk_vk_def.py"}" "{outdir / "exports.txt"}" "{outdir / "vulkan-1.def"}"\n'
        "if errorlevel 1 exit /b 1\n"
        f'lib /nologo /def:"{outdir / "vulkan-1.def"}" /machine:x64 /out:"{lib}"\n',
        newline="\r\n", encoding="mbcs")
    run([str(bat)])
    if not lib.exists():
        raise SystemExit("failed to create vulkan-1.lib")
    return lib


def cmd_api_host(args):
    """Build the in-process library (yolo_flat.c without main) + example client."""
    vs = find_vs()
    if not vs:
        raise SystemExit("Visual Studio C++ tools not found")
    ndk = find_ndk()
    if not ndk:
        raise SystemExit("NDK Vulkan headers not found (set ANDROID_NDK_HOME)")
    inc = BUILD / "vk" / "include"
    if not (inc / "vulkan" / "vulkan.h").exists():
        inc.mkdir(parents=True, exist_ok=True)
        hdr = ndk / "toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include"
        shutil.copytree(hdr / "vulkan", inc / "vulkan")
        shutil.copytree(hdr / "vk_video", inc / "vk_video")
    lib = make_vulkan_lib(vs)
    BUILD.mkdir(parents=True, exist_ok=True)
    src = MOBILE / "runtime" / "yolo_flat.c"
    ex = MOBILE / "runtime" / "example_api.c"
    obj = BUILD / "aexrt_m_api.obj"
    out = BUILD / "example_api.exe"
    bat = BUILD / "build_api_host.bat"
    bat.write_text(
        "@echo off\n"
        f'call "{vs}\\Common7\\Tools\\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul\n'
        f'cl /nologo /O2 /W3 /utf-8 /D_CRT_SECURE_NO_WARNINGS /D_WIN32_WINNT=0x0A00 '
        f'/DAEXRT_M_NO_MAIN /I "{inc}" "{src}" /Fo:"{obj}" /c\n'
        f'if errorlevel 1 exit /b 1\n'
        f'cl /nologo /O2 /W3 /utf-8 /D_CRT_SECURE_NO_WARNINGS /D_WIN32_WINNT=0x0A00 '
        f'/I "{MOBILE / "runtime"}" "{ex}" /Fo:"{BUILD / "example_api.obj"}" '
        f'/Fe:"{out}" "{obj}" /link "{lib}"\n',
        newline="\r\n", encoding="mbcs")
    run([str(bat)])
    print(f"api example: {out}")


def cmd_lib_android(args):
    """Build libruntime as a shared object for Android apps (no main)."""
    ndk = find_ndk()
    if not ndk:
        raise SystemExit("Android NDK not found (set ANDROID_NDK_HOME)")
    binroot = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin"
    clang = binroot / f"aarch64-linux-android{args.api}-clang.cmd"
    if not clang.exists():
        clang = binroot / f"aarch64-linux-android{args.api}-clang.exe"
    if not clang.exists():
        raise SystemExit(f"NDK clang for API {args.api} not found under {binroot}")
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / "libaexrt_m.so"
    src = MOBILE / "runtime" / "yolo_flat.c"
    run([str(clang), "-O2", "-std=c11", "-DANDROID", "-DAEXRT_M_NO_MAIN",
         "-fPIC", "-shared", "-Wall", "-Wextra", "-Wno-unused-parameter",
         "-Wno-missing-field-initializers", "-Wno-unused-function", "-Wno-unused-variable",
         "-I", str(ndk / "toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include"),
         "-I", str(MOBILE / "runtime"),
         str(src), "-o", str(out), "-lvulkan", "-landroid"])
    print(f"android shared lib: {out} ({out.stat().st_size} bytes)")
    # dlopen smoke test (host-independent: run it on the device with adb)
    dl = BUILD / "dl_test"
    run([str(clang), "-O2", "-std=c11", "-DANDROID", "-fPIE", "-pie",
         "-I", str(MOBILE / "runtime"), str(MOBILE / "runtime" / "dl_test.c"),
         "-o", str(dl), "-ldl"])
    print(f"dlopen test: {dl}")


def cmd_host(args):
    vs = find_vs()
    if not vs:
        raise SystemExit("Visual Studio C++ tools not found")
    ndk = find_ndk()
    if not ndk:
        raise SystemExit("NDK Vulkan headers not found (set ANDROID_NDK_HOME)")
    # Only the Vulkan headers are portable; pointing -I at the whole NDK
    # sysroot shadows MSVC's own stdint.h/linux headers.
    inc = BUILD / "vk" / "include"
    if not (inc / "vulkan" / "vulkan.h").exists():
        inc.mkdir(parents=True, exist_ok=True)
        hdr = ndk / "toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include"
        shutil.copytree(hdr / "vulkan", inc / "vulkan")
        shutil.copytree(hdr / "vk_video", inc / "vk_video")
    lib = make_vulkan_lib(vs)
    BUILD.mkdir(parents=True, exist_ok=True)
    src = MOBILE / "runtime" / "yolo_flat.c"
    out = BUILD / "yolo_flat_host.exe"
    bat = BUILD / "build_host.bat"
    bat.write_text(
        "@echo off\n"
        f'call "{vs}\\Common7\\Tools\\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul\n'
        f'cl /nologo /O2 /W3 /utf-8 /D_CRT_SECURE_NO_WARNINGS /D_WIN32_WINNT=0x0A00 /I "{inc}" '
        f'"{src}" /Fo:"{BUILD / "host.obj"}" /Fe:"{out}" /link "{lib}"\n',
        newline="\r\n", encoding="mbcs")
    run([str(bat)])
    print(f"host binary: {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("shaders")
    sp.add_argument("--glslc")
    sp.set_defaults(fn=cmd_shaders)
    sp = sub.add_parser("android")
    sp.add_argument("--api", default="35")
    sp.set_defaults(fn=cmd_android)
    sub.add_parser("host").set_defaults(fn=cmd_host)
    sub.add_parser("api-host").set_defaults(fn=cmd_api_host)
    sp = sub.add_parser("api-android")
    sp.add_argument("--api", default="35")
    sp.set_defaults(fn=cmd_api_android)
    sp = sub.add_parser("lib-android")
    sp.add_argument("--api", default="35")
    sp.set_defaults(fn=cmd_lib_android)

    def all_(a):
        cmd_shaders(a)
        cmd_android(a)
        cmd_host(a)
        cmd_api_host(a)
        cmd_api_android(a)
        cmd_lib_android(a)
    sp = sub.add_parser("all")
    sp.add_argument("--api", default="35")
    sp.add_argument("--glslc")
    sp.set_defaults(fn=all_)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
