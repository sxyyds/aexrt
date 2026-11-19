#!/usr/bin/env python3
"""Compile GLSL/Slang to SPIR-V through the Slang compiler DLL.

The NDK's glslc predates GL_KHR_cooperative_matrix; the Slang standard module
shipped inside the `slangpy` PyPI wheel bundles a current Slang compiler that
can emit SPV_KHR_cooperative_matrix. This script drives its deprecated C API
via ctypes so we can produce reference modules without the Vulkan SDK.

  py mobile/tools/slang_glsl_spv.py <slang-compiler.dll> <in.comp> <out.spv> [glsl|slang]
"""
import ctypes
import os
import sys
from pathlib import Path

SLANG_SPIRV = 6
SLANG_SOURCE_LANGUAGE_SLANG = 1
SLANG_SOURCE_LANGUAGE_GLSL = 3
SLANG_STAGE_COMPUTE = 6


def main():
    dll_path, src_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    lang = sys.argv[4] if len(sys.argv) > 4 else "glsl"
    os.add_dll_directory(str(Path(dll_path).resolve().parent))
    d = ctypes.WinDLL(dll_path)
    c_void_p, c_char_p, c_int, c_size_t = (ctypes.c_void_p, ctypes.c_char_p,
                                           ctypes.c_int, ctypes.c_size_t)
    d.spCreateSession.restype = c_void_p
    d.spCreateSession.argtypes = [c_char_p]
    d.spDestroySession.argtypes = [c_void_p]
    d.spCreateCompileRequest.restype = c_void_p
    d.spCreateCompileRequest.argtypes = [c_void_p]
    d.spDestroyCompileRequest.argtypes = [c_void_p]
    d.spAddCodeGenTarget.restype = c_int
    d.spAddCodeGenTarget.argtypes = [c_void_p, c_int]
    d.spAddTranslationUnit.restype = c_int
    d.spAddTranslationUnit.argtypes = [c_void_p, c_int, c_char_p]
    d.spAddTranslationUnitSourceString.argtypes = [c_void_p, c_int, c_char_p, c_char_p]
    d.spAddEntryPoint.restype = c_int
    d.spAddEntryPoint.argtypes = [c_void_p, c_int, c_char_p, c_int]
    d.spCompile.restype = c_int
    d.spCompile.argtypes = [c_void_p]
    d.spGetDiagnosticOutput.restype = c_char_p
    d.spGetDiagnosticOutput.argtypes = [c_void_p]
    d.spGetEntryPointCode.restype = c_void_p
    d.spGetEntryPointCode.argtypes = [c_void_p, c_int, ctypes.POINTER(c_size_t)]

    session = d.spCreateSession(None)
    if not session:
        raise SystemExit("spCreateSession failed")
    req = d.spCreateCompileRequest(session)
    target = d.spAddCodeGenTarget(req, SLANG_SPIRV)
    tu = d.spAddTranslationUnit(req, SLANG_SOURCE_LANGUAGE_GLSL if lang == "glsl"
                                     else SLANG_SOURCE_LANGUAGE_SLANG, b"shader")
    src = open(src_path, "rb").read()
    d.spAddTranslationUnitSourceString(req, tu, b"shader.comp", src)
    ep = d.spAddEntryPoint(req, tu, b"main", SLANG_STAGE_COMPUTE)
    res = d.spCompile(req)
    diag = d.spGetDiagnosticOutput(req)
    if diag:
        text = diag.decode("utf-8", "replace")
        if text.strip():
            print(text)
    if res != 0:
        d.spDestroyCompileRequest(req)
        d.spDestroySession(session)
        raise SystemExit(f"Slang compile failed ({res})")
    n = c_size_t()
    code = d.spGetEntryPointCode(req, ep, ctypes.byref(n))
    if not code or n.value == 0:
        raise SystemExit("no SPIR-V produced")
    data = ctypes.string_at(code, n.value)
    open(out_path, "wb").write(data)
    print(f"wrote {out_path}: {len(data)} bytes ({n.value // 4} words)")
    d.spDestroyCompileRequest(req)
    d.spDestroySession(session)


if __name__ == "__main__":
    main()
