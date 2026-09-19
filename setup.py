from __future__ import annotations

import sys
from pathlib import Path

from setuptools import Extension, setup


ROOT = Path(__file__).parent


def ext_modules():
    if sys.platform != "win32":
        return []
    return [
        Extension(
            "aexrt_native_d3d12",
            sources=[str(ROOT / "native" / "aexrt_native_d3d12.cpp")],
            language="c++",
            extra_compile_args=["/std:c++17", "/EHsc", "/O2"],
            libraries=["d3d12", "dxgi", "dxguid", "d3dcompiler"],
        )
    ]


setup(ext_modules=ext_modules())
