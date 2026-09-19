from __future__ import annotations

from .base import BackendInfo
from .torch_backend import TorchBackend

try:
    import torch_directml
except Exception as e:  # pragma: no cover
    torch_directml = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class DirectMLBackend(TorchBackend):
    """Experimental DirectML bridge through torch-directml.

    This gives AEXRT an immediate cross-vendor Windows GPU path while the native
    D3D12/DirectML backend is developed.
    """

    name = "directml"
    supported_dtypes = frozenset({"float16", "float32", "int32", "int64", "bool"})

    def __init__(self, output_numpy: bool = True, dtype: str | None = None) -> None:
        if torch_directml is None:
            raise RuntimeError(f"torch-directml is not available: {_IMPORT_ERROR}")
        super().__init__(
            device="directml",
            output_numpy=output_numpy,
            dtype=dtype,
            cuda_graph=False,
        )

    def info(self) -> BackendInfo:
        base = super().info()
        caps = dict(base.capabilities)
        caps.update({
            "gpu": True,
            "directml": True,
            "cuda_graph": False,
            "device_type": "directml",
            "device_name": _directml_device_name(),
            "dtypes": sorted(self.supported_dtypes),
            "features": sorted(self.supported_features - {"cuda_graph_replay"}),
        })
        return BackendInfo("directml", str(self.device), caps)


def _directml_device_name() -> str:
    if torch_directml is None:
        return "DirectML"
    try:
        return str(torch_directml.device_name(0))
    except Exception:
        return "DirectML"
