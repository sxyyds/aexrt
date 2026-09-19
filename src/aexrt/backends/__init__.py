from .base import Backend, BackendInfo
from .numpy_backend import NumpyBackend

try:
    from .torch_backend import TorchBackend
except Exception:  # torch is optional
    TorchBackend = None  # type: ignore

# Keep the DirectML bridge lazy. Some Windows environments can raise loader
# errors while importing torch-directml even when the native D3D12 path is used.
DirectMLBackend = None  # type: ignore

try:
    from .native_d3d12_backend import NativeD3D12Backend
except Exception:
    NativeD3D12Backend = None  # type: ignore

__all__ = ["Backend", "BackendInfo", "NumpyBackend", "TorchBackend", "DirectMLBackend", "NativeD3D12Backend"]
