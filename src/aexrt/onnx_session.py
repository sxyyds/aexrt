from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np


def _providers_for_device(device: str) -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    lowered = str(device).lower()
    if lowered in {"dml", "directml", "auto"} and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if lowered in {"cuda", "gpu", "auto"} and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@dataclass(frozen=True)
class OnnxTensorInfo:
    name: str
    shape: tuple[Any, ...]
    dtype: str


class OnnxInferenceSession:
    """Whole-ONNX execution session for production model coverage.

    This complements AEXRT's native IR path: unsupported ONNX graphs can still
    be served through the same project API while native kernels catch up.
    """

    def __init__(self, model_path: str, device: str = "auto", providers: Iterable[str] | None = None) -> None:
        import onnxruntime as ort

        self.model_path = str(model_path)
        self.providers = list(providers) if providers is not None else _providers_for_device(device)
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.inputs = tuple(OnnxTensorInfo(i.name, tuple(i.shape), i.type) for i in self.session.get_inputs())
        self.outputs = tuple(OnnxTensorInfo(o.name, tuple(o.shape), o.type) for o in self.session.get_outputs())

    def info(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "providers": self.session.get_providers(),
            "inputs": [x.__dict__ for x in self.inputs],
            "outputs": [x.__dict__ for x in self.outputs],
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        feed = {name: np.ascontiguousarray(value) for name, value in inputs.items()}
        output_names = [o.name for o in self.outputs]
        result = self.session.run(output_names, feed)
        return dict(zip(output_names, result))
