from __future__ import annotations

from typing import Any, Dict

from .graph import Graph
from .optimizer import optimize_graph
from .backends.numpy_backend import NumpyBackend


class InferenceSession:
    """Compiled inference session.

    The session owns an optimized immutable graph and backend-resident constants.
    Reuse one session for low-latency repeated inference.
    """

    def __init__(
        self,
        graph: Graph,
        backend: str = "auto",
        device: str = "auto",
        optimize: bool = True,
        dtype: str | None = None,
        output_numpy: bool = True,
        cuda_graph: bool = False,
    ) -> None:
        self.original_graph = graph
        self.graph = optimize_graph(graph) if optimize else graph.clone()
        self.backend_name = backend
        self.backend = self._make_backend(backend, device, dtype, output_numpy, cuda_graph)
        self.backend.prepare(self.graph)

    def _make_backend(self, backend: str, device: str, dtype: str | None, output_numpy: bool, cuda_graph: bool):
        if backend == "auto" and device in ("dml", "directml"):
            backend = "directml"
        if backend == "auto" and device in ("d3d12", "native_d3d12"):
            backend = "native_d3d12"
        if backend == "auto":
            return self._make_auto_backend(device, dtype, output_numpy, cuda_graph)
        if backend == "numpy":
            return NumpyBackend()
        if backend == "torch":
            from .backends.torch_backend import TorchBackend
            return TorchBackend(device=device, output_numpy=output_numpy, dtype=dtype, cuda_graph=cuda_graph)
        if backend == "directml":
            from .backends.directml_backend import DirectMLBackend
            return DirectMLBackend(output_numpy=output_numpy, dtype=dtype)
        if backend in ("native_d3d12", "d3d12"):
            from .backends.native_d3d12_backend import NativeD3D12Backend
            return NativeD3D12Backend(output_numpy=output_numpy)
        raise ValueError(f"unknown backend: {backend}")

    def _make_auto_backend(self, device: str, dtype: str | None, output_numpy: bool, cuda_graph: bool):
        if device != "auto":
            try:
                from .backends.torch_backend import TorchBackend
                return TorchBackend(device=device, output_numpy=output_numpy, dtype=dtype, cuda_graph=cuda_graph)
            except Exception:
                return NumpyBackend()

        try:
            import torch
            if torch.cuda.is_available():
                from .backends.torch_backend import TorchBackend
                return TorchBackend(device="cuda", output_numpy=output_numpy, dtype=dtype, cuda_graph=cuda_graph)
        except Exception:
            pass

        try:
            from .backends.directml_backend import DirectMLBackend
            return DirectMLBackend(output_numpy=output_numpy, dtype=dtype)
        except Exception:
            pass

        try:
            from .backends.torch_backend import TorchBackend
            return TorchBackend(device="cpu", output_numpy=output_numpy, dtype=dtype, cuda_graph=False)
        except Exception:
            return NumpyBackend()

    def info(self):
        return self.backend.info()

    def execution_plan(self):
        return getattr(self.backend, "execution_plan", None)

    def schedule(self):
        return getattr(self.backend, "schedule", None)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        missing = set(self.graph.inputs) - set(inputs)
        if missing:
            raise ValueError(f"missing inputs: {sorted(missing)}")
        return self.backend.run(inputs)

    def run_yolo_gpu(
        self,
        inputs: Dict[str, Any],
        *,
        output_name: str | None = None,
        anchors: int | None = None,
        channels: int | None = None,
        classes: int | None = None,
        max_candidates: int = 512,
        max_detections: int = 100,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        missing = set(self.graph.inputs) - set(inputs)
        if missing:
            raise ValueError(f"missing inputs: {sorted(missing)}")
        runner = getattr(self.backend, "run_yolo_gpu", None)
        if runner is None:
            raise NotImplementedError(f"backend {self.backend.info().name} does not support run_yolo_gpu")
        return runner(
            inputs,
            output_name=output_name,
            anchors=anchors,
            channels=channels,
            classes=classes,
            max_candidates=max_candidates,
            max_detections=max_detections,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
