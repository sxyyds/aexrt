from __future__ import annotations

from ctypes import byref, sizeof

import numpy as np

from aexrt import Graph, NativeCppYoloModel, NativeD3D12Device, TensorSpec, save_aexrt_engine
from aexrt.cpp_runtime import _YoloRunTiming


def test_native_yolo_last_run_timing_abi_when_available(tmp_path):
    if not NativeD3D12Device.probe().available:
        return

    graph = Graph("native_yolo_run_timing")
    graph.input("images", TensorSpec((1, 8, 4, 4), "float32"))
    graph.const("weight", np.zeros((8, 8, 1, 1), dtype="float32"))
    graph.const("bias", np.zeros((8,), dtype="float32"))
    graph.node(
        "Conv",
        "conv",
        "images",
        "weight",
        "bias",
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        dilations=[1, 1],
        group=1,
    )
    graph.reshape("output0", "conv", (1, 8, 16))
    graph.output("output0")

    engine_path = tmp_path / "run_timing.aexrt"
    save_aexrt_engine(graph, engine_path, classes=4, max_detections=4)
    runtime = NativeCppYoloModel(engine_path)
    try:
        assert hasattr(runtime.lib, "aexrt_yolo_get_last_run_timing")
        assert runtime.last_run_timing is None

        runtime.run(np.zeros((1, 8, 4, 4), dtype="float32"), max_detections=4)
        timing = runtime.last_run_timing
        assert timing is not None
        assert set(timing) == {"memcpy_ms", "submit_ms", "fence_ms", "readback_ms"}
        assert all(np.isfinite(value) and value >= 0.0 for value in timing.values())

        raw = _YoloRunTiming()
        raw.struct_size = sizeof(_YoloRunTiming)
        assert runtime.lib.aexrt_yolo_get_last_run_timing(runtime.model, byref(raw))
        assert raw.struct_size == sizeof(_YoloRunTiming)
        assert raw.valid == 1
    finally:
        runtime.close()
