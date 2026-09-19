from __future__ import annotations

import os
from ctypes import POINTER, Structure, byref, c_char_p, c_double, c_float, c_int, c_ubyte, c_uint32, c_uint64, c_void_p, cdll, sizeof
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .abi import export_graph_abi, load_graph_abi
from .graph import Graph
from .yolo import Detection


class _NodeDesc(Structure):
    _fields_ = [
        ("op", c_int),
        ("input0", c_uint32),
        ("input1", c_uint32),
        ("output", c_uint32),
    ]


class _ScalarConstantDesc(Structure):
    _fields_ = [
        ("value", c_uint32),
        ("scalar", c_float),
    ]


class _GraphDesc(Structure):
    _fields_ = [
        ("element_count", c_uint64),
        ("input_count", c_uint32),
        ("node_count", c_uint32),
        ("nodes", POINTER(_NodeDesc)),
        ("output_value", c_uint32),
        ("constant_count", c_uint32),
        ("constants", POINTER(_ScalarConstantDesc)),
    ]


class _YoloDetection(Structure):
    _fields_ = [
        ("x1", c_float),
        ("y1", c_float),
        ("x2", c_float),
        ("y2", c_float),
        ("score", c_float),
        ("class_id", c_float),
    ]


class _D3D12Capabilities(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("highest_shader_model", c_uint32),
        ("native_fp16_supported", c_uint32),
        ("wave_mma_tier", c_uint32),
        ("dxc_available", c_uint32),
        ("reserved", c_uint32 * 3),
    ]


class _YoloRunTiming(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("valid", c_uint32),
        ("memcpy_ms", c_double),
        ("submit_ms", c_double),
        ("fence_ms", c_double),
        ("readback_ms", c_double),
        ("reserved", c_uint64 * 4),
    ]


def _get_device_capabilities(lib, device) -> dict[str, int | bool]:
    raw = _D3D12Capabilities()
    if not lib.aexrt_d3d12_get_capabilities(device, byref(raw)):
        raise RuntimeError("failed to query native C++ AEXRT D3D12 capabilities")
    return {
        "highest_shader_model": int(raw.highest_shader_model),
        "native_fp16_supported": bool(raw.native_fp16_supported),
        "wave_mma_tier": int(raw.wave_mma_tier),
        "dxc_available": bool(raw.dxc_available),
    }


OP_KIND = {
    "Relu": 1,
    "Gelu": 2,
    "Add": 3,
    "Sub": 4,
    "Mul": 5,
    "Div": 6,
    "Sigmoid": 7,
    "Tanh": 8,
}


class NativeCppGraph:
    def __init__(self, graph: Graph, adapter_index: int = 0) -> None:
        self.abi = export_graph_abi(graph)
        self._open(adapter_index)
        self._compile_from_abi()

    @classmethod
    def load_aexrt(cls, path: str | os.PathLike[str], adapter_index: int = 0) -> "NativeCppGraph":
        obj = cls.__new__(cls)
        obj.abi = load_graph_abi(str(path))
        obj._open(adapter_index)
        encoded_path = os.fsencode(path)
        obj.graph = obj.lib.aexrt_load_graph_json(obj.device, encoded_path)
        if not obj.graph:
            obj.close()
            raise RuntimeError(f"failed to load native C++ AEXRT graph: {path}")
        obj.closed = False
        return obj

    def _compile_from_abi(self) -> None:
        self._nodes = (_NodeDesc * len(self.abi["nodes"]))(
            *[
                _NodeDesc(OP_KIND[n["op"]], int(n["input0"]), int(n["input1"]), int(n["output"]))
                for n in self.abi["nodes"]
            ]
        )
        abi_constants = self.abi.get("constants", [])
        self._constants = (_ScalarConstantDesc * len(abi_constants))(
            *[
                _ScalarConstantDesc(int(c["value"]), float(c["scalar"]))
                for c in abi_constants
            ]
        )
        constants_ptr = self._constants if len(abi_constants) else None
        desc = _GraphDesc(
            int(self.abi["element_count"]),
            len(self.abi["inputs"]),
            len(self.abi["nodes"]),
            self._nodes,
            int(self.abi["output"]["value"]),
            len(abi_constants),
            constants_ptr,
        )
        self.graph = self.lib.aexrt_compile_graph(self.device, byref(desc))
        if not self.graph:
            self.close()
            raise RuntimeError("failed to compile native C++ AEXRT graph")
        self.closed = False

    def _open(self, adapter_index: int) -> None:
        self.lib = _load_native_cpp()
        self._configure()
        self.device = self.lib.aexrt_d3d12_create_device(c_uint32(adapter_index))
        if not self.device:
            raise RuntimeError("failed to create native C++ AEXRT D3D12 device")
        self.graph = None
        self.closed = False

    def run(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        input_names = [x["name"] for x in self.abi["inputs"]]
        feeds = [np.ascontiguousarray(inputs[name], dtype=np.float32).reshape(-1) for name in input_names]
        element_count = int(self.abi["element_count"])
        for feed in feeds:
            if feed.size != element_count:
                raise ValueError(f"expected {element_count} elements, got {feed.size}")

        out_value = self.abi["values"][int(self.abi["output"]["value"])]
        out = np.empty(element_count, dtype=np.float32)
        input_ptrs = (POINTER(c_float) * len(feeds))(*[_ptr(feed) for feed in feeds])
        ok = self.lib.aexrt_run_n(self.device, self.graph, input_ptrs, c_uint32(len(feeds)), _ptr(out))
        if not ok:
            raise RuntimeError("native C++ AEXRT graph run failed")
        return {self.abi["output"]["name"]: out.reshape(tuple(out_value["shape"])).copy()}

    @property
    def dispatch_count(self) -> int:
        if not getattr(self, "graph", None):
            return 0
        return int(self.lib.aexrt_graph_dispatch_count(self.graph))

    @property
    def buffer_count(self) -> int:
        if not getattr(self, "graph", None):
            return 0
        return int(self.lib.aexrt_graph_buffer_count(self.graph))

    @property
    def capabilities(self) -> dict[str, int | bool]:
        return _get_device_capabilities(self.lib, self.device)

    def close(self) -> None:
        if getattr(self, "graph", None):
            self.lib.aexrt_destroy_graph(self.graph)
            self.graph = None
        if getattr(self, "device", None):
            self.lib.aexrt_d3d12_destroy_device(self.device)
            self.device = None
        self.closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _configure(self) -> None:
        self.lib.aexrt_d3d12_create_device.argtypes = [c_uint32]
        self.lib.aexrt_d3d12_create_device.restype = c_void_p
        self.lib.aexrt_d3d12_get_capabilities.argtypes = [c_void_p, POINTER(_D3D12Capabilities)]
        self.lib.aexrt_d3d12_get_capabilities.restype = c_int
        self.lib.aexrt_d3d12_destroy_device.argtypes = [c_void_p]
        self.lib.aexrt_compile_graph.argtypes = [c_void_p, POINTER(_GraphDesc)]
        self.lib.aexrt_compile_graph.restype = c_void_p
        self.lib.aexrt_load_graph_json.argtypes = [c_void_p, c_char_p]
        self.lib.aexrt_load_graph_json.restype = c_void_p
        self.lib.aexrt_graph_dispatch_count.argtypes = [c_void_p]
        self.lib.aexrt_graph_dispatch_count.restype = c_uint32
        self.lib.aexrt_graph_buffer_count.argtypes = [c_void_p]
        self.lib.aexrt_graph_buffer_count.restype = c_uint32
        self.lib.aexrt_destroy_graph.argtypes = [c_void_p]
        self.lib.aexrt_run.argtypes = [c_void_p, c_void_p, POINTER(c_float), POINTER(c_float)]
        self.lib.aexrt_run.restype = c_int
        self.lib.aexrt_run2.argtypes = [c_void_p, c_void_p, POINTER(c_float), POINTER(c_float), POINTER(c_float)]
        self.lib.aexrt_run2.restype = c_int
        self.lib.aexrt_run_n.argtypes = [c_void_p, c_void_p, POINTER(POINTER(c_float)), c_uint32, POINTER(c_float)]
        self.lib.aexrt_run_n.restype = c_int


class NativeCppYoloModel:
    def __init__(self, package_path: str | os.PathLike[str], adapter_index: int = 0) -> None:
        self.package_path = str(package_path)
        self.lib = _load_native_cpp()
        self._configure()
        self.device = self.lib.aexrt_d3d12_create_device(c_uint32(adapter_index))
        if not self.device:
            raise RuntimeError("failed to create native C++ AEXRT D3D12 device")
        encoded_path = os.fsencode(package_path)
        lower_path = str(package_path).lower()
        if lower_path.endswith(".onnx"):
            self.model = self.lib.aexrt_yolo_compile_from_onnx(self.device, encoded_path)
        elif lower_path.endswith(".aexrt"):
            self.model = self.lib.aexrt_yolo_load_engine(self.device, encoded_path)
        else:
            self.model = self.lib.aexrt_yolo_compile_from_package(self.device, encoded_path)
        if not self.model:
            self.close()
            raise RuntimeError(f"failed to compile native C++ AEXRT YOLO model: {package_path}")
        self.closed = False

    @property
    def input_element_count(self) -> int:
        return int(self.lib.aexrt_yolo_input_element_count(self.model))

    @property
    def class_count(self) -> int:
        return int(self.lib.aexrt_yolo_class_count(self.model))

    @property
    def anchor_count(self) -> int:
        return int(self.lib.aexrt_yolo_anchor_count(self.model))

    @property
    def channel_count(self) -> int:
        return int(self.lib.aexrt_yolo_channel_count(self.model))

    @property
    def output_layout(self) -> int:
        return int(self.lib.aexrt_yolo_output_layout(self.model))

    @property
    def has_objectness(self) -> bool:
        return bool(self.lib.aexrt_yolo_has_objectness(self.model))

    @property
    def package_mode(self) -> int:
        return int(self.lib.aexrt_yolo_package_mode(self.model))

    @property
    def executable(self) -> bool:
        return bool(self.lib.aexrt_yolo_is_executable(self.model))

    @property
    def capabilities(self) -> dict[str, int | bool]:
        return _get_device_capabilities(self.lib, self.device)

    @property
    def graph_node_count(self) -> int:
        return int(self.lib.aexrt_yolo_graph_node_count(self.model))

    @property
    def graph_value_count(self) -> int:
        return int(self.lib.aexrt_yolo_graph_value_count(self.model))

    @property
    def constant_count(self) -> int:
        return int(self.lib.aexrt_yolo_constant_count(self.model))

    @property
    def prepared_command_count(self) -> int:
        return int(self.lib.aexrt_yolo_prepared_command_count(self.model))

    @property
    def supported_prepared_command_count(self) -> int:
        return int(self.lib.aexrt_yolo_supported_prepared_command_count(self.model))

    @property
    def unsupported_prepared_command_count(self) -> int:
        return int(self.lib.aexrt_yolo_unsupported_prepared_command_count(self.model))

    @property
    def prepared_skipped_command_count(self) -> int:
        return int(self.lib.aexrt_yolo_prepared_skipped_command_count(self.model))

    @property
    def late_concat_conv1x1_fusion_count(self) -> int:
        return int(self.lib.aexrt_yolo_late_concat_conv1x1_fusion_count(self.model))

    @property
    def concat_residual_conv1x1_fusion_count(self) -> int:
        return int(self.lib.aexrt_yolo_concat_residual_conv1x1_fusion_count(self.model))

    @property
    def paired_conv3x3_fusion_count(self) -> int:
        return int(self.lib.aexrt_yolo_paired_conv3x3_fusion_count(self.model))

    @property
    def c2f_bottleneck_superblock_count(self) -> int:
        return int(self.lib.aexrt_yolo_c2f_bottleneck_superblock_count(self.model))

    @property
    def c2f_tail_residual_fusion_count(self) -> int:
        return int(self.lib.aexrt_yolo_c2f_tail_residual_fusion_count(self.model))

    @property
    def head_fusion_enabled(self) -> bool:
        return bool(self.lib.aexrt_yolo_head_fusion_enabled(self.model))

    @property
    def head_final_conv_fusion_enabled(self) -> bool:
        return bool(self.lib.aexrt_yolo_head_final_conv_fusion_enabled(self.model))

    @property
    def tileflow_conv1x1_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_conv1x1_count(self.model))

    @property
    def tileflow_3x3_spatial_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_spatial_count(self.model))

    @property
    def tileflow_3x3_pack4_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_pack4_count(self.model))

    @property
    def tileflow_3x3_pack8_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_pack8_count(self.model))

    @property
    def tileflow_3x3_implicit_gemm_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_count(self.model))

    @property
    def tileflow_3x3_implicit_gemm_40x40_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_40x40_count(self.model))

    @property
    def tileflow_3x3_implicit_gemm_20x20_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_20x20_count(self.model))

    @property
    def tileflow_3x3_implicit_gemm_10x10_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_10x10_count(self.model))

    @property
    def tileflow_3x3_exact_40x40_64x64_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_exact_40x40_64x64_count(self.model))

    @property
    def tileflow_3x3_exact_20x20_64x64_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_exact_20x20_64x64_count(self.model))

    @property
    def tileflow_3x3_exact_10x10_128x128_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_exact_10x10_128x128_count(self.model))

    @property
    def tileflow_3x3_exact_20x20_128x64_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_exact_20x20_128x64_count(self.model))

    @property
    def tileflow_3x3_exact_10x10_256x64_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_3x3_exact_10x10_256x64_count(self.model))

    @property
    def tileflow_native_fp16_count(self) -> int:
        return int(self.lib.aexrt_yolo_tileflow_native_fp16_count(self.model))

    def run(self, output0: Any, max_detections: int = 100) -> list[Detection]:
        if not self.executable:
            raise RuntimeError("native C++ AEXRT YOLO package metadata loaded, but this package is not executable by the current pure C++ runtime yet")
        arr = np.ascontiguousarray(output0, dtype=np.float32).reshape(-1)
        expected = self.input_element_count
        if arr.size != expected:
            raise ValueError(f"expected {expected} YOLO output elements, got {arr.size}")

        raw_dets = (_YoloDetection * int(max_detections))()
        count = c_uint32(0)
        ok = self.lib.aexrt_yolo_run(
            self.device,
            self.model,
            _ptr(arr),
            c_uint64(arr.size),
            raw_dets,
            c_uint32(max_detections),
            byref(count),
        )
        if not ok:
            # 读取运行时 fail-closed 的具体原因（aexrt_yolo_get_last_error）。
            reason = ""
            try:
                fn = getattr(self.lib, "aexrt_yolo_get_last_error", None)
                if fn is not None:
                    fn.restype = c_char_p
                    fn.argtypes = [c_void_p]
                    raw = fn(self.model)
                    if raw:
                        reason = raw.decode("utf-8", errors="replace")
            except Exception:
                reason = ""
            raise RuntimeError(
                "native C++ AEXRT YOLO run failed"
                + (f": {reason}" if reason else " (no detailed reason recorded)")
            )
        return [
            Detection(
                int(raw_dets[i].class_id),
                float(raw_dets[i].score),
                (float(raw_dets[i].x1), float(raw_dets[i].y1), float(raw_dets[i].x2), float(raw_dets[i].y2)),
            )
            for i in range(int(count.value))
        ]

    def flush_pipeline(self, max_detections: int = 100) -> list[Detection]:
        """排空 N-buffer 流水线（AEXRT_NATIVE_D3D12_PIPELINE_FRAMES >= 2）。

        等待最后一个 in-flight 帧完成并回收其检测结果；同步模式
        （frames=1）下无 in-flight 帧，返回空列表。
        """
        if not self.executable:
            raise RuntimeError("native C++ AEXRT YOLO package metadata loaded, but this package is not executable by the current pure C++ runtime yet")
        raw_dets = (_YoloDetection * int(max_detections))()
        count = c_uint32(0)
        ok = self.lib.aexrt_yolo_flush_pipeline(
            self.model,
            raw_dets,
            c_uint32(max_detections),
            byref(count),
        )
        if not ok:
            reason = ""
            try:
                fn = getattr(self.lib, "aexrt_yolo_get_last_error", None)
                if fn is not None:
                    fn.restype = c_char_p
                    fn.argtypes = [c_void_p]
                    raw = fn(self.model)
                    if raw:
                        reason = raw.decode("utf-8", errors="replace")
            except Exception:
                reason = ""
            raise RuntimeError(
                "native C++ AEXRT YOLO pipeline flush failed"
                + (f": {reason}" if reason else " (no detailed reason recorded)")
            )
        return [
            Detection(
                int(raw_dets[i].class_id),
                float(raw_dets[i].score),
                (float(raw_dets[i].x1), float(raw_dets[i].y1), float(raw_dets[i].x2), float(raw_dets[i].y2)),
            )
            for i in range(int(count.value))
        ]

    def profile_events(self) -> list[tuple[str, float]]:
        count = int(self.lib.aexrt_yolo_profile_event_count(self.model))
        events: list[tuple[str, float]] = []
        for i in range(count):
            raw = self.lib.aexrt_yolo_profile_event_label(self.model, c_uint32(i))
            label = raw.decode("utf-8", errors="replace") if raw else ""
            ms = float(self.lib.aexrt_yolo_profile_event_ms(self.model, c_uint32(i)))
            events.append((label, ms))
        return events

    @property
    def last_run_timing(self) -> dict[str, float] | None:
        timing = _YoloRunTiming()
        timing.struct_size = sizeof(_YoloRunTiming)
        if not self.lib.aexrt_yolo_get_last_run_timing(self.model, byref(timing)) or not timing.valid:
            return None
        return {
            "memcpy_ms": float(timing.memcpy_ms),
            "submit_ms": float(timing.submit_ms),
            "fence_ms": float(timing.fence_ms),
            "readback_ms": float(timing.readback_ms),
        }

    def export_pipeline_cache(self) -> bytes:
        size = int(self.lib.aexrt_yolo_pipeline_cache_size(self.model))
        if size <= 0:
            raise RuntimeError("native runtime did not produce an AEXRT pipeline cache")
        buffer = (c_ubyte * size)()
        if not self.lib.aexrt_yolo_export_pipeline_cache(self.model, buffer, c_uint64(size)):
            raise RuntimeError("failed to export the AEXRT pipeline cache")
        return bytes(buffer)

    @property
    def dxil_cache_hits(self) -> int:
        return int(self.lib.aexrt_yolo_dxil_cache_hit_count(self.model))

    @property
    def shader_cache_hits(self) -> int:
        return int(self.lib.aexrt_yolo_shader_cache_hit_count(self.model))

    @property
    def pso_cache_hits(self) -> int:
        return int(self.lib.aexrt_yolo_pso_cache_hit_count(self.model))

    def close(self) -> None:
        if getattr(self, "model", None):
            self.lib.aexrt_yolo_destroy(self.model)
            self.model = None
        if getattr(self, "device", None):
            self.lib.aexrt_d3d12_destroy_device(self.device)
            self.device = None
        self.closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _configure(self) -> None:
        self.lib.aexrt_d3d12_create_device.argtypes = [c_uint32]
        self.lib.aexrt_d3d12_create_device.restype = c_void_p
        self.lib.aexrt_d3d12_get_capabilities.argtypes = [c_void_p, POINTER(_D3D12Capabilities)]
        self.lib.aexrt_d3d12_get_capabilities.restype = c_int
        self.lib.aexrt_d3d12_destroy_device.argtypes = [c_void_p]
        self.lib.aexrt_yolo_compile_from_package.argtypes = [c_void_p, c_char_p]
        self.lib.aexrt_yolo_compile_from_package.restype = c_void_p
        self.lib.aexrt_yolo_compile_from_onnx.argtypes = [c_void_p, c_char_p]
        self.lib.aexrt_yolo_compile_from_onnx.restype = c_void_p
        self.lib.aexrt_yolo_load_engine.argtypes = [c_void_p, c_char_p]
        self.lib.aexrt_yolo_load_engine.restype = c_void_p
        self.lib.aexrt_yolo_input_element_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_input_element_count.restype = c_uint64
        self.lib.aexrt_yolo_class_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_class_count.restype = c_uint32
        self.lib.aexrt_yolo_anchor_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_anchor_count.restype = c_uint32
        self.lib.aexrt_yolo_channel_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_channel_count.restype = c_uint32
        self.lib.aexrt_yolo_output_layout.argtypes = [c_void_p]
        self.lib.aexrt_yolo_output_layout.restype = c_uint32
        self.lib.aexrt_yolo_has_objectness.argtypes = [c_void_p]
        self.lib.aexrt_yolo_has_objectness.restype = c_uint32
        self.lib.aexrt_yolo_package_mode.argtypes = [c_void_p]
        self.lib.aexrt_yolo_package_mode.restype = c_uint32
        self.lib.aexrt_yolo_graph_node_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_graph_node_count.restype = c_uint32
        self.lib.aexrt_yolo_graph_value_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_graph_value_count.restype = c_uint32
        self.lib.aexrt_yolo_constant_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_constant_count.restype = c_uint32
        self.lib.aexrt_yolo_prepared_command_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_prepared_command_count.restype = c_uint32
        self.lib.aexrt_yolo_supported_prepared_command_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_supported_prepared_command_count.restype = c_uint32
        self.lib.aexrt_yolo_unsupported_prepared_command_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_unsupported_prepared_command_count.restype = c_uint32
        self.lib.aexrt_yolo_prepared_skipped_command_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_prepared_skipped_command_count.restype = c_uint32
        self.lib.aexrt_yolo_late_concat_conv1x1_fusion_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_late_concat_conv1x1_fusion_count.restype = c_uint32
        self.lib.aexrt_yolo_concat_residual_conv1x1_fusion_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_concat_residual_conv1x1_fusion_count.restype = c_uint32
        self.lib.aexrt_yolo_paired_conv3x3_fusion_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_paired_conv3x3_fusion_count.restype = c_uint32
        self.lib.aexrt_yolo_c2f_bottleneck_superblock_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_c2f_bottleneck_superblock_count.restype = c_uint32
        self.lib.aexrt_yolo_c2f_tail_residual_fusion_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_c2f_tail_residual_fusion_count.restype = c_uint32
        self.lib.aexrt_yolo_head_fusion_enabled.argtypes = [c_void_p]
        self.lib.aexrt_yolo_head_fusion_enabled.restype = c_uint32
        self.lib.aexrt_yolo_head_final_conv_fusion_enabled.argtypes = [c_void_p]
        self.lib.aexrt_yolo_head_final_conv_fusion_enabled.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_conv1x1_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_conv1x1_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_spatial_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_spatial_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_pack4_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_pack4_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_pack8_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_pack8_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_40x40_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_40x40_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_20x20_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_20x20_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_10x10_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_implicit_gemm_10x10_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_exact_40x40_64x64_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_exact_40x40_64x64_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_exact_20x20_64x64_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_exact_20x20_64x64_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_exact_10x10_128x128_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_exact_10x10_128x128_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_exact_20x20_128x64_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_exact_20x20_128x64_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_3x3_exact_10x10_256x64_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_3x3_exact_10x10_256x64_count.restype = c_uint32
        self.lib.aexrt_yolo_tileflow_native_fp16_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_tileflow_native_fp16_count.restype = c_uint32
        self.lib.aexrt_yolo_is_executable.argtypes = [c_void_p]
        self.lib.aexrt_yolo_is_executable.restype = c_int
        self.lib.aexrt_yolo_run.argtypes = [
            c_void_p,
            c_void_p,
            POINTER(c_float),
            c_uint64,
            POINTER(_YoloDetection),
            c_uint32,
            POINTER(c_uint32),
        ]
        self.lib.aexrt_yolo_run.restype = c_int
        self.lib.aexrt_yolo_flush_pipeline.argtypes = [
            c_void_p,
            POINTER(_YoloDetection),
            c_uint32,
            POINTER(c_uint32),
        ]
        self.lib.aexrt_yolo_flush_pipeline.restype = c_int
        self.lib.aexrt_yolo_get_last_run_timing.argtypes = [c_void_p, POINTER(_YoloRunTiming)]
        self.lib.aexrt_yolo_get_last_run_timing.restype = c_int
        self.lib.aexrt_yolo_profile_event_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_profile_event_count.restype = c_uint32
        self.lib.aexrt_yolo_profile_event_label.argtypes = [c_void_p, c_uint32]
        self.lib.aexrt_yolo_profile_event_label.restype = c_char_p
        self.lib.aexrt_yolo_profile_event_ms.argtypes = [c_void_p, c_uint32]
        self.lib.aexrt_yolo_profile_event_ms.restype = c_double
        self.lib.aexrt_yolo_pipeline_cache_size.argtypes = [c_void_p]
        self.lib.aexrt_yolo_pipeline_cache_size.restype = c_uint64
        self.lib.aexrt_yolo_export_pipeline_cache.argtypes = [c_void_p, c_void_p, c_uint64]
        self.lib.aexrt_yolo_export_pipeline_cache.restype = c_int
        self.lib.aexrt_yolo_dxil_cache_hit_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_dxil_cache_hit_count.restype = c_uint32
        self.lib.aexrt_yolo_shader_cache_hit_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_shader_cache_hit_count.restype = c_uint32
        self.lib.aexrt_yolo_pso_cache_hit_count.argtypes = [c_void_p]
        self.lib.aexrt_yolo_pso_cache_hit_count.restype = c_uint32
        self.lib.aexrt_yolo_destroy.argtypes = [c_void_p]


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(POINTER(c_float))


def _load_native_cpp():
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "build" / "native" / "aexrt_native_cpp.dll",
        root / "examples" / "cpp" / "bin" / "aexrt_native_cpp.dll",
    ]
    for path in candidates:
        if path.exists():
            return cdll.LoadLibrary(str(path))
    raise RuntimeError("aexrt_native_cpp.dll not found; run examples\\cpp\\build_pure_native_relu.ps1")
