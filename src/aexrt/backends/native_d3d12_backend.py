from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np

from .base import Backend, BackendInfo
from .. import _numpy_ops
from ..abi import export_graph_abi
from ..cpp_runtime import NativeCppGraph
from ..device import DeviceBuffer, NativeD3D12Device
from ..execution import infer_value_specs
from ..graph import Graph
from ..scheduler import GraphSchedule, compile_graph_schedule


def _cpu_fallback_enabled() -> bool:
    """native 无 kernel 的算子回退到 CPU 逐算子执行（默认开启，可用
    AEXRT_NATIVE_D3D12_CPU_FALLBACK=0 关闭恢复原来的 NotImplementedError）。"""
    value = os.environ.get("AEXRT_NATIVE_D3D12_CPU_FALLBACK", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


class NativeD3D12Backend(Backend):
    """AEXRT-owned D3D12 backend entry point, not a DirectML wrapper."""

    name = "native_d3d12"
    supported_ops = frozenset({
        "Add", "BatchNormalization", "Concat", "Conv", "Div", "Gelu", "Identity", "MaxPool", "Mul",
        "Relu", "Reshape", "Resize", "Sigmoid", "Slice", "Softmax", "Split", "Sub", "Tanh", "Transpose",
    }) | _numpy_ops.NUMPY_EVAL_OPS  # 无 native kernel 的算子运行时走 CPU 逐算子回退
    supported_dtypes = frozenset({"float32", "int64"})
    supported_features = frozenset({
        "aexrt_native_device",
        "device_buffer",
        "host_device_upload",
        "device_host_download",
        "persistent_constants",
        "static_execution_plan",
        "lifetime_colored_arena",
        "shape_specialized_kernel_abi",
        "tile_wave_scheduler",
        "fused_conv2d_silu",
        "prepared_conv2d_silu_replay",
        "persistent_mapped_upload_ring",
        "gpu_yolo_decode_nms_topk",
        "session_run_yolo_gpu",
        "native_yolo_dag_runner",
        "native_layout_primitives",
        "experimental_batch_command_recording",
        "prepared_yolo_graph_replay",
        "prepared_yolo_gpu_postprocess",
        "prepared_yolo_graph_decode_nms_replay",
        "yolo_detect_head_dfl_decode_nms_superblock",
        "native_dfl_project",
        "conv1x1_fast_path",
        "conv3x3_tiled_shared_memory_silu",
        "experimental_conv3x3_winograd_f2x2_silu",
        "experimental_conv3x3_winograd_packed_weight_silu",
        "experimental_conv3x3_winograd_oc4_tile_silu",
        "persistent_winograd_packed_weight_buffer",
        "experimental_fp16_packed_concat_conv1x1_superblock",
        "experimental_int8_activation_weight_concat_conv1x1_superblock",
        "experimental_sppf_tail_superblock_maxpool_concat_conv1x1",
        "zero_copy_channel_split_views",
        "fused_channel_concat_conv1x1",
        "c2f_superblock_compiler",
        "c2f_superblock_virtual_residual_concat",
        "c2f_superblock_tiled_tail_kernel",
        "c2f_superblock_tiled_bottleneck_kernel",
        "c2f_superblock_persistent_weight_tile",
    })

    def __init__(self, adapter_index: int = 0, output_numpy: bool = True) -> None:
        self.adapter_index = int(adapter_index)
        self.output_numpy = output_numpy
        self.device = NativeD3D12Device(adapter_index)
        self.graph: Graph | None = None
        self.execution_plan = None
        self.schedule: GraphSchedule | None = None
        self.constants: Dict[str, DeviceBuffer] = {}
        self._persistent_outputs: Dict[str, DeviceBuffer] = {}
        self._compiled_cpp: NativeCppGraph | None = None
        self._conv_silu_plan: Dict[str, Any] | None = None
        self._yolo_nms_plan: Dict[str, Any] | None = None
        self._generic_specs: Dict[str, Any] = {}
        self._generic_buffers: Dict[str, DeviceBuffer] = {}
        self._zero_bias_buffers: Dict[int, DeviceBuffer] = {}
        self._winograd_packed_constants: Dict[str, DeviceBuffer] = {}
        self._fp16_superblock_constants: Dict[str, DeviceBuffer] = {}
        self._int8_superblock_constants: Dict[str, tuple[DeviceBuffer, DeviceBuffer, float]] = {}
        self._batch_recording = os.environ.get("AEXRT_NATIVE_D3D12_BATCH", "0") == "1"
        self._c2f_superblock_enabled = os.environ.get("AEXRT_NATIVE_D3D12_C2F_SUPERBLOCK", "1") != "0"
        self._c2f_front_enabled = os.environ.get("AEXRT_NATIVE_D3D12_C2F_FRONT", "0") == "1"
        self._prepared_generic_plan: Dict[str, Any] | None = None
        self._prepared_generic_error: str | None = None
        self._prepared_yolo_nms_plan: Dict[str, Any] | None = None
        self._prepared_yolo_graph_plan: Dict[str, Any] | None = None
        self._fused_concat_conv1x1_count = 0
        self._fused_c2f_residual_tail_count = 0
        self._fused_c2f_bottleneck_count = 0
        self._compiled_c2f_superblock_count = 0
        self._fused_sppf_tail_count = 0
        self._fused_yolo_detect_head_count = 0
        self._virtual_residual_adds: Dict[str, tuple[str, str]] = {}
        self._c2f_virtual_residual_candidates: Dict[str, tuple[str, str]] = {}
        self._c2f_superblocks: list[Dict[str, Any]] = []

    @staticmethod
    def probe(adapter_index: int = 0):
        return NativeD3D12Device.probe(adapter_index)

    def info(self) -> BackendInfo:
        device_info = self.device.info()
        return BackendInfo("native_d3d12", device_info.device_type, {
            "gpu": True,
            "api": device_info.api,
            "device_type": device_info.device_type,
            "device_name": device_info.name,
            "vendor": device_info.vendor,
            "device_index": device_info.index,
            "ops": sorted(self.supported_ops),
            "dtypes": sorted(self.supported_dtypes),
            "features": sorted(self.supported_features),
            "directml": False,
        })

    def prepare(self, graph: Graph) -> None:
        self.graph = graph
        self.execution_plan = self.compile_plan(graph)
        self.schedule = compile_graph_schedule(graph, self.execution_plan)
        self.constants = {name: self.device.upload(value, label=name) for name, value in graph.constants.items()}
        self._generic_specs = infer_value_specs(graph)
        self._generic_buffers = {}
        self._zero_bias_buffers = {}
        self._winograd_packed_constants = {}
        self._fp16_superblock_constants = {}
        self._int8_superblock_constants = {}
        self._prepack_winograd_constants_for_graph()
        self._prepack_fp16_superblock_constants_for_graph()
        self._prepack_int8_superblock_constants_for_graph()
        self._prepared_generic_plan = None
        self._prepared_generic_error = None
        self._prepared_yolo_nms_plan = None
        self._prepared_yolo_graph_plan = None
        self._fused_concat_conv1x1_count = 0
        self._fused_c2f_residual_tail_count = 0
        self._fused_c2f_bottleneck_count = 0
        self._compiled_c2f_superblock_count = 0
        self._fused_sppf_tail_count = 0
        self._fused_yolo_detect_head_count = 0
        self._virtual_residual_adds = {}
        self._c2f_virtual_residual_candidates = {}
        self._c2f_superblocks = self._compile_c2f_superblocks()
        self._compiled_c2f_superblock_count = len(self._c2f_superblocks)
        self._conv_silu_plan = self._prepare_conv_silu_chain(graph)
        if self._conv_silu_plan is not None:
            self._compiled_cpp = None
            return
        self._prepared_generic_plan = self._try_prepare_generic_graph()
        if self._prepared_generic_plan is not None:
            self._compiled_cpp = None
            return
        try:
            export_graph_abi(graph)
            self._compiled_cpp = NativeCppGraph(graph, adapter_index=self.adapter_index)
        except Exception:
            self._compiled_cpp = None

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._conv_silu_plan is not None or self._can_run_single_relu(inputs):
            outputs = self._run_graph_to_device_outputs(inputs)
            if self.output_numpy:
                return {name: self.device.download(buffer) for name, buffer in outputs.items()}
            return outputs
        if self._prepared_generic_plan is not None:
            outputs = self._run_prepared_generic_graph_to_device_outputs(inputs)
            if self.output_numpy:
                return {name: self.device.download(buffer) for name, buffer in outputs.items()}
            return outputs
        if self._can_run_generic_graph():
            outputs = self._run_generic_graph_to_device_outputs(inputs)
            if self.output_numpy:
                return {name: self.device.download(buffer) for name, buffer in outputs.items()}
            return outputs
        if self._compiled_cpp is not None:
            outputs = self._compiled_cpp.run(inputs)
            if self.output_numpy:
                return outputs
            return {k: self.device.upload(v, label=k) for k, v in outputs.items()}
        raise NotImplementedError(
            "native_d3d12 has Device/Buffer/upload/download/capability wired, "
            "and currently implements compiled elementwise DAGs plus fused Conv2D+SiLU; this graph is not supported yet"
        )

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
    ) -> np.ndarray:
        if output_name is None:
            if self.graph is None or len(self.graph.outputs) != 1:
                raise ValueError("output_name is required when the graph has multiple outputs")
            output_name = self.graph.outputs[0]

        if self.graph is None:
            raise RuntimeError("native backend is not prepared")
        out_spec = self._generic_specs.get(output_name)
        if out_spec is None:
            outputs = self._run_graph_to_device_outputs(inputs)
            if output_name not in outputs:
                raise ValueError(f"output {output_name!r} was not produced by the native graph")
            yolo_output = outputs[output_name]
            anchors_i, channels_i, classes_i = _infer_yolo_head_layout(
                yolo_output.shape,
                anchors=anchors,
                channels=channels,
                classes=classes,
            )
        else:
            anchors_i, channels_i, classes_i = _infer_yolo_head_layout(
                out_spec.shape,
                anchors=anchors,
                channels=channels,
                classes=classes,
            )
        max_candidates_i = int(max_candidates)
        max_detections_i = int(max_detections)
        if max_candidates_i <= 0 or max_detections_i <= 0:
            raise ValueError("max_candidates and max_detections must be positive")

        prepared_yolo = self._get_or_create_prepared_yolo_graph(
            output_name=output_name,
            anchors=anchors_i,
            channels=channels_i,
            classes=classes_i,
            max_candidates=max_candidates_i,
            max_detections=max_detections_i,
            conf_threshold=float(conf_threshold),
            iou_threshold=float(iou_threshold),
        )
        if prepared_yolo is not None:
            input_buffers: Dict[str, DeviceBuffer] = prepared_yolo["inputs"]
            for name, spec in self.graph.inputs.items():
                if name not in inputs:
                    raise ValueError(f"missing input: {name}")
                arr = np.ascontiguousarray(inputs[name], dtype=np.float32)
                if tuple(arr.shape) != tuple(spec.shape):
                    raise ValueError(f"expected input {name} shape {spec.shape}, got {tuple(arr.shape)}")
                self.device.upload_into(input_buffers[name], arr)
            self.device.execute_prepared_graph(prepared_yolo["handle"])
            detections = prepared_yolo["detections"]
            counter = prepared_yolo["counter"]
            return self.device.download_yolo_topk(detections, counter)

        outputs = self._run_graph_to_device_outputs(inputs)
        if output_name not in outputs:
            raise ValueError(f"output {output_name!r} was not produced by the native graph")
        yolo_output = outputs[output_name]
        candidates, candidate_counter, keep, detections, counter = self._get_or_create_yolo_nms_plan(
            anchors=anchors_i,
            channels=channels_i,
            classes=classes_i,
            max_candidates=max_candidates_i,
            max_detections=max_detections_i,
        )
        prepared_post = self._get_or_create_prepared_yolo_nms(
            yolo_output,
            candidates,
            candidate_counter,
            keep,
            detections,
            counter,
            anchors=anchors_i,
            channels=channels_i,
            classes=classes_i,
            max_candidates=max_candidates_i,
            max_detections=max_detections_i,
            conf_threshold=float(conf_threshold),
            iou_threshold=float(iou_threshold),
        )
        if prepared_post is not None:
            self.device.execute_prepared_graph(prepared_post)
        else:
            self.device.dispatch_yolo_decode_nms_float32(
                yolo_output,
                candidates,
                candidate_counter,
                keep,
                detections,
                counter,
                anchors=anchors_i,
                channels=channels_i,
                classes=classes_i,
                max_candidates=max_candidates_i,
                max_detections=max_detections_i,
                conf_threshold=float(conf_threshold),
                iou_threshold=float(iou_threshold),
            )
        return self.device.download_yolo_topk(detections, counter)
    def _run_graph_to_device_outputs(self, inputs: Dict[str, Any]) -> Dict[str, DeviceBuffer]:
        if self._conv_silu_plan is not None:
            y_name, y_buffer = self._run_conv_silu_to_buffer(inputs)
            return {y_name: y_buffer}
        if self._can_run_single_relu(inputs):
            assert self.graph is not None
            node = self.graph.nodes[0]
            x_name = node.inputs[0]
            y_name = node.outputs[0]
            x = self.device.upload(np.ascontiguousarray(inputs[x_name], dtype=np.float32), label=x_name)
            element_count = _numel(x.shape)
            y = self._get_or_create_output(y_name, x)
            self.device.dispatch_relu_float32_into(x, y, element_count)
            return {y_name: y}
        if self._prepared_generic_plan is not None:
            return self._run_prepared_generic_graph_to_device_outputs(inputs)
        if self._can_run_generic_graph():
            return self._run_generic_graph_to_device_outputs(inputs)
        if self._compiled_cpp is not None:
            return {name: self.device.upload(value, label=name) for name, value in self._compiled_cpp.run(inputs).items()}
        raise NotImplementedError(
            "native_d3d12 cannot produce a GPU output buffer for this graph yet; "
            "run_yolo_gpu currently needs a native Conv/SiLU chain, native elementwise graph, or single ReLU graph"
        )

    def _prepare_conv_silu_chain(self, graph: Graph) -> Dict[str, Any] | None:
        match = _match_conv_silu_chain(graph)
        if match is None:
            return None
        input_name, output_name, blocks = match
        owned_buffers = []
        weight_buffers = []
        bias_buffers = []
        descs = []
        for block in blocks:
            weight_value = _pack_winograd_f2x2_3x3_weights(block["weight"]) if _winograd_packed_enabled() and _is_winograd_packable_desc(block["desc"]) else block["weight"]
            weight_buffer = self.device.upload(weight_value, label=f"{block['output_name']}_folded_w")
            bias_buffer = self.device.upload(block["bias"], label=f"{block['output_name']}_folded_b")
            weight_buffers.append(weight_buffer)
            bias_buffers.append(bias_buffer)
            descs.append(block["desc"])
            owned_buffers.extend([weight_buffer, bias_buffer])
        output_shape = blocks[-1]["output_shape"]
        output_buffer = self.device.allocate_uav(
            int(np.prod(output_shape)) * 4,
            dtype="float32",
            shape=output_shape,
            label=output_name,
        )
        owned_buffers.append(output_buffer)
        prepared = self.device.prepare_conv2d_silu_chain_upload_float32_dispatch(
            weight_buffers,
            bias_buffers,
            output_buffer,
            descs,
            ring_size=2,
        )
        return {
            "input_name": input_name,
            "output_name": output_name,
            "input_shape": blocks[0]["input_shape"],
            "output_shape": blocks[-1]["output_shape"],
            "prepared": prepared,
            "output_buffer": output_buffer,
            "owned_buffers": owned_buffers,
        }

    def _run_conv_silu(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        y_name, y_buffer = self._run_conv_silu_to_buffer(inputs)
        if self.output_numpy:
            return {y_name: self.device.download(y_buffer)}
        return {y_name: y_buffer}

    def _run_conv_silu_to_buffer(self, inputs: Dict[str, Any]) -> tuple[str, DeviceBuffer]:
        assert self._conv_silu_plan is not None
        x_name = self._conv_silu_plan["input_name"]
        x = np.ascontiguousarray(inputs[x_name], dtype=np.float32)
        if tuple(x.shape) != tuple(self._conv_silu_plan["input_shape"]):
            raise ValueError(f"expected input {x_name} shape {self._conv_silu_plan['input_shape']}, got {tuple(x.shape)}")
        self.device.execute_conv2d_silu_chain_upload_float32_dispatch(self._conv_silu_plan["prepared"], x)
        y_name = self._conv_silu_plan["output_name"]
        y_buffer = self._conv_silu_plan["output_buffer"]
        return y_name, y_buffer

    def _can_run_single_relu(self, inputs: Dict[str, Any]) -> bool:
        if self.graph is None or len(self.graph.nodes) != 1:
            return False
        node = self.graph.nodes[0]
        if node.op != "Relu" or len(node.inputs) != 1 or len(node.outputs) != 1:
            return False
        if node.outputs[0] not in self.graph.outputs:
            return False
        x = np.asarray(inputs.get(node.inputs[0]))
        return x.dtype == np.float32

    def upload(self, value: Any, *, label: str | None = None) -> DeviceBuffer:
        return self.device.upload(np.ascontiguousarray(value), label=label)

    def download(self, buffer: DeviceBuffer) -> np.ndarray:
        return self.device.download(buffer)

    def _get_or_create_output(self, name: str, like: DeviceBuffer) -> DeviceBuffer:
        existing = self._persistent_outputs.get(name)
        if existing is not None and existing.nbytes == like.nbytes and existing.dtype == like.dtype and existing.shape == like.shape:
            return existing
        output = self.device.allocate_uav(
            like.nbytes,
            dtype=like.dtype,
            shape=like.shape,
            label=name,
        )
        self._persistent_outputs[name] = output
        return output

    def _get_or_create_yolo_nms_plan(
        self,
        *,
        anchors: int,
        channels: int,
        classes: int,
        max_candidates: int,
        max_detections: int,
    ) -> tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer]:
        key = (int(anchors), int(channels), int(classes), int(max_candidates), int(max_detections))
        if self._yolo_nms_plan is not None and self._yolo_nms_plan["key"] == key:
            return self._yolo_nms_plan["buffers"]
        buffers = self.device.allocate_yolo_nms_buffers(
            int(max_candidates),
            int(max_detections),
            label=f"session_yolo_{int(anchors)}x{int(channels)}",
        )
        self._yolo_nms_plan = {"key": key, "buffers": buffers}
        return buffers

    def _get_or_create_prepared_yolo_nms(
        self,
        yolo_output: DeviceBuffer,
        candidates: DeviceBuffer,
        candidate_counter: DeviceBuffer,
        keep: DeviceBuffer,
        detections: DeviceBuffer,
        counter: DeviceBuffer,
        *,
        anchors: int,
        channels: int,
        classes: int,
        max_candidates: int,
        max_detections: int,
        conf_threshold: float,
        iou_threshold: float,
    ):
        key = (
            id(yolo_output.handle),
            id(candidates.handle),
            id(candidate_counter.handle),
            id(keep.handle),
            id(detections.handle),
            id(counter.handle),
            int(anchors),
            int(channels),
            int(classes),
            int(max_candidates),
            int(max_detections),
            float(conf_threshold),
            float(iou_threshold),
        )
        if self._prepared_yolo_nms_plan is not None and self._prepared_yolo_nms_plan["key"] == key:
            return self._prepared_yolo_nms_plan["handle"]
        if os.environ.get("AEXRT_NATIVE_D3D12_PREPARED_POST", "1") == "0":
            return None
        try:
            self.device.begin_prepared_graph()
            self.device.dispatch_yolo_decode_nms_float32(
                yolo_output,
                candidates,
                candidate_counter,
                keep,
                detections,
                counter,
                anchors=anchors,
                channels=channels,
                classes=classes,
                max_candidates=max_candidates,
                max_detections=max_detections,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            handle = self.device.end_prepared_graph()
            self._prepared_yolo_nms_plan = {"key": key, "handle": handle}
            return handle
        except Exception:
            try:
                self.device.end_prepared_graph()
            except Exception:
                pass
            self._prepared_yolo_nms_plan = None
            return None

    def _get_or_create_prepared_yolo_graph(
        self,
        *,
        output_name: str,
        anchors: int,
        channels: int,
        classes: int,
        max_candidates: int,
        max_detections: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> Dict[str, Any] | None:
        if not self._can_run_generic_graph() or self.graph is None:
            return None
        if os.environ.get("AEXRT_NATIVE_D3D12_PREPARED_YOLO_GRAPH", "1") == "0":
            return None
        key = (
            output_name,
            int(anchors),
            int(channels),
            int(classes),
            int(max_candidates),
            int(max_detections),
            float(conf_threshold),
            float(iou_threshold),
            tuple((name, tuple(spec.shape)) for name, spec in self.graph.inputs.items()),
        )
        if self._prepared_yolo_graph_plan is not None and self._prepared_yolo_graph_plan["key"] == key:
            return self._prepared_yolo_graph_plan
        try:
            values: Dict[str, DeviceBuffer] = dict(self.constants)
            input_buffers: Dict[str, DeviceBuffer] = {}
            for name, spec in self.graph.inputs.items():
                nbytes = int(np.prod(spec.shape)) * 4
                buffer = self.device.allocate_uav(nbytes, dtype="float32", shape=spec.shape, label=f"{name}_prepared_yolo_input")
                input_buffers[name] = buffer
                values[name] = buffer

            candidates, candidate_counter, keep, detections, counter = self._get_or_create_yolo_nms_plan(
                anchors=anchors,
                channels=channels,
                classes=classes,
                max_candidates=max_candidates,
                max_detections=max_detections,
            )

            detect_head = self._match_yolo_detect_head_superblock(output_name, classes=classes)
            detect_skip = set(detect_head["skip_indices"]) if detect_head is not None else set()
            self.device.begin_prepared_graph()
            i = 0
            while i < len(self.graph.nodes):
                if detect_head is not None and i in detect_skip and i != detect_head["start"]:
                    i += 1
                    continue
                if detect_head is not None and i == detect_head["start"]:
                    box_outputs = [values[name] for name in detect_head["box_outputs"]]
                    class_outputs = [values[name] for name in detect_head["class_outputs"]]
                    self.device.dispatch_yolo_head_decode_nms_float32(
                        box_outputs,
                        class_outputs,
                        candidates,
                        candidate_counter,
                        keep,
                        detections,
                        counter,
                        classes=classes,
                        max_candidates=max_candidates,
                        max_detections=max_detections,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        strides=detect_head["strides"],
                    )
                    self._fused_yolo_detect_head_count += 1
                    i = detect_head["end"] + 1
                    continue
                i = self._emit_generic_node(values, i)
            if detect_head is None and output_name not in values:
                raise ValueError(f"output {output_name!r} was not produced by the native graph")
            if detect_head is None:
                yolo_output = values[output_name]
                self.device.dispatch_yolo_decode_nms_float32(
                    yolo_output,
                    candidates,
                    candidate_counter,
                    keep,
                    detections,
                    counter,
                    anchors=anchors,
                    channels=channels,
                    classes=classes,
                    max_candidates=max_candidates,
                    max_detections=max_detections,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )
            handle = self.device.end_prepared_graph()
            self._prepared_yolo_graph_plan = {
                "key": key,
                "handle": handle,
                "inputs": input_buffers,
                "detections": detections,
                "counter": counter,
            }
            return self._prepared_yolo_graph_plan
        except Exception:
            try:
                self.device.end_prepared_graph()
            except Exception:
                pass
            self._prepared_yolo_graph_plan = None
            return None

    def _match_yolo_detect_head_superblock(self, output_name: str, *, classes: int) -> Dict[str, Any] | None:
        if os.environ.get("AEXRT_NATIVE_D3D12_YOLO_HEAD_SUPERBLOCK", "1") == "0":
            return None
        if self.graph is None:
            return None
        nodes = self.graph.nodes
        producers = {
            node.outputs[0]: index
            for index, node in enumerate(nodes)
            if len(node.outputs) == 1
        }
        for index, concat in enumerate(nodes):
            if concat.op != "Concat" or concat.attrs.get("axis") != 2 or len(concat.inputs) != 3:
                continue
            if index + 17 >= len(nodes):
                continue
            split, reshape_dfl, sigmoid_cls, transpose, softmax, dfl_conv, reshape_box = nodes[index + 1:index + 8]
            sl0, sl1, sub0, add1, add2, sub1, div1, concat_box, mul_stride, final_concat = nodes[index + 8:index + 18]
            if (
                split.op != "Split"
                or reshape_dfl.op != "Reshape"
                or sigmoid_cls.op != "Sigmoid"
                or transpose.op != "Transpose"
                or softmax.op != "Softmax"
                or dfl_conv.op != "Conv"
                or reshape_box.op != "Reshape"
                or sl0.op != "Slice"
                or sl1.op != "Slice"
                or sub0.op != "Sub"
                or add1.op != "Add"
                or add2.op != "Add"
                or sub1.op != "Sub"
                or div1.op != "Div"
                or concat_box.op != "Concat"
                or mul_stride.op != "Mul"
                or final_concat.op != "Concat"
                or final_concat.outputs != [output_name]
            ):
                continue
            box_outputs = []
            class_outputs = []
            skip_indices = set(range(index, index + 18))
            ok = True
            for reshape_name in concat.inputs:
                reshape_index = producers.get(reshape_name)
                if reshape_index is None:
                    ok = False
                    break
                reshape = nodes[reshape_index]
                if reshape.op != "Reshape" or len(reshape.inputs) != 1:
                    ok = False
                    break
                raw_concat_index = producers.get(reshape.inputs[0])
                if raw_concat_index is None:
                    ok = False
                    break
                raw_concat = nodes[raw_concat_index]
                if raw_concat.op != "Concat" or raw_concat.attrs.get("axis") != 1 or len(raw_concat.inputs) != 2:
                    ok = False
                    break
                box_name, class_name = raw_concat.inputs
                box_spec = self._generic_specs.get(box_name)
                class_spec = self._generic_specs.get(class_name)
                if box_spec is None or class_spec is None or len(box_spec.shape) != 4 or len(class_spec.shape) != 4:
                    ok = False
                    break
                if int(box_spec.shape[1]) != 64 or int(class_spec.shape[1]) != int(classes) or tuple(box_spec.shape[2:]) != tuple(class_spec.shape[2:]):
                    ok = False
                    break
                box_outputs.append(box_name)
                class_outputs.append(class_name)
                skip_indices.add(reshape_index)
                skip_indices.add(raw_concat_index)
            if not ok:
                continue
            shapes = [tuple(int(v) for v in self._generic_specs[name].shape[2:]) for name in box_outputs]
            areas = [h * w for h, w in shapes]
            if areas != sorted(areas, reverse=True):
                continue
            return {
                "start": index,
                "end": index + 17,
                "skip_indices": skip_indices,
                "box_outputs": box_outputs,
                "class_outputs": class_outputs,
                "strides": (8.0, 16.0, 32.0),
            }
        return None

    def _can_run_generic_graph(self) -> bool:
        if self.graph is None:
            return False
        return all(node.op in self.supported_ops for node in self.graph.nodes)

    def _try_prepare_generic_graph(self) -> Dict[str, Any] | None:
        if not self._can_run_generic_graph() or self.graph is None:
            return None
        if os.environ.get("AEXRT_NATIVE_D3D12_PREPARED", "1") == "0":
            return None
        try:
            values: Dict[str, DeviceBuffer] = dict(self.constants)
            input_buffers: Dict[str, DeviceBuffer] = {}
            for name, spec in self.graph.inputs.items():
                nbytes = int(np.prod(spec.shape)) * 4
                buffer = self.device.allocate_uav(nbytes, dtype="float32", shape=spec.shape, label=f"{name}_prepared_input")
                input_buffers[name] = buffer
                values[name] = buffer

            steps = []
            self.device.begin_prepared_graph()
            recording = True
            i = 0
            while i < len(self.graph.nodes):
                node = self.graph.nodes[i]
                if node.op == "Conv" and self._needs_batch_boundary_for_conv(node):
                    if recording:
                        steps.append(("prepared", self.device.end_prepared_graph()))
                        recording = False
                    self._reserve_generic_conv_output(node, values)
                    steps.append(("conv", node))
                    self.device.begin_prepared_graph()
                    recording = True
                    i += 1
                    continue
                i = self._emit_generic_node(values, i)
            if recording:
                steps.append(("prepared", self.device.end_prepared_graph()))
            return {
                "inputs": input_buffers,
                "outputs": {name: values[name] for name in self.graph.outputs},
                "steps": steps,
                "values": values,
            }
        except Exception:
            import traceback
            self._prepared_generic_error = traceback.format_exc()
            try:
                self.device.end_prepared_graph()
            except Exception:
                pass
            return None

    def _run_prepared_generic_graph_to_device_outputs(self, inputs: Dict[str, Any]) -> Dict[str, DeviceBuffer]:
        assert self.graph is not None and self._prepared_generic_plan is not None
        input_buffers: Dict[str, DeviceBuffer] = self._prepared_generic_plan["inputs"]
        for name, spec in self.graph.inputs.items():
            if name not in inputs:
                raise ValueError(f"missing input: {name}")
            arr = np.ascontiguousarray(inputs[name], dtype=np.float32)
            if tuple(arr.shape) != tuple(spec.shape):
                raise ValueError(f"expected input {name} shape {spec.shape}, got {tuple(arr.shape)}")
            self.device.upload_into(input_buffers[name], arr)
        values = self._prepared_generic_plan["values"]
        for kind, payload in self._prepared_generic_plan["steps"]:
            if kind == "prepared":
                self.device.execute_prepared_graph(payload)
            elif kind == "conv":
                self._run_generic_conv(payload, values, silu=False)
            else:
                raise RuntimeError(f"unknown prepared graph step: {kind}")
        return dict(self._prepared_generic_plan["outputs"])

    def _emit_generic_node(self, values: Dict[str, DeviceBuffer], index: int) -> int:
        assert self.graph is not None
        node = self.graph.nodes[index]
        if node.op == "Conv":
            if self._is_dfl_projection_conv(node):
                self._run_dfl_projection(node, values)
                return index + 1
            c2f = self._try_run_c2f_bottleneck_superblock(index, values)
            if c2f is not None:
                return c2f
            silu = self._try_run_generic_conv_silu(node, index, values)
            if silu is not None:
                return silu
            self._run_generic_conv(node, values, silu=False)
        elif node.op in {"Gelu", "Relu", "Sigmoid", "Tanh"}:
            x = values[node.inputs[0]]
            y = self._get_or_create_graph_value(node.outputs[0])
            self.device.dispatch_unary_float32_into(node.op, x, y)
            values[node.outputs[0]] = y
        elif node.op in {"Add", "Sub", "Mul", "Div"}:
            if node.op == "Add" and self._try_defer_residual_add(index):
                return index + 1
            a = values[node.inputs[0]]
            b = values[node.inputs[1]]
            y = self._get_or_create_graph_value(node.outputs[0])
            self.device.dispatch_binary_broadcast_float32_into(node.op, a, b, y)
            values[node.outputs[0]] = y
        elif node.op == "Identity":
            values[node.outputs[0]] = values[node.inputs[0]]
        elif node.op == "Reshape":
            src = values[node.inputs[0]]
            spec = self._generic_specs[node.outputs[0]]
            values[node.outputs[0]] = DeviceBuffer(src.device, src.nbytes, dtype=src.dtype, shape=spec.shape, label=node.outputs[0], handle=src.handle)
        elif node.op == "Split":
            if not self._try_emit_channel_split_views(node, values):
                src = values[node.inputs[0]]
                axis = _normalize_axis(int(node.attrs.get("axis", 0)), len(src.shape or ()))
                start = 0
                for out_name in node.outputs:
                    out = self._get_or_create_graph_value(out_name)
                    self.device.dispatch_slice_float32_into(src, out, axis=axis, start=start)
                    values[out_name] = out
                    start += int(out.shape[axis])
        elif node.op == "Slice":
            src = values[node.inputs[0]]
            axes = node.attrs.get("axes") or [0]
            starts = node.attrs.get("starts") or [0]
            steps = node.attrs.get("steps") or [1]
            if len(axes) != 1 or len(starts) != 1 or int(steps[0]) != 1:
                raise NotImplementedError("native Slice currently supports one axis with step=1")
            axis = _normalize_axis(int(axes[0]), len(src.shape or ()))
            out = self._get_or_create_graph_value(node.outputs[0])
            self.device.dispatch_slice_float32_into(src, out, axis=axis, start=int(starts[0]))
            values[node.outputs[0]] = out
        elif node.op == "Concat":
            fused = self._try_run_concat_conv1x1(index, values)
            if fused is not None:
                return fused
            axis = _normalize_axis(int(node.attrs.get("axis", 0)), len(self._generic_specs[node.outputs[0]].shape))
            out = self._get_or_create_graph_value(node.outputs[0])
            self.device.dispatch_concat_float32_into([values[name] for name in node.inputs], out, axis=axis)
            values[node.outputs[0]] = out
        elif node.op == "Resize":
            mode = str(node.attrs.get("mode", "nearest"))
            coord = str(node.attrs.get("coordinate_transformation_mode", "asymmetric"))
            if mode != "nearest" or coord != "asymmetric":
                raise NotImplementedError("native Resize currently supports nearest/asymmetric")
            out = self._get_or_create_graph_value(node.outputs[0])
            self.device.dispatch_resize_nearest_float32_into(values[node.inputs[0]], out)
            values[node.outputs[0]] = out
        elif node.op == "MaxPool":
            sppf = self._try_run_sppf_tail_superblock(index, values)
            if sppf is not None:
                return sppf
            if int(node.attrs.get("ceil_mode", 0)) != 0:
                raise NotImplementedError("native MaxPool currently supports ceil_mode=0")
            out = self._get_or_create_graph_value(node.outputs[0])
            self.device.dispatch_maxpool2d_float32_into(
                values[node.inputs[0]],
                out,
                kernel_shape=node.attrs["kernel_shape"],
                strides=node.attrs.get("strides", node.attrs["kernel_shape"]),
                pads=node.attrs.get("pads", [0, 0, 0, 0]),
                dilations=node.attrs.get("dilations", [1, 1]),
            )
            values[node.outputs[0]] = out
        elif node.op == "Transpose":
            out = self._get_or_create_graph_value(node.outputs[0])
            self.device.dispatch_transpose_float32_into(values[node.inputs[0]], out, node.attrs["axes"])
            values[node.outputs[0]] = out
        elif node.op == "Softmax":
            axis = _normalize_axis(int(node.attrs.get("axis", -1)), len(values[node.inputs[0]].shape or ()))
            if axis != 1:
                raise NotImplementedError("native Softmax currently supports axis=1")
            out = self._get_or_create_graph_value(node.outputs[0])
            self.device.dispatch_softmax_axis1_float32_into(values[node.inputs[0]], out)
            values[node.outputs[0]] = out
        else:
            self._run_generic_op_cpu_fallback(node, values)
        return index + 1

    def _run_generic_op_cpu_fallback(self, node, values: Dict[str, DeviceBuffer]) -> None:
        """native 路径无对应 kernel 的算子：下载 -> numpy 求值 -> 上传。

        “任给 ONNX 不崩”的兜底路径：数值正确，性能不作承诺。
        注意：调用方负责在 batch 录制期间先 end_batch 再调用本方法。
        """
        if not _cpu_fallback_enabled():
            raise NotImplementedError(f"native generic DAG op not implemented: {node.op}")
        arrays = []
        for name in node.inputs:
            buf = values.get(name)
            if buf is None:
                raise KeyError(f"native CPU fallback missing input value: {name}")
            if isinstance(buf, DeviceBuffer):
                arrays.append(self.device.download(buf))
            else:
                arrays.append(np.asarray(buf))
        result = _numpy_ops.eval_node(node.op, dict(node.attrs), arrays)
        result = np.ascontiguousarray(result)
        if result.dtype != np.float32:
            result = result.astype(np.float32)
        values[node.outputs[0]] = self.device.upload(result, label=node.outputs[0])

    def _run_generic_graph_to_device_outputs(self, inputs: Dict[str, Any]) -> Dict[str, DeviceBuffer]:
        assert self.graph is not None
        values: Dict[str, DeviceBuffer] = dict(self.constants)
        for name, spec in self.graph.inputs.items():
            if name not in inputs:
                raise ValueError(f"missing input: {name}")
            arr = np.ascontiguousarray(inputs[name], dtype=np.float32)
            if tuple(arr.shape) != tuple(spec.shape):
                raise ValueError(f"expected input {name} shape {spec.shape}, got {tuple(arr.shape)}")
            values[name] = self.device.upload(arr, label=name)

        batch_open = False
        if self._batch_recording:
            self.device.begin_batch()
            batch_open = True
        try:
            i = 0
            while i < len(self.graph.nodes):
                node = self.graph.nodes[i]
                if node.op == "Conv":
                    if self._is_dfl_projection_conv(node):
                        self._run_dfl_projection(node, values)
                        i += 1
                        continue
                    c2f = self._try_run_c2f_bottleneck_superblock(i, values)
                    if c2f is not None:
                        i = c2f
                        continue
                    silu = self._try_run_generic_conv_silu(node, i, values)
                    if silu is not None:
                        i = silu
                        continue
                    if self._batch_recording and self._needs_batch_boundary_for_conv(node):
                        self.device.end_batch()
                        batch_open = False
                        self._run_generic_conv(node, values, silu=False)
                        self.device.begin_batch()
                        batch_open = True
                        i += 1
                        continue
                    self._run_generic_conv(node, values, silu=False)
                elif node.op in {"Gelu", "Relu", "Sigmoid", "Tanh"}:
                    x = values[node.inputs[0]]
                    y = self._get_or_create_graph_value(node.outputs[0])
                    self.device.dispatch_unary_float32_into(node.op, x, y)
                    values[node.outputs[0]] = y
                elif node.op in {"Add", "Sub", "Mul", "Div"}:
                    if node.op == "Add" and self._try_defer_residual_add(i):
                        i += 1
                        continue
                    a = values[node.inputs[0]]
                    b = values[node.inputs[1]]
                    y = self._get_or_create_graph_value(node.outputs[0])
                    self.device.dispatch_binary_broadcast_float32_into(node.op, a, b, y)
                    values[node.outputs[0]] = y
                elif node.op == "Identity":
                    values[node.outputs[0]] = values[node.inputs[0]]
                elif node.op == "Reshape":
                    src = values[node.inputs[0]]
                    spec = self._generic_specs[node.outputs[0]]
                    values[node.outputs[0]] = DeviceBuffer(src.device, src.nbytes, dtype=src.dtype, shape=spec.shape, label=node.outputs[0], handle=src.handle)
                elif node.op == "Split":
                    if not self._try_emit_channel_split_views(node, values):
                        src = values[node.inputs[0]]
                        axis = _normalize_axis(int(node.attrs.get("axis", 0)), len(src.shape or ()))
                        start = 0
                        for out_name in node.outputs:
                            out = self._get_or_create_graph_value(out_name)
                            self.device.dispatch_slice_float32_into(src, out, axis=axis, start=start)
                            values[out_name] = out
                            start += int(out.shape[axis])
                elif node.op == "Slice":
                    src = values[node.inputs[0]]
                    axes = node.attrs.get("axes") or [0]
                    starts = node.attrs.get("starts") or [0]
                    steps = node.attrs.get("steps") or [1]
                    if len(axes) != 1 or len(starts) != 1 or int(steps[0]) != 1:
                        raise NotImplementedError("native Slice currently supports one axis with step=1")
                    axis = _normalize_axis(int(axes[0]), len(src.shape or ()))
                    out = self._get_or_create_graph_value(node.outputs[0])
                    self.device.dispatch_slice_float32_into(src, out, axis=axis, start=int(starts[0]))
                    values[node.outputs[0]] = out
                elif node.op == "Concat":
                    fused = self._try_run_concat_conv1x1(i, values)
                    if fused is not None:
                        i = fused
                        continue
                    axis = _normalize_axis(int(node.attrs.get("axis", 0)), len(self._generic_specs[node.outputs[0]].shape))
                    out = self._get_or_create_graph_value(node.outputs[0])
                    self.device.dispatch_concat_float32_into([values[name] for name in node.inputs], out, axis=axis)
                    values[node.outputs[0]] = out
                elif node.op == "Resize":
                    mode = str(node.attrs.get("mode", "nearest"))
                    coord = str(node.attrs.get("coordinate_transformation_mode", "asymmetric"))
                    if mode != "nearest" or coord != "asymmetric":
                        raise NotImplementedError("native Resize currently supports nearest/asymmetric")
                    out = self._get_or_create_graph_value(node.outputs[0])
                    self.device.dispatch_resize_nearest_float32_into(values[node.inputs[0]], out)
                    values[node.outputs[0]] = out
                elif node.op == "MaxPool":
                    sppf = self._try_run_sppf_tail_superblock(i, values)
                    if sppf is not None:
                        i = sppf
                        continue
                    if int(node.attrs.get("ceil_mode", 0)) != 0:
                        raise NotImplementedError("native MaxPool currently supports ceil_mode=0")
                    out = self._get_or_create_graph_value(node.outputs[0])
                    self.device.dispatch_maxpool2d_float32_into(
                        values[node.inputs[0]],
                        out,
                        kernel_shape=node.attrs["kernel_shape"],
                        strides=node.attrs.get("strides", node.attrs["kernel_shape"]),
                        pads=node.attrs.get("pads", [0, 0, 0, 0]),
                        dilations=node.attrs.get("dilations", [1, 1]),
                    )
                    values[node.outputs[0]] = out
                elif node.op == "Transpose":
                    out = self._get_or_create_graph_value(node.outputs[0])
                    self.device.dispatch_transpose_float32_into(values[node.inputs[0]], out, node.attrs["axes"])
                    values[node.outputs[0]] = out
                elif node.op == "Softmax":
                    axis = _normalize_axis(int(node.attrs.get("axis", -1)), len(values[node.inputs[0]].shape or ()))
                    if axis != 1:
                        raise NotImplementedError("native Softmax currently supports axis=1")
                    out = self._get_or_create_graph_value(node.outputs[0])
                    self.device.dispatch_softmax_axis1_float32_into(values[node.inputs[0]], out)
                    values[node.outputs[0]] = out
                else:
                    if batch_open:
                        self.device.end_batch()
                        batch_open = False
                        self._run_generic_op_cpu_fallback(node, values)
                        self.device.begin_batch()
                        batch_open = True
                    else:
                        self._run_generic_op_cpu_fallback(node, values)
                i += 1
            if batch_open:
                self.device.end_batch()
                batch_open = False
        except Exception:
            if batch_open:
                try:
                    self.device.end_batch()
                except Exception:
                    pass
            raise
        return {name: values[name] for name in self.graph.outputs}

    def _try_run_generic_conv_silu(self, node, index: int, values: Dict[str, DeviceBuffer]) -> int | None:
        assert self.graph is not None
        if index + 2 >= len(self.graph.nodes) or len(node.outputs) != 1:
            return None
        sigmoid = self.graph.nodes[index + 1]
        mul = self.graph.nodes[index + 2]
        conv_out = node.outputs[0]
        if sigmoid.op != "Sigmoid" or mul.op != "Mul" or sigmoid.inputs != [conv_out]:
            return None
        if set(mul.inputs) != {conv_out, sigmoid.outputs[0]} or len(mul.outputs) != 1:
            return None
        self._run_generic_conv(node, values, silu=True, output_name=mul.outputs[0])
        return index + 3

    def _try_run_c2f_bottleneck_superblock(self, index: int, values: Dict[str, DeviceBuffer]) -> int | None:
        assert self.graph is not None
        if not self._c2f_front_enabled or index + 6 >= len(self.graph.nodes):
            return None
        nodes = self.graph.nodes
        conv1, sig1, mul1, conv2, sig2, mul2, add = nodes[index:index + 7]
        if conv1.op != "Conv" or conv2.op != "Conv" or sig1.op != "Sigmoid" or sig2.op != "Sigmoid" or mul1.op != "Mul" or mul2.op != "Mul" or add.op != "Add":
            return None
        if len(conv1.outputs) != 1 or len(conv2.outputs) != 1 or len(sig1.outputs) != 1 or len(sig2.outputs) != 1 or len(mul1.outputs) != 1 or len(mul2.outputs) != 1 or len(add.outputs) != 1:
            return None
        x_name = conv1.inputs[0]
        conv1_out = conv1.outputs[0]
        conv1_act = mul1.outputs[0]
        conv2_out = conv2.outputs[0]
        conv2_act = mul2.outputs[0]
        if sig1.inputs != [conv1_out] or set(mul1.inputs) != {conv1_out, sig1.outputs[0]}:
            return None
        if conv2.inputs[0] != conv1_act or sig2.inputs != [conv2_out] or set(mul2.inputs) != {conv2_out, sig2.outputs[0]}:
            return None
        if set(add.inputs) != {x_name, conv2_act}:
            return None
        if self._value_use_count(conv1_out) != 2 or self._value_use_count(conv1_act) != 1 or self._value_use_count(conv2_out) != 2 or self._value_use_count(conv2_act) != 1:
            return None
        if not self._is_same_3x3_conv(conv1) or not self._is_same_3x3_conv(conv2):
            return None
        if len(conv1.inputs) < 2 or len(conv2.inputs) < 2 or conv1.inputs[1] not in self.graph.constants or conv2.inputs[1] not in self.graph.constants:
            return None
        x = values.get(x_name)
        if x is None or x.shape is None or len(x.shape) != 4:
            return None
        w1_arr = np.asarray(self.graph.constants[conv1.inputs[1]], dtype=np.float32)
        w2_arr = np.asarray(self.graph.constants[conv2.inputs[1]], dtype=np.float32)
        if w1_arr.ndim != 4 or w2_arr.ndim != 4 or tuple(w1_arr.shape[2:]) != (3, 3) or tuple(w2_arr.shape[2:]) != (3, 3):
            return None
        in_channels = int(x.shape[1])
        if int(w1_arr.shape[1]) != in_channels or int(w2_arr.shape[1]) != int(w1_arr.shape[0]) or int(w2_arr.shape[0]) != in_channels:
            return None
        w1 = values[conv1.inputs[1]]
        b1 = values[conv1.inputs[2]] if len(conv1.inputs) >= 3 and conv1.inputs[2] in values else self._get_zero_bias(int(w1_arr.shape[0]))
        w2 = values[conv2.inputs[1]]
        b2 = values[conv2.inputs[2]] if len(conv2.inputs) >= 3 and conv2.inputs[2] in values else self._get_zero_bias(int(w2_arr.shape[0]))
        out = self._get_or_create_graph_value(add.outputs[0])
        self.device.dispatch_c2f_bottleneck_tiled_float32_into(x, w1, b1, w2, b2, out)
        values[add.outputs[0]] = out
        self._fused_c2f_bottleneck_count += 1
        return index + 7

    def _try_run_sppf_tail_superblock(self, index: int, values: Dict[str, DeviceBuffer]) -> int | None:
        assert self.graph is not None
        if not _sppf_superblock_enabled():
            return None
        nodes = self.graph.nodes
        if index + 6 >= len(nodes):
            return None
        pool1, pool2, pool3, concat, conv, sigmoid, mul = nodes[index:index + 7]
        if pool1.op != "MaxPool" or pool2.op != "MaxPool" or pool3.op != "MaxPool" or concat.op != "Concat" or conv.op != "Conv":
            return None
        if sigmoid.op != "Sigmoid" or mul.op != "Mul":
            return None
        if len(pool1.inputs) != 1 or len(pool1.outputs) != 1 or len(pool2.outputs) != 1 or len(pool3.outputs) != 1:
            return None
        x_name = pool1.inputs[0]
        p1 = pool1.outputs[0]
        p2 = pool2.outputs[0]
        p3 = pool3.outputs[0]
        if pool2.inputs != [p1] or pool3.inputs != [p2]:
            return None
        if concat.inputs != [x_name, p1, p2, p3] or len(concat.outputs) != 1:
            return None
        concat_out = concat.outputs[0]
        conv_out = conv.outputs[0] if len(conv.outputs) == 1 else None
        if conv_out is None or conv.inputs[0] != concat_out:
            return None
        if sigmoid.inputs != [conv_out] or len(sigmoid.outputs) != 1 or set(mul.inputs) != {conv_out, sigmoid.outputs[0]} or len(mul.outputs) != 1:
            return None
        if self._value_use_count(p1) != 2 or self._value_use_count(p2) != 2 or self._value_use_count(p3) != 1 or self._value_use_count(concat_out) != 1 or self._value_use_count(conv_out) != 2:
            return None
        if not all(_is_sppf_pool(node) for node in (pool1, pool2, pool3)):
            return None
        if len(conv.inputs) < 2 or conv.inputs[1] not in self.graph.constants:
            return None
        weight_arr = np.asarray(self.graph.constants[conv.inputs[1]], dtype=np.float32)
        if weight_arr.ndim != 4 or tuple(int(v) for v in weight_arr.shape[2:]) != (1, 1):
            return None
        strides = tuple(int(v) for v in conv.attrs.get("strides", [1, 1]))
        pads = tuple(int(v) for v in conv.attrs.get("pads", [0, 0, 0, 0]))
        dilations = tuple(int(v) for v in conv.attrs.get("dilations", [1, 1]))
        if strides != (1, 1) or pads != (0, 0, 0, 0) or dilations != (1, 1) or int(conv.attrs.get("group", 1)) != 1:
            return None
        x = values.get(x_name)
        if x is None or x.shape is None or len(x.shape) != 4:
            return None
        n, channels, h, w = (int(v) for v in x.shape)
        if int(weight_arr.shape[1]) != channels * 4:
            return None
        weight = values[conv.inputs[1]]
        if len(conv.inputs) >= 3 and conv.inputs[2] in values:
            bias = values[conv.inputs[2]]
        else:
            bias = self._get_zero_bias(int(weight_arr.shape[0]))
        out = self._get_or_create_graph_value(mul.outputs[0])
        if out.shape is None or tuple(int(v) for v in out.shape) != (n, int(weight_arr.shape[0]), h, w):
            return None
        self.device.dispatch_sppf_tail_float32_into(x, weight, bias, out, activation="silu")
        values[mul.outputs[0]] = out
        self._fused_sppf_tail_count += 1
        return index + 7

    def _is_same_3x3_conv(self, node) -> bool:
        strides = tuple(int(v) for v in node.attrs.get("strides", [1, 1]))
        pads = tuple(int(v) for v in node.attrs.get("pads", [0, 0, 0, 0]))
        dilations = tuple(int(v) for v in node.attrs.get("dilations", [1, 1]))
        return strides == (1, 1) and pads == (1, 1, 1, 1) and dilations == (1, 1) and int(node.attrs.get("group", 1)) == 1

    def _try_emit_channel_split_views(self, node, values: Dict[str, DeviceBuffer]) -> bool:
        src = values[node.inputs[0]]
        src_shape = tuple(int(v) for v in (src.shape or ()))
        if not src_shape or len(src_shape) < 2:
            return False
        axis = _normalize_axis(int(node.attrs.get("axis", 0)), len(src_shape))
        if axis != 1 or src_shape[0] != 1:
            return False
        trailing = 1
        for dim in src_shape[axis + 1:]:
            trailing *= int(dim)
        start = 0
        for out_name in node.outputs:
            spec = self._generic_specs.get(out_name)
            if spec is None:
                return False
            out_shape = tuple(int(v) for v in spec.shape)
            if len(out_shape) != len(src_shape) or out_shape[0] != 1 or tuple(out_shape[2:]) != tuple(src_shape[2:]):
                return False
            channels = int(out_shape[axis])
            if channels <= 0:
                return False
            view = self.device.create_buffer_view(
                src,
                element_offset=start * trailing,
                element_count=channels * trailing,
                dtype="float32",
                shape=out_shape,
                label=f"{out_name}_view",
            )
            values[out_name] = view
            start += channels
        return start == src_shape[axis]

    def _try_run_concat_conv1x1(self, index: int, values: Dict[str, DeviceBuffer]) -> int | None:
        assert self.graph is not None
        nodes = self.graph.nodes
        if index + 1 >= len(nodes):
            return None
        concat = nodes[index]
        conv = nodes[index + 1]
        if concat.op != "Concat" or conv.op != "Conv" or len(concat.outputs) != 1 or len(conv.outputs) != 1:
            return None
        concat_out = concat.outputs[0]
        if conv.inputs[0] != concat_out or len(concat.inputs) > 8:
            return None
        if self._value_use_count(concat_out) != 1 or concat_out in self.graph.outputs:
            return None
        out_spec = self._generic_specs.get(conv.outputs[0])
        if out_spec is None:
            return None
        axis = _normalize_axis(int(concat.attrs.get("axis", 0)), len(out_spec.shape))
        if axis != 1:
            return None
        if len(conv.inputs) < 2 or conv.inputs[1] not in self.graph.constants:
            return None
        weight_arr = np.asarray(self.graph.constants[conv.inputs[1]], dtype=np.float32)
        if weight_arr.ndim != 4 or tuple(weight_arr.shape[2:]) != (1, 1):
            return None
        strides = tuple(int(v) for v in conv.attrs.get("strides", [1, 1]))
        pads = tuple(int(v) for v in conv.attrs.get("pads", [0, 0, 0, 0]))
        dilations = tuple(int(v) for v in conv.attrs.get("dilations", [1, 1]))
        if strides != (1, 1) or pads != (0, 0, 0, 0) or dilations != (1, 1) or int(conv.attrs.get("group", 1)) != 1:
            return None
        branch_infos = [self._resolve_concat_branch(name, values) for name in concat.inputs]
        if any(info is None for info in branch_infos):
            return None
        resolved = [info for info in branch_infos if info is not None]
        input_buffers = [info[0] for info in resolved]
        residual_buffers = [info[1] for info in resolved]
        residual_flags = [info[2] for info in resolved]
        if any(buffer.shape is None or len(buffer.shape) != 4 for buffer in input_buffers):
            return None
        shapes = [tuple(int(v) for v in buffer.shape) for buffer in input_buffers]
        n, _, h, w = shapes[0]
        if any(shape[0] != n or shape[2:] != (h, w) for shape in shapes):
            return None
        for residual, flag, shape in zip(residual_buffers, residual_flags, shapes):
            if flag and (residual.shape is None or tuple(int(v) for v in residual.shape) != shape):
                return None
        total_channels = sum(shape[1] for shape in shapes)
        if int(weight_arr.shape[1]) != total_channels:
            return None

        activation = "linear"
        output_name = conv.outputs[0]
        next_index = index + 2
        if index + 3 < len(nodes):
            sigmoid = nodes[index + 2]
            mul = nodes[index + 3]
            conv_out = conv.outputs[0]
            if (
                sigmoid.op == "Sigmoid"
                and mul.op == "Mul"
                and sigmoid.inputs == [conv_out]
                and len(sigmoid.outputs) == 1
                and set(mul.inputs) == {conv_out, sigmoid.outputs[0]}
                and len(mul.outputs) == 1
                and self._value_use_count(conv_out) == 2
            ):
                activation = "silu"
                output_name = mul.outputs[0]
                next_index = index + 4

        weight = values[conv.inputs[1]]
        int8_pack = None
        if not any(residual_flags) and _int8_superblock_enabled():
            int8_pack = self._get_int8_superblock_weight(conv.inputs[1], weight_arr)
        if int8_pack is None and not any(residual_flags) and _fp16_superblock_enabled():
            packed_weight = self._get_fp16_superblock_weight(conv.inputs[1], weight_arr)
            if packed_weight is not None:
                weight = packed_weight
        if len(conv.inputs) >= 3 and conv.inputs[2] in values:
            bias = values[conv.inputs[2]]
        else:
            bias = self._get_zero_bias(int(weight_arr.shape[0]))
        out = self._get_or_create_graph_value(output_name)
        if any(residual_flags):
            self.device.dispatch_concat_residual_conv1x1_float32_into(
                input_buffers,
                residual_buffers,
                residual_flags,
                weight,
                bias,
                out,
                activation=activation,
            )
            self._fused_c2f_residual_tail_count += 1
        else:
            if int8_pack is not None:
                int8_weight, product_scale, activation_scale = int8_pack
                self.device.dispatch_concat_conv1x1_int8_float32_into(
                    input_buffers,
                    int8_weight,
                    product_scale,
                    bias,
                    out,
                    activation_scale=activation_scale,
                    activation=activation,
                )
            else:
                self.device.dispatch_concat_conv1x1_float32_into(input_buffers, weight, bias, out, activation=activation)
        values[output_name] = out
        self._fused_concat_conv1x1_count += 1
        return next_index

    def _get_int8_superblock_weight(self, weight_name: str, weight_arr: np.ndarray) -> tuple[DeviceBuffer, DeviceBuffer, float] | None:
        if weight_name not in self.constants:
            return None
        if weight_arr.ndim != 4 or tuple(int(v) for v in weight_arr.shape[2:]) != (1, 1):
            return None
        existing = self._int8_superblock_constants.get(weight_name)
        if existing is not None:
            return existing
        activation_scale = _int8_activation_scale()
        packed_weight, product_scale = _pack_int8_per_channel_1x1(weight_arr, activation_scale)
        raw_weight = self.device.upload(packed_weight, label=f"{weight_name}_int8x4")
        wrapped_weight = DeviceBuffer(
            raw_weight.device,
            raw_weight.nbytes,
            dtype="uint32",
            shape=tuple(int(v) for v in weight_arr.shape),
            label=raw_weight.label,
            handle=raw_weight.handle,
        )
        scale_buffer = self.device.upload(product_scale, label=f"{weight_name}_int8_scale")
        packed = (wrapped_weight, scale_buffer, activation_scale)
        self._int8_superblock_constants[weight_name] = packed
        return packed

    def _get_fp16_superblock_weight(self, weight_name: str, weight_arr: np.ndarray) -> DeviceBuffer | None:
        if weight_name not in self.constants:
            return None
        if weight_arr.ndim != 4 or tuple(int(v) for v in weight_arr.shape[2:]) != (1, 1):
            return None
        existing = self._fp16_superblock_constants.get(weight_name)
        if existing is not None:
            return existing
        packed = _pack_fp16_to_uint32(weight_arr.reshape(-1))
        raw = self.device.upload(packed, label=f"{weight_name}_fp16_half2")
        wrapped = DeviceBuffer(
            raw.device,
            raw.nbytes,
            dtype="uint32",
            shape=tuple(int(v) for v in weight_arr.shape),
            label=raw.label,
            handle=raw.handle,
        )
        self._fp16_superblock_constants[weight_name] = wrapped
        return wrapped

    def _prepack_fp16_superblock_constants_for_graph(self) -> None:
        if not _fp16_superblock_enabled() or self.graph is None:
            return
        nodes = self.graph.nodes
        for index, node in enumerate(nodes[:-1]):
            if node.op != "Concat" or len(node.outputs) != 1 or len(node.inputs) > 8:
                continue
            conv = nodes[index + 1]
            if conv.op != "Conv" or len(conv.inputs) < 2 or conv.inputs[0] != node.outputs[0]:
                continue
            weight_name = conv.inputs[1]
            if weight_name not in self.graph.constants or weight_name in self._fp16_superblock_constants:
                continue
            weight_arr = np.asarray(self.graph.constants[weight_name], dtype=np.float32)
            if weight_arr.ndim != 4 or tuple(int(v) for v in weight_arr.shape[2:]) != (1, 1):
                continue
            strides = tuple(int(v) for v in conv.attrs.get("strides", [1, 1]))
            pads = tuple(int(v) for v in conv.attrs.get("pads", [0, 0, 0, 0]))
            dilations = tuple(int(v) for v in conv.attrs.get("dilations", [1, 1]))
            if strides != (1, 1) or pads != (0, 0, 0, 0) or dilations != (1, 1) or int(conv.attrs.get("group", 1)) != 1:
                continue
            self._get_fp16_superblock_weight(weight_name, weight_arr)

    def _prepack_int8_superblock_constants_for_graph(self) -> None:
        if not _int8_superblock_enabled() or self.graph is None:
            return
        nodes = self.graph.nodes
        for index, node in enumerate(nodes[:-1]):
            if node.op != "Concat" or len(node.outputs) != 1 or len(node.inputs) > 8:
                continue
            conv = nodes[index + 1]
            if conv.op != "Conv" or len(conv.inputs) < 2 or conv.inputs[0] != node.outputs[0]:
                continue
            weight_name = conv.inputs[1]
            if weight_name not in self.graph.constants or weight_name in self._int8_superblock_constants:
                continue
            weight_arr = np.asarray(self.graph.constants[weight_name], dtype=np.float32)
            if weight_arr.ndim != 4 or tuple(int(v) for v in weight_arr.shape[2:]) != (1, 1):
                continue
            strides = tuple(int(v) for v in conv.attrs.get("strides", [1, 1]))
            pads = tuple(int(v) for v in conv.attrs.get("pads", [0, 0, 0, 0]))
            dilations = tuple(int(v) for v in conv.attrs.get("dilations", [1, 1]))
            if strides != (1, 1) or pads != (0, 0, 0, 0) or dilations != (1, 1) or int(conv.attrs.get("group", 1)) != 1:
                continue
            self._get_int8_superblock_weight(weight_name, weight_arr)

    def _compile_c2f_superblocks(self) -> list[Dict[str, Any]]:
        if not self._c2f_superblock_enabled or self.graph is None:
            return []
        blocks: list[Dict[str, Any]] = []
        nodes = self.graph.nodes
        i = 0
        while i < len(nodes):
            block = self._match_c2f_superblock_at(i)
            if block is None:
                i += 1
                continue
            blocks.append(block)
            for out_name, pair in block["residual_adds"].items():
                self._c2f_virtual_residual_candidates[out_name] = pair
            i = int(block["end"]) + 1
        return blocks

    def _match_c2f_superblock_at(self, index: int) -> Dict[str, Any] | None:
        assert self.graph is not None
        nodes = self.graph.nodes
        if index + 5 >= len(nodes):
            return None
        cv1 = nodes[index]
        sig = nodes[index + 1]
        mul = nodes[index + 2]
        split = nodes[index + 3]
        if cv1.op != "Conv" or sig.op != "Sigmoid" or mul.op != "Mul" or split.op != "Split":
            return None
        if len(cv1.outputs) != 1 or sig.inputs != cv1.outputs or len(sig.outputs) != 1:
            return None
        cv1_act = mul.outputs[0] if len(mul.outputs) == 1 else None
        if cv1_act is None or set(mul.inputs) != {cv1.outputs[0], sig.outputs[0]}:
            return None
        if split.inputs != [cv1_act] or len(split.outputs) != 2 or int(split.attrs.get("axis", 0)) != 1:
            return None
        cv1_weight = self.graph.constants.get(cv1.inputs[1]) if len(cv1.inputs) >= 2 else None
        if cv1_weight is None or np.asarray(cv1_weight).ndim != 4 or tuple(np.asarray(cv1_weight).shape[2:]) != (1, 1):
            return None

        left_name, current = split.outputs
        concat_inputs = [left_name, current]
        bottlenecks = []
        residual_adds: Dict[str, tuple[str, str]] = {}
        j = index + 4
        while j + 2 < len(nodes):
            if nodes[j].op == "Concat":
                break
            parsed = self._match_c2f_bottleneck_at(j, current)
            if parsed is None:
                return None
            bottlenecks.append(parsed)
            current = parsed["output"]
            concat_inputs.append(current)
            if parsed["residual"] is not None:
                residual_adds[current] = parsed["residual"]
            j = parsed["next"]
        if not bottlenecks or j + 1 >= len(nodes):
            return None
        concat = nodes[j]
        cv2 = nodes[j + 1]
        if concat.op != "Concat" or concat.attrs.get("axis") != 1 or concat.inputs != concat_inputs or len(concat.outputs) != 1:
            return None
        if cv2.op != "Conv" or len(cv2.inputs) < 2 or cv2.inputs[0] != concat.outputs[0] or len(cv2.outputs) != 1:
            return None
        cv2_weight = self.graph.constants.get(cv2.inputs[1])
        if cv2_weight is None or np.asarray(cv2_weight).ndim != 4 or tuple(np.asarray(cv2_weight).shape[2:]) != (1, 1):
            return None
        end = j + 2
        if j + 3 < len(nodes):
            sig2 = nodes[j + 2]
            mul2 = nodes[j + 3]
            if (
                sig2.op == "Sigmoid"
                and mul2.op == "Mul"
                and sig2.inputs == cv2.outputs
                and len(sig2.outputs) == 1
                and len(mul2.outputs) == 1
                and set(mul2.inputs) == {cv2.outputs[0], sig2.outputs[0]}
            ):
                end = j + 3
        return {
            "start": index,
            "end": end,
            "cv1": cv1.name,
            "split": split.name,
            "concat": concat.name,
            "cv2": cv2.name,
            "bottleneck_count": len(bottlenecks),
            "residual_adds": residual_adds,
        }

    def _match_c2f_bottleneck_at(self, index: int, input_name: str) -> Dict[str, Any] | None:
        assert self.graph is not None
        nodes = self.graph.nodes
        if index + 5 >= len(nodes):
            return None
        conv1, sig1, mul1, conv2, sig2, mul2 = nodes[index:index + 6]
        if conv1.op != "Conv" or sig1.op != "Sigmoid" or mul1.op != "Mul" or conv2.op != "Conv" or sig2.op != "Sigmoid" or mul2.op != "Mul":
            return None
        if conv1.inputs[0] != input_name or len(conv1.outputs) != 1 or sig1.inputs != conv1.outputs or len(sig1.outputs) != 1 or len(mul1.outputs) != 1:
            return None
        conv1_act = mul1.outputs[0]
        if set(mul1.inputs) != {conv1.outputs[0], sig1.outputs[0]}:
            return None
        if conv2.inputs[0] != conv1_act or len(conv2.outputs) != 1 or sig2.inputs != conv2.outputs or len(sig2.outputs) != 1 or len(mul2.outputs) != 1:
            return None
        conv2_act = mul2.outputs[0]
        if set(mul2.inputs) != {conv2.outputs[0], sig2.outputs[0]}:
            return None
        if not self._is_same_3x3_conv(conv1) or not self._is_same_3x3_conv(conv2):
            return None
        residual = None
        output = conv2_act
        next_index = index + 6
        if next_index < len(nodes) and nodes[next_index].op == "Add":
            add = nodes[next_index]
            if len(add.outputs) != 1 or set(add.inputs) != {input_name, conv2_act}:
                return None
            residual = (input_name, conv2_act)
            output = add.outputs[0]
            next_index += 1
        return {"output": output, "next": next_index, "residual": residual}

    def _try_defer_residual_add(self, index: int) -> bool:
        assert self.graph is not None
        if not self._c2f_superblock_enabled:
            return False
        node = self.graph.nodes[index]
        if node.op != "Add" or len(node.inputs) != 2 or len(node.outputs) != 1:
            return False
        out_name = node.outputs[0]
        if self._value_use_count(out_name) != 1:
            return False
        a_spec = self._generic_specs.get(node.inputs[0])
        b_spec = self._generic_specs.get(node.inputs[1])
        out_spec = self._generic_specs.get(out_name)
        if a_spec is None or b_spec is None or out_spec is None:
            return False
        if tuple(a_spec.shape) != tuple(b_spec.shape) or tuple(a_spec.shape) != tuple(out_spec.shape) or len(out_spec.shape) != 4:
            return False
        planned = self._c2f_virtual_residual_candidates.get(out_name)
        if planned is not None:
            self._virtual_residual_adds[out_name] = planned
            return True
        for j in range(index + 1, len(self.graph.nodes) - 1):
            concat = self.graph.nodes[j]
            if concat.op == "Concat" and out_name in concat.inputs and self._concat_followed_by_1x1_conv(j):
                self._virtual_residual_adds[out_name] = (node.inputs[0], node.inputs[1])
                return True
        return False

    def _concat_followed_by_1x1_conv(self, index: int) -> bool:
        assert self.graph is not None
        nodes = self.graph.nodes
        if index + 1 >= len(nodes):
            return False
        concat = nodes[index]
        conv = nodes[index + 1]
        if concat.op != "Concat" or conv.op != "Conv" or len(concat.outputs) != 1 or conv.inputs[0] != concat.outputs[0]:
            return False
        if len(conv.inputs) < 2 or conv.inputs[1] not in self.graph.constants:
            return False
        weight = np.asarray(self.graph.constants[conv.inputs[1]], dtype=np.float32)
        strides = tuple(int(v) for v in conv.attrs.get("strides", [1, 1]))
        pads = tuple(int(v) for v in conv.attrs.get("pads", [0, 0, 0, 0]))
        dilations = tuple(int(v) for v in conv.attrs.get("dilations", [1, 1]))
        return (
            weight.ndim == 4
            and tuple(weight.shape[2:]) == (1, 1)
            and strides == (1, 1)
            and pads == (0, 0, 0, 0)
            and dilations == (1, 1)
            and int(conv.attrs.get("group", 1)) == 1
        )

    def _resolve_concat_branch(self, name: str, values: Dict[str, DeviceBuffer]) -> tuple[DeviceBuffer, DeviceBuffer, int] | None:
        if name in self._virtual_residual_adds:
            a_name, b_name = self._virtual_residual_adds[name]
            a = values.get(a_name)
            b = values.get(b_name)
            if a is None or b is None:
                return None
            return a, b, 1
        buffer = values.get(name)
        if buffer is None:
            return None
        return buffer, buffer, 0

    def _value_use_count(self, name: str) -> int:
        assert self.graph is not None
        count = 0
        for node in self.graph.nodes:
            count += sum(1 for input_name in node.inputs if input_name == name)
        count += sum(1 for output_name in self.graph.outputs if output_name == name)
        return count

    def _run_generic_conv(self, node, values: Dict[str, DeviceBuffer], *, silu: bool, output_name: str | None = None) -> None:
        x = values[node.inputs[0]]
        weight_arr = np.asarray(self.graph.constants[node.inputs[1]], dtype=np.float32)
        out_channels = int(weight_arr.shape[0])
        if len(node.inputs) >= 3 and node.inputs[2] in values:
            bias = values[node.inputs[2]]
        else:
            bias = self._get_zero_bias(out_channels)
        out_name = output_name or node.outputs[0]
        out = self._get_or_create_graph_value(out_name)
        desc = _conv_desc_from_shapes(x.shape, weight_arr.shape, out.shape, node.attrs)
        weight = self._get_conv_weight_buffer(node, silu=silu, desc=desc)
        if silu:
            self.device.dispatch_conv2d_silu_float32_into(x, weight, bias, out, desc)
        else:
            self.device.dispatch_conv2d_float32_into(x, weight, bias, out, desc)
        values[out_name] = out

    def _get_conv_weight_buffer(self, node, *, silu: bool, desc: Dict[str, int]) -> DeviceBuffer:
        assert self.graph is not None
        weight_name = node.inputs[1]
        if (
            silu
            and _winograd_packed_enabled()
            and weight_name in self.graph.constants
            and weight_name in self.constants
        ):
            weight_arr = np.asarray(self.graph.constants[weight_name], dtype=np.float32)
            if _is_winograd_packable_desc(desc) and weight_arr.ndim == 4 and tuple(weight_arr.shape[2:]) == (3, 3):
                packed = self._winograd_packed_constants.get(weight_name)
                if packed is None:
                    packed_value = _pack_winograd_f2x2_3x3_weights(weight_arr)
                    packed = self.device.upload(packed_value, label=f"{weight_name}_winograd_u")
                    self._winograd_packed_constants[weight_name] = packed
                return packed
        return self.constants[weight_name]

    def _prepack_winograd_constants_for_graph(self) -> None:
        if not _winograd_packed_enabled() or self.graph is None:
            return
        nodes = self.graph.nodes
        for index, node in enumerate(nodes):
            if node.op != "Conv" or len(node.inputs) < 2 or len(node.outputs) != 1:
                continue
            if node.inputs[1] not in self.graph.constants or node.inputs[1] in self._winograd_packed_constants:
                continue
            if index + 2 >= len(nodes):
                continue
            sigmoid = nodes[index + 1]
            mul = nodes[index + 2]
            conv_out = node.outputs[0]
            if sigmoid.op != "Sigmoid" or mul.op != "Mul" or sigmoid.inputs != [conv_out]:
                continue
            if len(sigmoid.outputs) != 1 or set(mul.inputs) != {conv_out, sigmoid.outputs[0]}:
                continue
            input_spec = self._generic_specs.get(node.inputs[0])
            output_spec = self._generic_specs.get(conv_out)
            if input_spec is None or output_spec is None:
                continue
            weight_arr = np.asarray(self.graph.constants[node.inputs[1]], dtype=np.float32)
            if weight_arr.ndim != 4 or tuple(weight_arr.shape[2:]) != (3, 3):
                continue
            desc = _conv_desc_from_shapes(input_spec.shape, weight_arr.shape, output_spec.shape, node.attrs)
            if not _is_winograd_packable_desc(desc):
                continue
            packed = self.device.upload(_pack_winograd_f2x2_3x3_weights(weight_arr), label=f"{node.inputs[1]}_winograd_u")
            self._winograd_packed_constants[node.inputs[1]] = packed

    def _run_dfl_projection(self, node, values: Dict[str, DeviceBuffer]) -> None:
        x = values[node.inputs[0]]
        out = self._get_or_create_graph_value(node.outputs[0])
        self.device.dispatch_dfl_project_float32_into(x, out)
        values[node.outputs[0]] = out

    def _reserve_generic_conv_output(self, node, values: Dict[str, DeviceBuffer], *, output_name: str | None = None) -> None:
        out_name = output_name or node.outputs[0]
        values[out_name] = self._get_or_create_graph_value(out_name)

    def _needs_batch_boundary_for_conv(self, node) -> bool:
        if self._is_dfl_projection_conv(node):
            return False
        if not node.name.endswith("/dfl/conv/Conv"):
            return False
        input_spec = self._generic_specs.get(node.inputs[0])
        output_spec = self._generic_specs.get(node.outputs[0])
        return (
            input_spec is not None
            and output_spec is not None
            and tuple(input_spec.shape[:3]) == (1, 16, 4)
            and tuple(output_spec.shape[:3]) == (1, 1, 4)
        )

    def _is_dfl_projection_conv(self, node) -> bool:
        if self.graph is None or node.op != "Conv" or not node.name.endswith("/dfl/conv/Conv"):
            return False
        if len(node.inputs) < 2 or node.inputs[1] not in self.graph.constants:
            return False
        input_spec = self._generic_specs.get(node.inputs[0])
        output_spec = self._generic_specs.get(node.outputs[0])
        weight = np.asarray(self.graph.constants[node.inputs[1]], dtype=np.float32)
        if input_spec is None or output_spec is None:
            return False
        return (
            tuple(input_spec.shape[:3]) == (1, 16, 4)
            and tuple(output_spec.shape[:3]) == (1, 1, 4)
            and weight.shape == (1, 16, 1, 1)
            and np.allclose(weight.reshape(-1), np.arange(16, dtype=np.float32))
        )

    def _get_or_create_graph_value(self, name: str) -> DeviceBuffer:
        spec = self._generic_specs.get(name)
        if spec is None:
            raise ValueError(f"missing inferred spec for {name}")
        nbytes = int(np.prod(spec.shape)) * 4
        existing = self._generic_buffers.get(name)
        if existing is not None and existing.nbytes == nbytes and existing.shape == spec.shape:
            return existing
        buffer = self.device.allocate_uav(nbytes, dtype="float32", shape=spec.shape, label=name)
        self._generic_buffers[name] = buffer
        return buffer

    def _get_zero_bias(self, channels: int) -> DeviceBuffer:
        channels_i = int(channels)
        existing = self._zero_bias_buffers.get(channels_i)
        if existing is not None:
            return existing
        buffer = self.device.upload(np.zeros((channels_i,), dtype=np.float32), label=f"zero_bias_{channels_i}")
        self._zero_bias_buffers[channels_i] = buffer
        return buffer


def _numel(shape):
    total = 1
    for dim in shape or ():
        total *= int(dim)
    return total


def _normalize_axis(axis: int, rank: int) -> int:
    axis_i = int(axis)
    if axis_i < 0:
        axis_i += int(rank)
    if axis_i < 0 or axis_i >= int(rank):
        raise ValueError(f"axis {axis} is out of range for rank {rank}")
    return axis_i


def _conv_desc_from_buffers(
    input_buffer: DeviceBuffer,
    weight_buffer: DeviceBuffer,
    output_buffer: DeviceBuffer,
    attrs: Dict[str, Any],
) -> Dict[str, int]:
    if input_buffer.shape is None or weight_buffer.shape is None or output_buffer.shape is None:
        raise ValueError("Conv buffers require shape metadata")
    return _conv_desc_from_shapes(input_buffer.shape, weight_buffer.shape, output_buffer.shape, attrs)


def _conv_desc_from_shapes(
    input_shape: Tuple[int, ...],
    weight_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    attrs: Dict[str, Any],
) -> Dict[str, int]:
    n, in_channels, in_h, in_w = (int(v) for v in input_shape)
    out_channels, _, kernel_h, kernel_w = (int(v) for v in weight_shape)
    _, _, out_h, out_w = (int(v) for v in output_shape)
    strides = tuple(int(v) for v in attrs.get("strides", [1, 1]))
    pads = tuple(int(v) for v in attrs.get("pads", [0, 0, 0, 0]))
    dilations = tuple(int(v) for v in attrs.get("dilations", [1, 1]))
    return {
        "batch": n,
        "in_channels": in_channels,
        "in_h": in_h,
        "in_w": in_w,
        "out_channels": out_channels,
        "out_h": out_h,
        "out_w": out_w,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "stride_h": strides[0],
        "stride_w": strides[1],
        "pad_top": pads[0],
        "pad_left": pads[1],
        "dilation_h": dilations[0],
        "dilation_w": dilations[1],
        "groups": int(attrs.get("group", 1)),
    }


def _winograd_packed_enabled() -> bool:
    return os.environ.get("AEXRT_NATIVE_D3D12_WINOGRAD_PACKED", "0") != "0"


def _sppf_superblock_enabled() -> bool:
    return os.environ.get("AEXRT_NATIVE_D3D12_SPPF_SUPERBLOCK", "0") != "0"


def _fp16_superblock_enabled() -> bool:
    return os.environ.get("AEXRT_NATIVE_D3D12_FP16_SUPERBLOCK", "0") != "0"


def _int8_superblock_enabled() -> bool:
    return os.environ.get("AEXRT_NATIVE_D3D12_INT8_SUPERBLOCK", "0") != "0"


def _int8_activation_scale() -> float:
    raw = os.environ.get("AEXRT_NATIVE_D3D12_INT8_ACT_SCALE")
    if raw is not None:
        value = float(raw)
        if np.isfinite(value) and value > 0:
            return value
    return float(8.0 / 127.0)


def _pack_fp16_to_uint32(values: np.ndarray) -> np.ndarray:
    half = np.ascontiguousarray(values, dtype=np.float16).reshape(-1).view(np.uint16)
    if half.size % 2:
        half = np.concatenate([half, np.zeros((1,), dtype=np.uint16)])
    lo = half[0::2].astype(np.uint32)
    hi = half[1::2].astype(np.uint32) << np.uint32(16)
    return np.ascontiguousarray(lo | hi, dtype=np.uint32)


def _pack_int8_per_channel_1x1(weight: np.ndarray, activation_scale: float) -> tuple[np.ndarray, np.ndarray]:
    w = np.ascontiguousarray(weight, dtype=np.float32)
    if w.ndim != 4 or tuple(int(v) for v in w.shape[2:]) != (1, 1):
        raise ValueError("INT8 1x1 packing requires OIHW 1x1 weights")
    flat = w.reshape(w.shape[0], w.shape[1])
    max_abs = np.max(np.abs(flat), axis=1)
    weight_scale = np.maximum(max_abs / np.float32(127.0), np.float32(1.0e-8)).astype(np.float32)
    q = np.rint(flat / weight_scale[:, None])
    q = np.clip(q, -128, 127).astype(np.int8).reshape(-1)
    if q.size % 4:
        q = np.concatenate([q, np.zeros((4 - q.size % 4,), dtype=np.int8)])
    u = q.view(np.uint8).astype(np.uint32)
    packed = u[0::4] | (u[1::4] << np.uint32(8)) | (u[2::4] << np.uint32(16)) | (u[3::4] << np.uint32(24))
    product_scale = (weight_scale * np.float32(activation_scale)).astype(np.float32)
    return np.ascontiguousarray(packed, dtype=np.uint32), np.ascontiguousarray(product_scale, dtype=np.float32)


def _is_winograd_packable_desc(desc: Dict[str, Any]) -> bool:
    return (
        int(desc["kernel_h"]) == 3
        and int(desc["kernel_w"]) == 3
        and int(desc["stride_h"]) == 1
        and int(desc["stride_w"]) == 1
        and int(desc["pad_top"]) == 1
        and int(desc["pad_left"]) == 1
        and int(desc["dilation_h"]) == 1
        and int(desc["dilation_w"]) == 1
        and int(desc["groups"]) == 1
        and int(desc["in_h"]) == int(desc["out_h"])
        and int(desc["in_w"]) == int(desc["out_w"])
        and int(desc["out_h"]) % 2 == 0
        and int(desc["out_w"]) % 2 == 0
    )


def _is_sppf_pool(node) -> bool:
    return (
        node.op == "MaxPool"
        and tuple(int(v) for v in node.attrs.get("kernel_shape", [])) == (5, 5)
        and tuple(int(v) for v in node.attrs.get("strides", [1, 1])) == (1, 1)
        and tuple(int(v) for v in node.attrs.get("pads", [0, 0, 0, 0])) == (2, 2, 2, 2)
        and tuple(int(v) for v in node.attrs.get("dilations", [1, 1])) == (1, 1)
        and int(node.attrs.get("ceil_mode", 0)) == 0
    )


def _pack_winograd_f2x2_3x3_weights(weight: np.ndarray) -> np.ndarray:
    w = np.ascontiguousarray(weight, dtype=np.float32)
    if w.ndim != 4 or tuple(w.shape[2:]) != (3, 3):
        raise ValueError("Winograd F(2x2,3x3) packing requires OIHW 3x3 weights")
    out_channels, in_channels, _, _ = w.shape
    packed = np.empty((out_channels, in_channels, 4, 4), dtype=np.float32)
    g = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    for oc in range(out_channels):
        for ic in range(in_channels):
            packed[oc, ic] = g @ w[oc, ic] @ g.T
    return np.ascontiguousarray(packed, dtype=np.float32)


def _infer_yolo_head_layout(
    shape: Tuple[int, ...] | None,
    *,
    anchors: int | None,
    channels: int | None,
    classes: int | None,
) -> tuple[int, int, int]:
    if channels is not None:
        channels_i = int(channels)
        if anchors is None:
            if shape is None:
                raise ValueError("anchors is required when output shape is unknown")
            total = _numel(shape)
            if channels_i <= 0 or total % channels_i != 0:
                raise ValueError(f"cannot infer anchors from shape {shape} and channels={channels_i}")
            anchors_i = total // channels_i
        else:
            anchors_i = int(anchors)
    else:
        if shape is None:
            raise ValueError("channels is required when output shape is unknown")
        dims = tuple(int(x) for x in shape)
        if len(dims) == 2:
            channels_i, anchors_i = dims
        elif len(dims) == 3 and dims[0] == 1:
            channels_i, anchors_i = dims[1], dims[2]
        elif len(dims) == 4 and dims[0] == 1:
            channels_i = dims[1]
            anchors_i = dims[2] * dims[3]
        else:
            raise ValueError(
                "expected YOLO head output shaped [channels, anchors], [1, channels, anchors], "
                f"or [1, channels, height, width], got {shape}"
            )
        if anchors is not None and int(anchors) != anchors_i:
            raise ValueError(f"anchors={anchors} does not match inferred anchors={anchors_i} from shape {shape}")

    classes_i = int(classes) if classes is not None else channels_i - 4
    if anchors_i <= 0 or channels_i < 5 or classes_i <= 0 or channels_i < classes_i + 4:
        raise ValueError(
            f"invalid YOLO layout: anchors={anchors_i}, channels={channels_i}, classes={classes_i}"
        )
    return int(anchors_i), int(channels_i), int(classes_i)


def _match_conv_silu_chain(graph: Graph) -> Tuple[str, str, list[Dict[str, Any]]] | None:
    if len(graph.inputs) != 1 or len(graph.outputs) != 1 or not graph.nodes:
        return None
    input_name = next(iter(graph.inputs))
    current = input_name
    current_shape = tuple(int(x) for x in graph.inputs[input_name].shape)
    blocks: list[Dict[str, Any]] = []
    i = 0
    while i < len(graph.nodes):
        parsed = _parse_conv_silu_block(graph, i, current, current_shape)
        if parsed is None:
            return None
        block, next_i = parsed
        blocks.append(block)
        current = block["output_name"]
        current_shape = block["output_shape"]
        i = next_i
    if current != graph.outputs[0] or not blocks:
        return None
    return input_name, graph.outputs[0], blocks


def _parse_conv_silu_block(graph: Graph, index: int, input_name: str, input_shape: Tuple[int, ...]) -> Tuple[Dict[str, Any], int] | None:
    nodes = graph.nodes
    if index >= len(nodes):
        return None
    conv = nodes[index]
    if conv.op != "Conv" or len(conv.inputs) not in {2, 3} or len(conv.outputs) != 1 or conv.inputs[0] != input_name:
        return None
    conv_out = conv.outputs[0]
    activation_input = conv_out
    next_index = index + 1
    bn = None
    if next_index < len(nodes) and nodes[next_index].op == "BatchNormalization":
        candidate = nodes[next_index]
        if len(candidate.inputs) != 5 or len(candidate.outputs) != 1 or candidate.inputs[0] != conv_out:
            return None
        bn = candidate
        activation_input = bn.outputs[0]
        next_index += 1
    if next_index + 1 >= len(nodes):
        return None
    sigmoid = nodes[next_index]
    mul = nodes[next_index + 1]
    if sigmoid.op != "Sigmoid" or mul.op != "Mul" or len(sigmoid.inputs) != 1 or len(sigmoid.outputs) != 1 or len(mul.inputs) != 2 or len(mul.outputs) != 1:
        return None
    if sigmoid.inputs[0] != activation_input or set(mul.inputs) != {activation_input, sigmoid.outputs[0]}:
        return None

    block = _build_conv_silu_block(graph, conv, bn, input_shape, mul.outputs[0])
    if block is None:
        return None
    return block, next_index + 2


def _build_conv_silu_block(graph: Graph, conv, bn, input_shape: Tuple[int, ...], output_name: str) -> Dict[str, Any] | None:
    weight_name = conv.inputs[1]
    bias_name = conv.inputs[2] if len(conv.inputs) == 3 else None
    if weight_name not in graph.constants:
        return None
    weight = np.ascontiguousarray(graph.constants[weight_name], dtype=np.float32)
    if weight.ndim != 4 or len(input_shape) != 4:
        return None
    n, in_channels, in_h, in_w = (int(v) for v in input_shape)
    out_channels, kernel_ic, kernel_h, kernel_w = (int(v) for v in weight.shape)
    if bias_name is None:
        bias = np.zeros((out_channels,), dtype=np.float32)
    elif bias_name in graph.constants:
        bias = np.ascontiguousarray(graph.constants[bias_name], dtype=np.float32).reshape(-1)
    else:
        return None
    if bias.shape != (out_channels,):
        return None
    if bn is not None:
        folded = _fold_batchnorm(weight, bias, graph, bn)
        if folded is None:
            return None
        weight, bias = folded

    strides = tuple(int(v) for v in conv.attrs.get("strides", [1, 1]))
    pads = tuple(int(v) for v in conv.attrs.get("pads", [0, 0, 0, 0]))
    dilations = tuple(int(v) for v in conv.attrs.get("dilations", [1, 1]))
    group = int(conv.attrs.get("group", 1))
    if len(strides) != 2 or len(pads) != 4 or len(dilations) != 2 or group <= 0:
        return None
    if in_channels % group != 0 or kernel_ic != in_channels // group:
        return None
    stride_h, stride_w = strides
    pad_top, pad_left, pad_bottom, pad_right = pads
    dilation_h, dilation_w = dilations
    if min(stride_h, stride_w, dilation_h, dilation_w) <= 0:
        return None
    out_h = (in_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        return None
    output_shape = (n, out_channels, out_h, out_w)
    desc = {
        "batch": n,
        "in_channels": in_channels,
        "in_h": in_h,
        "in_w": in_w,
        "out_channels": out_channels,
        "out_h": out_h,
        "out_w": out_w,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "dilation_h": dilation_h,
        "dilation_w": dilation_w,
        "groups": group,
    }
    return {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "output_name": output_name,
        "weight": np.ascontiguousarray(weight, dtype=np.float32),
        "bias": np.ascontiguousarray(bias, dtype=np.float32),
        "desc": desc,
    }


def _fold_batchnorm(weight: np.ndarray, bias: np.ndarray, graph: Graph, bn) -> Tuple[np.ndarray, np.ndarray] | None:
    names = bn.inputs[1:5]
    if any(name not in graph.constants for name in names):
        return None
    scale, beta, mean, var = [np.asarray(graph.constants[name], dtype=np.float32).reshape(-1) for name in names]
    out_channels = weight.shape[0]
    if any(x.shape != (out_channels,) for x in (scale, beta, mean, var)):
        return None
    eps = float(bn.attrs.get("epsilon", 1e-5))
    alpha = scale / np.sqrt(var + eps)
    folded_weight = weight * alpha.reshape(-1, 1, 1, 1)
    folded_bias = (bias - mean) * alpha + beta
    return folded_weight.astype(np.float32, copy=False), folded_bias.astype(np.float32, copy=False)
