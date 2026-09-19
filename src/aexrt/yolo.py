from __future__ import annotations

import ast
import base64
from dataclasses import dataclass
import json
import os
from typing import Any, Sequence

import numpy as np

from .graph import Graph, TensorSpec


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    xyxy: tuple[float, float, float, float]


def _infer_yolo_output_layout(shape: Sequence[int], *, layout: str = "auto") -> tuple[str, int, int]:
    dims = tuple(int(x) for x in shape)
    if len(dims) != 2:
        raise ValueError(f"expected YOLO output shape without batch to be rank 2, got {dims}")
    layout_i = layout.lower()
    if layout_i in {"channels_first", "nchw"}:
        channels, anchors = dims
        return "channels_first", int(channels), int(anchors)
    if layout_i in {"channels_last", "nhwc", "anchors_first"}:
        anchors, channels = dims
        return "channels_last", int(channels), int(anchors)
    if layout_i != "auto":
        raise ValueError(f"unknown YOLO layout {layout!r}")
    a, b = dims
    if a <= 4096 and (b > a or b >= 64):
        return "channels_first", int(a), int(b)
    if b <= 4096 and a >= 64 and a > b:
        return "channels_last", int(b), int(a)
    return "channels_first", int(a), int(b)


def _metadata_class_count(metadata: dict[str, str] | None) -> int | None:
    if not metadata:
        return None
    lowered = {str(key).lower(): str(value) for key, value in metadata.items()}
    for key in ("nc", "classes", "class_count"):
        value = lowered.get(key)
        if value is None:
            continue
        try:
            count = int(value.strip())
        except ValueError:
            continue
        if 0 < count <= 4096:
            return count
    for key in ("names", "class_names"):
        value = lowered.get(key)
        if value is None:
            continue
        try:
            names = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            continue
        if isinstance(names, (dict, list, tuple)) and 0 < len(names) <= 4096:
            return len(names)
    return None


def _resolve_yolo_classes_objectness(
    channels: int,
    *,
    classes: int | None,
    objectness: bool | None,
    metadata: dict[str, str] | None = None,
) -> tuple[int, bool]:
    known_classes = int(classes) if classes is not None else _metadata_class_count(metadata)
    if objectness is None and known_classes is not None:
        if channels == known_classes + 5:
            return known_classes, True
        if channels == known_classes + 4:
            return known_classes, False
        raise ValueError(f"YOLO channels={channels} do not match classes={known_classes} with or without objectness")

    has_objectness = bool(objectness) if objectness is not None else False
    classes_i = known_classes if known_classes is not None else channels - (5 if has_objectness else 4)
    if classes_i <= 0 or classes_i + (5 if has_objectness else 4) > channels:
        raise ValueError(f"invalid YOLO classes/channels: classes={classes_i}, channels={channels}, objectness={has_objectness}")
    return classes_i, has_objectness


def yolo_postprocess(
    output: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    *,
    layout: str = "auto",
    objectness: bool | None = None,
) -> list[Detection]:
    y = np.asarray(output)
    if y.ndim == 3:
        y = y[0]
    if y.ndim != 2:
        raise ValueError(f"expected YOLO output rank 2/3, got {output.shape}")
    layout_i, channels, anchors = _infer_yolo_output_layout(y.shape, layout=layout)
    if channels < 5:
        raise ValueError(f"expected YOLO channel count >= 5, got {channels}")
    if layout_i == "channels_last":
        y = y.T

    has_objectness = bool(objectness) if objectness is not None else False
    cls_offset = 5 if has_objectness else 4
    if channels <= cls_offset:
        raise ValueError(f"invalid YOLO channels={channels} for objectness={has_objectness}")

    boxes_xywh = y[:4].T.astype(np.float32, copy=False)
    scores = y[cls_offset:].T.astype(np.float32, copy=False)
    if has_objectness:
        scores = scores * y[4:5].T.astype(np.float32, copy=False)
    class_ids = np.argmax(scores, axis=1)
    conf = scores[np.arange(scores.shape[0]), class_ids]
    keep = conf >= float(conf_threshold)
    if not np.any(keep):
        return []

    boxes = _xywh_to_xyxy(boxes_xywh[keep])
    conf = conf[keep]
    class_ids = class_ids[keep]

    detections: list[Detection] = []
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        selected = _nms(boxes[idx], conf[idx], iou_threshold)
        for local in selected:
            original = idx[local]
            detections.append(Detection(int(cls), float(conf[original]), tuple(float(x) for x in boxes[original])))
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections


def build_yolo_output0_package(
    output_shape: Sequence[int],
    *,
    source_model: str | None = None,
    output_name: str = "output0",
    classes: int | None = None,
    layout: str = "auto",
    objectness: bool | None = None,
    max_candidates: int = 512,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> dict[str, object]:
    shape = tuple(int(x) for x in output_shape)
    if len(shape) != 3:
        raise ValueError(f"expected YOLO output rank 3, got {shape}")
    layout_i, channels, anchors = _infer_yolo_output_layout(shape[1:], layout=layout)
    inferred_classes, has_objectness = _resolve_yolo_classes_objectness(
        channels,
        classes=classes,
        objectness=objectness,
    )

    package: dict[str, object] = {
        "format": "aexrt.yolo.package",
        "version": 1,
        "mode": "output0_postprocess",
        "source_model": os.path.abspath(source_model) if source_model else "",
        "output_name": output_name,
        "layout": layout_i,
        "objectness": has_objectness,
        "channels": int(channels),
        "anchors": int(anchors),
        "classes": inferred_classes,
        "max_candidates": int(max_candidates),
        "conf_threshold": float(conf_threshold),
        "iou_threshold": float(iou_threshold),
        "next_mode": "native_d3d12_graph",
    }
    return package


def build_yolo_native_d3d12_graph_package(
    graph: Graph,
    *,
    source_model: str | None = None,
    output_name: str | None = None,
    classes: int | None = None,
    anchors: int | None = None,
    channels: int | None = None,
    layout: str = "auto",
    objectness: bool | None = None,
    max_candidates: int = 512,
    max_detections: int = 100,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    embed_constants: bool = True,
) -> dict[str, object]:
    from .execution import infer_value_specs

    if output_name is None:
        if len(graph.outputs) != 1:
            raise ValueError("output_name is required when the graph has multiple outputs")
        output_name = graph.outputs[0]
    value_specs = infer_value_specs(graph)
    output_spec = value_specs.get(output_name)
    if output_spec is None or len(output_spec.shape) != 3:
        raise ValueError(f"expected YOLO output {output_name!r} to be rank 3")
    inferred_layout, inferred_channels, inferred_anchors = _infer_yolo_output_layout(output_spec.shape[1:], layout=layout)
    channels_i = int(channels) if channels is not None else inferred_channels
    anchors_i = int(anchors) if anchors is not None else inferred_anchors
    classes_i, has_objectness = _resolve_yolo_classes_objectness(
        channels_i,
        classes=classes,
        objectness=objectness,
        metadata=graph.metadata,
    )

    constants = {
        name: _constant_payload(value) if embed_constants else _constant_metadata(value)
        for name, value in graph.constants.items()
    }
    resource_table = _build_resource_table(graph, value_specs)
    prepared_commands = _build_prepared_command_stream(graph, value_specs, output_name=output_name, classes=classes_i)
    cxx_kernel_coverage = _cxx_kernel_coverage(prepared_commands)
    cxx_replay_text = _build_cxx_replay_text(graph, value_specs, prepared_commands, output_name=output_name)
    input_element_count = sum(_numel(spec.shape) for spec in graph.inputs.values())

    return {
        "format": "aexrt.yolo.package",
        "version": 2,
        "mode": "native_d3d12_graph",
        "source_model": os.path.abspath(source_model) if source_model else "",
        "runtime_target": "aexrt_native_d3d12",
        "graph": {
            "name": graph.name,
            "inputs": {name: spec.to_json() for name, spec in graph.inputs.items()},
            "outputs": list(graph.outputs),
            "output_name": output_name,
            "nodes": [node.to_json() for node in graph.nodes],
            "value_specs": {name: spec.to_json() for name, spec in value_specs.items()},
        },
        "constants": constants,
        "resource_table": resource_table,
        "cxx_replay_text": cxx_replay_text,
        "prepared_graph": {
            "kind": "prepared_yolo_graph",
            "command_replay": True,
            "descriptor_arena": {
                "strategy": "packed_static_cbv_srv_uav",
                "resource_count": len(resource_table),
            },
            "upload": {
                "strategy": "persistent_mapped_upload_ring",
                "ring_size": 2,
                "input_element_count": int(input_element_count),
            },
            "commands": prepared_commands,
            "command_count": len(prepared_commands),
        },
        "yolo": {
            "output_name": output_name,
            "layout": inferred_layout,
            "objectness": has_objectness,
            "channels": channels_i,
            "anchors": anchors_i,
            "classes": classes_i,
            "max_candidates": int(max_candidates),
            "max_detections": int(max_detections),
            "conf_threshold": float(conf_threshold),
            "iou_threshold": float(iou_threshold),
            "postprocess": "gpu_decode_classwise_nms_topk",
        },
        "compiler": {
            "lowering": "python_native_d3d12_prepared_graph_v1",
            "cxx_loader": "native_d3d12_graph_metadata_v1",
            "cxx_kernel_covered": bool(cxx_kernel_coverage["unsupported_command_count"] == 0),
            "executable_in_cxx": bool(cxx_kernel_coverage["unsupported_command_count"] == 0),
            "cxx_kernel_coverage": cxx_kernel_coverage,
            "next_step": "port Python native_d3d12 dispatchers into pure C++ PreparedYoloGraph",
        },
    }


def export_yolo_package_from_onnx(
    model_path: str,
    *,
    classes: int | None = None,
    layout: str = "auto",
    objectness: bool | None = None,
    max_candidates: int = 512,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> dict[str, object]:
    from .execution import infer_value_specs
    from .onnx_importer import load_onnx

    graph = load_onnx(model_path)
    output_name = graph.outputs[0]
    spec = graph.value_specs.get(output_name)
    if spec is None:
        spec = infer_value_specs(graph)[output_name]
    _, channels, _ = _infer_yolo_output_layout(spec.shape[1:], layout=layout)
    resolved_classes, resolved_objectness = _resolve_yolo_classes_objectness(
        channels,
        classes=classes,
        objectness=objectness,
        metadata=graph.metadata,
    )
    return build_yolo_output0_package(
        spec.shape,
        source_model=model_path,
        output_name=output_name,
        classes=resolved_classes,
        layout=layout,
        objectness=resolved_objectness,
        max_candidates=max_candidates,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


def export_yolo_native_d3d12_graph_package_from_onnx(
    model_path: str,
    *,
    output_name: str | None = None,
    classes: int | None = None,
    layout: str = "auto",
    objectness: bool | None = None,
    max_candidates: int = 512,
    max_detections: int = 100,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    embed_constants: bool = True,
) -> dict[str, object]:
    from .onnx_importer import load_onnx

    graph = load_onnx(model_path)
    return build_yolo_native_d3d12_graph_package(
        graph,
        source_model=model_path,
        output_name=output_name,
        classes=classes,
        layout=layout,
        objectness=objectness,
        max_candidates=max_candidates,
        max_detections=max_detections,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        embed_constants=embed_constants,
    )


def save_yolo_package(package: dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(package, f, indent=2)


def save_yolo_package_from_onnx(
    model_path: str,
    path: str,
    *,
    classes: int | None = None,
    layout: str = "auto",
    objectness: bool | None = None,
    max_candidates: int = 512,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> dict[str, object]:
    package = export_yolo_package_from_onnx(
        model_path,
        classes=classes,
        layout=layout,
        objectness=objectness,
        max_candidates=max_candidates,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    save_yolo_package(package, path)
    return package


def save_yolo_native_d3d12_graph_package_from_onnx(
    model_path: str,
    path: str,
    *,
    output_name: str | None = None,
    classes: int | None = None,
    layout: str = "auto",
    objectness: bool | None = None,
    max_candidates: int = 512,
    max_detections: int = 100,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    embed_constants: bool = True,
) -> dict[str, object]:
    package = export_yolo_native_d3d12_graph_package_from_onnx(
        model_path,
        output_name=output_name,
        classes=classes,
        layout=layout,
        objectness=objectness,
        max_candidates=max_candidates,
        max_detections=max_detections,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        embed_constants=embed_constants,
    )
    save_yolo_package(package, path)
    return package


def _numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _constant_metadata(value: Any) -> dict[str, object]:
    arr = np.ascontiguousarray(value)
    return {
        "dtype": str(arr.dtype),
        "shape": [int(v) for v in arr.shape],
        "nbytes": int(arr.nbytes),
        "encoding": "external_or_omitted",
    }


def _constant_payload(value: Any) -> dict[str, object]:
    arr = np.ascontiguousarray(value)
    payload = _constant_metadata(arr)
    payload["encoding"] = "base64"
    payload["data"] = base64.b64encode(arr.tobytes(order="C")).decode("ascii")
    return payload


def _build_resource_table(graph: Graph, value_specs: dict[str, TensorSpec]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    produced = {out for node in graph.nodes for out in node.outputs}
    for name, spec in value_specs.items():
        if name in graph.inputs:
            kind = "input"
        elif name in graph.constants:
            kind = "constant"
        elif name in graph.outputs:
            kind = "output"
        elif name in produced:
            kind = "temporary"
        else:
            kind = "value"
        rows.append({
            "name": name,
            "kind": kind,
            "shape": [int(v) for v in spec.shape],
            "dtype": spec.dtype,
            "nbytes": int(_numel(spec.shape) * np.dtype(spec.dtype).itemsize),
            "arena": "default" if kind not in {"constant", "input"} else kind,
        })
    rows.sort(key=lambda row: (str(row["kind"]), str(row["name"])))
    return rows


def _build_prepared_command_stream(
    graph: Graph,
    value_specs: dict[str, TensorSpec],
    *,
    output_name: str,
    classes: int,
) -> list[dict[str, object]]:
    commands: list[dict[str, object]] = []
    silu_patterns = _find_conv_silu_patterns(graph)
    concat_patterns = _find_concat_conv1x1_patterns(graph, silu_patterns)
    # Build switch: unfuse concat + conv1x1 for SMALL spatial outputs so the
    # int8 algo map can claim the conv (the fp16 fused kernel bypasses
    # quantization). Large-spatial patterns stay fused (their separate fp16
    # conv fallback is slower than the fused kernel).
    import os as _os
    _uf = _os.environ.get("AEXRTC_UNFUSE_SMALL_CONCAT", "")
    if _uf:
        threshold = 44 if _uf == "2" else 20
        kept = {}
        for pat_i, pat in concat_patterns.items():
            spec = value_specs.get(str(pat["output"]))
            h = int(_shape4(spec.shape)[2]) if spec is not None else 0
            if h > threshold:
                kept[pat_i] = pat
        concat_patterns = kept
    skip_indices = {idx for pattern in silu_patterns.values() for idx in pattern}
    for pattern in concat_patterns.values():
        skip_indices.update(pattern["skip"])
    i = 0
    while i < len(graph.nodes):
        if i in skip_indices:
            i += 1
            continue
        node = graph.nodes[i]
        concat_pattern = concat_patterns.get(i)
        if concat_pattern is not None:
            conv = graph.nodes[int(concat_pattern["conv_index"])]
            commands.append(_command(
                "concat_conv1x1",
                conv,
                value_specs,
                outputs=[str(concat_pattern["output"])],
                concat_inputs=list(node.inputs),
                concat_node=node.name,
                activation=str(concat_pattern["activation"]),
                fused_nodes=list(concat_pattern["fused_nodes"]),
            ))
            i = int(concat_pattern["end"])
            continue
        silu_pattern = silu_patterns.get(i)
        if silu_pattern is not None:
            _, mul_index = silu_pattern
            mul = graph.nodes[mul_index]
            commands.append(_command(
                "conv2d_silu",
                node,
                value_specs,
                outputs=list(mul.outputs),
                fused_nodes=[node.name, graph.nodes[silu_pattern[0]].name, mul.name],
            ))
        elif node.op == "Split" and _is_channel_split_view(node, value_specs):
            commands.append(_command("channel_split_view", node, value_specs, materialized=False))
        elif node.outputs and node.outputs[0] == output_name:
            commands.append(_command("yolo_output", node, value_specs, classes=int(classes)))
        else:
            commands.append(_command(node.op.lower(), node, value_specs))
        i += 1
    commands.append({
        "kernel": "yolo_decode_nms_topk",
        "inputs": [output_name],
        "outputs": ["detections", "detection_count"],
        "classes": int(classes),
    })
    return commands


def _command(kernel: str, node, value_specs: dict[str, TensorSpec], outputs: list[str] | None = None, **extra: object) -> dict[str, object]:
    command_outputs = outputs if outputs is not None else list(node.outputs)
    out_shapes = {
        name: [int(v) for v in value_specs[name].shape]
        for name in command_outputs
        if name in value_specs
    }
    data: dict[str, object] = {
        "kernel": kernel,
        "node": node.name,
        "op": node.op,
        "inputs": list(node.inputs),
        "outputs": command_outputs,
        "attrs": dict(node.attrs),
        "output_shapes": out_shapes,
    }
    data.update(extra)
    return data


def _match_conv_silu(graph: Graph, index: int) -> int | None:
    if index + 2 >= len(graph.nodes):
        return None
    conv, sigmoid, mul = graph.nodes[index:index + 3]
    if conv.op != "Conv" or sigmoid.op != "Sigmoid" or mul.op != "Mul":
        return None
    if len(conv.outputs) != 1 or len(sigmoid.outputs) != 1 or len(mul.outputs) != 1:
        return None
    conv_out = conv.outputs[0]
    if sigmoid.inputs != [conv_out] or set(mul.inputs) != {conv_out, sigmoid.outputs[0]}:
        return None
    return index + 3


def _is_channel_split_view(node, value_specs: dict[str, TensorSpec]) -> bool:
    if node.op != "Split" or not node.inputs:
        return False
    src = value_specs.get(node.inputs[0])
    if src is None or len(src.shape) < 2 or int(src.shape[0]) != 1:
        return False
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += len(src.shape)
    return axis == 1


def _is_concat_1x1_conv(graph: Graph, index: int) -> bool:
    if index + 1 >= len(graph.nodes):
        return False
    concat, conv = graph.nodes[index], graph.nodes[index + 1]
    if concat.op != "Concat" or conv.op != "Conv" or not concat.outputs or not conv.inputs:
        return False
    if conv.inputs[0] != concat.outputs[0]:
        return False
    weight_name = conv.inputs[1] if len(conv.inputs) > 1 else ""
    weight = graph.constants.get(weight_name)
    return weight is not None and np.asarray(weight).ndim == 4 and tuple(np.asarray(weight).shape[2:]) == (1, 1)


def _find_conv_silu_patterns(graph: Graph) -> dict[int, tuple[int, int]]:
    uses: dict[str, list[int]] = {}
    for index, node in enumerate(graph.nodes):
        for name in node.inputs:
            uses.setdefault(name, []).append(index)
    patterns: dict[int, tuple[int, int]] = {}
    for index, conv in enumerate(graph.nodes):
        if conv.op != "Conv" or len(conv.outputs) != 1:
            continue
        conv_out = conv.outputs[0]
        if len(uses.get(conv_out, [])) != 2:
            continue
        sigmoid_indices = [i for i in uses.get(conv_out, []) if graph.nodes[i].op == "Sigmoid" and len(graph.nodes[i].outputs) == 1]
        if not sigmoid_indices:
            continue
        sigmoid_index = sigmoid_indices[0]
        sigmoid_out = graph.nodes[sigmoid_index].outputs[0]
        for mul_index in uses.get(sigmoid_out, []):
            mul = graph.nodes[mul_index]
            if mul.op == "Mul" and len(mul.outputs) == 1 and set(mul.inputs) == {conv_out, sigmoid_out}:
                patterns[index] = (sigmoid_index, mul_index)
                break
    return patterns


def _find_concat_conv1x1_patterns(graph: Graph, silu_patterns: dict[int, tuple[int, int]]) -> dict[int, dict[str, object]]:
    uses: dict[str, list[int]] = {}
    for node_index, node in enumerate(graph.nodes):
        for name in node.inputs:
            uses.setdefault(name, []).append(node_index)
    patterns: dict[int, dict[str, object]] = {}
    for index, concat in enumerate(graph.nodes):
        if index + 1 >= len(graph.nodes) or concat.op != "Concat" or len(concat.outputs) != 1:
            continue
        if len(uses.get(concat.outputs[0], [])) != 1:
            continue
        conv_index = index + 1
        conv = graph.nodes[conv_index]
        if conv.op != "Conv" or not conv.inputs or conv.inputs[0] != concat.outputs[0] or not _is_concat_1x1_conv(graph, index):
            continue
        activation = "linear"
        output = conv.outputs[0]
        end = conv_index + 1
        skip = {conv_index}
        fused_nodes = [concat.name, conv.name]
        if conv_index in silu_patterns:
            sigmoid_index, mul_index = silu_patterns[conv_index]
            activation = "silu"
            output = graph.nodes[mul_index].outputs[0]
            end = max(end, sigmoid_index + 1, mul_index + 1)
            skip.update({sigmoid_index, mul_index})
            fused_nodes.extend([graph.nodes[sigmoid_index].name, graph.nodes[mul_index].name])
        patterns[index] = {
            "conv_index": conv_index,
            "activation": activation,
            "output": output,
            "end": end,
            "skip": skip,
            "fused_nodes": fused_nodes,
        }
    return patterns


def _cxx_kernel_coverage(commands: list[dict[str, object]]) -> dict[str, object]:
    supported = {
        "add",
        "channel_split_view",
        "concat",
        "conv",
        "conv2d_silu",
        "concat_conv1x1",
        "div",
        "maxpool",
        "matmul",
        "mul",
        "pow",
        "reshape",
        "resize",
        "sigmoid",
        "slice",
        "softmax",
        "split",
        "sub",
        "transpose",
        "yolo_output",
        "yolo_decode_nms_topk",
    }
    counts: dict[str, int] = {}
    for command in commands:
        kernel = str(command["kernel"])
        counts[kernel] = counts.get(kernel, 0) + 1
    supported_count = sum(count for kernel, count in counts.items() if kernel in supported)
    unsupported = {kernel: count for kernel, count in counts.items() if kernel not in supported}
    return {
        "supported_kernels": sorted(supported),
        "required_kernel_counts": dict(sorted(counts.items())),
        "unsupported_kernel_counts": dict(sorted(unsupported.items())),
        "supported_command_count": int(supported_count),
        "unsupported_command_count": int(len(commands) - supported_count),
    }


def _build_cxx_replay_text(
    graph: Graph,
    value_specs: dict[str, TensorSpec],
    commands: list[dict[str, object]],
    *,
    output_name: str,
) -> str:
    local_specs = dict(value_specs)
    generated_constants: dict[str, np.ndarray] = {}
    for index, command in enumerate(commands):
        if str(command.get("kernel")) not in {"conv", "conv2d_silu"}:
            continue
        inputs = [str(x) for x in command.get("inputs", [])]
        outputs = [str(x) for x in command.get("outputs", [])]
        if len(inputs) == 2 and outputs:
            out_channels = int(local_specs[outputs[0]].shape[1])
            bias_name = f"__aexrt_zero_bias_{index}_{out_channels}"
            generated_constants[bias_name] = np.zeros((out_channels,), dtype=np.float32)
            local_specs[bias_name] = TensorSpec((out_channels,), "float32")
            command["inputs"] = [*inputs, bias_name]

    names: list[str] = []
    seen: set[str] = set()

    def add_name(name: str) -> None:
        if name not in seen and name in local_specs:
            seen.add(name)
            names.append(name)

    for name in graph.inputs:
        add_name(name)
    for name in generated_constants:
        add_name(name)
    for command in commands:
        for name in command.get("inputs", []):
            add_name(str(name))
        for name in command.get("outputs", []):
            add_name(str(name))
        for name in command.get("concat_inputs", []):
            add_name(str(name))
    value_id = {name: i for i, name in enumerate(names)}
    lines = ["AEXRT_CXX_REPLAY_V1"]
    for name in names:
        spec = local_specs[name]
        shape = _shape4(spec.shape)
        elements = _numel(spec.shape)
        if name in graph.inputs:
            lines.append(f"INPUT|{value_id[name]}|{_escape_plan_name(name)}|{elements}|{_csv(shape)}")
        elif name in graph.constants or name in generated_constants:
            source = graph.constants[name] if name in graph.constants else generated_constants[name]
            arr = np.ascontiguousarray(source, dtype=np.float32)
            data = base64.b64encode(arr.tobytes(order="C")).decode("ascii")
            lines.append(f"CONST|{value_id[name]}|{_escape_plan_name(name)}|{arr.size}|{_csv(_shape4(arr.shape))}|{data}")
        else:
            lines.append(f"VALUE|{value_id[name]}|{_escape_plan_name(name)}|{elements}|{_csv(shape)}")

    for command in commands:
        kernel = str(command["kernel"])
        if kernel == "yolo_decode_nms_topk":
            continue
        inputs = [str(x) for x in command.get("inputs", [])]
        outputs = [str(x) for x in command.get("outputs", [])]
        if not outputs:
            continue
        out = outputs[0]
        if kernel in {"conv", "conv2d_silu"}:
            x, weight, bias = inputs[:3]
            desc = _conv_desc_from_specs(local_specs[x], local_specs[weight], local_specs[out], command.get("attrs", {}))
            kind = "CONV_SILU" if kernel == "conv2d_silu" else "CONV"
            lines.append(f"CMD|{kind}|{value_id[out]}|{_csv([value_id[x], value_id[weight], value_id[bias]])}|{_csv(desc)}")
        elif kernel == "concat_conv1x1":
            concat_inputs = [str(x) for x in command.get("concat_inputs", [])]
            weight = inputs[1]
            bias = inputs[2]
            out_shape = local_specs[out].shape
            channels = [int(local_specs[name].shape[1]) for name in concat_inputs]
            params = [int(out_shape[0]), int(out_shape[2]), int(out_shape[3]), int(out_shape[1]), 1 if command.get("activation") == "silu" else 0, *channels]
            ids = [value_id[name] for name in concat_inputs] + [value_id[weight], value_id[bias]]
            lines.append(f"CMD|CONCAT_CONV1X1|{value_id[out]}|{_csv(ids)}|{_csv(params)}")
        elif kernel == "channel_split_view":
            src = inputs[0]
            src_shape = local_specs[src].shape
            trailing = _numel(src_shape[2:])
            offset = 0
            for split_out in outputs:
                channels = int(local_specs[split_out].shape[1])
                lines.append(f"CMD|VIEW|{value_id[split_out]}|{value_id[src]}|{offset * trailing},{channels * trailing}")
                offset += channels
        elif kernel == "split":
            src = inputs[0]
            src_shape = local_specs[src].shape
            rank = len(src_shape)
            if rank <= 0 or rank > 5:
                raise NotImplementedError("C++ replay Split supports rank <= 4 or rank-5 with batch=1")
            attrs = command.get("attrs", {})
            axis = int(attrs.get("axis", 0))
            if axis < 0:
                axis += rank
            replay_rank = rank
            replay_axis = axis
            if rank == 5:
                if int(src_shape[0]) != 1 or axis == 0:
                    raise NotImplementedError("C++ replay rank-5 Split requires batch=1 and a non-batch axis")
                replay_rank = 4
                replay_axis = axis - 1
            split_sizes = [int(x) for x in attrs.get("split", [])]
            if not split_sizes:
                split_sizes = [int(local_specs[name].shape[axis]) for name in outputs]
            offset = 0
            for split_out, split_size in zip(outputs, split_sizes):
                starts4 = [0, 0, 0, 0]
                steps4 = [1, 1, 1, 1]
                starts4[replay_axis + (4 - replay_rank)] = offset
                params = [_numel(local_specs[split_out].shape), replay_rank, 0, 0, *_shape4(src_shape), *_shape4(local_specs[split_out].shape), *starts4, *steps4]
                lines.append(f"CMD|SLICE|{value_id[split_out]}|{value_id[src]}|{_csv(params)}")
                offset += split_size
        elif kernel == "reshape":
            src = inputs[0]
            lines.append(f"CMD|ALIAS|{value_id[out]}|{value_id[src]}|0,{_numel(local_specs[out].shape)}")
        elif kernel == "yolo_output":
            ids = [value_id[name] for name in inputs]
            axis = int(command.get("attrs", {}).get("axis", 1))
            params = _concat_constants([local_specs[name] for name in inputs], local_specs[out], axis)
            lines.append(f"CMD|CONCAT|{value_id[out]}|{_csv(ids)}|{_csv(params)}")
        elif kernel == "concat":
            axis = int(command.get("attrs", {}).get("axis", 0))
            ids = [value_id[name] for name in inputs]
            params = _concat_constants([local_specs[name] for name in inputs], local_specs[out], axis)
            lines.append(f"CMD|CONCAT|{value_id[out]}|{_csv(ids)}|{_csv(params)}")
        elif kernel == "resize":
            src = inputs[0]
            n, c, ih, iw = (int(v) for v in local_specs[src].shape)
            _, _, oh, ow = (int(v) for v in local_specs[out].shape)
            lines.append(f"CMD|RESIZE|{value_id[out]}|{value_id[src]}|{_csv([n, c, ih, iw, oh, ow])}")
        elif kernel == "maxpool":
            src = inputs[0]
            n, c, ih, iw = (int(v) for v in local_specs[src].shape)
            _, _, oh, ow = (int(v) for v in local_specs[out].shape)
            attrs = command.get("attrs", {})
            k = attrs.get("kernel_shape", [1, 1])
            s = attrs.get("strides", k)
            p = attrs.get("pads", [0, 0, 0, 0])
            d = attrs.get("dilations", [1, 1])
            lines.append(f"CMD|MAXPOOL|{value_id[out]}|{value_id[src]}|{_csv([n, c, ih, iw, oh, ow, int(k[0]), int(k[1]), int(s[0]), int(s[1]), int(p[0]), int(p[1]), int(d[0])])}")
        elif kernel in {"sigmoid", "tanh", "gelu", "relu"}:
            op = {"relu": 1, "sigmoid": 2, "tanh": 3, "gelu": 4}[kernel]
            src = inputs[0]
            lines.append(f"CMD|UNARY|{value_id[out]}|{value_id[src]}|{op},{_numel(local_specs[out].shape)}")
        elif kernel in {"add", "sub", "mul", "div", "pow"}:
            op = {"add": 0, "sub": 1, "mul": 2, "div": 3, "pow": 4}[kernel]
            a, b = inputs[:2]
            params = _binary_constants(op, local_specs[a], local_specs[b], local_specs[out])
            lines.append(f"CMD|BINARY|{value_id[out]}|{_csv([value_id[a], value_id[b]])}|{_csv(params)}")
        elif kernel == "matmul":
            a, b = inputs[:2]
            a4 = _shape4(local_specs[a].shape)
            b4 = _shape4(local_specs[b].shape)
            out4 = _shape4(local_specs[out].shape)
            if a4[3] != b4[2] or out4[2] != a4[2] or out4[3] != b4[3]:
                raise ValueError(f"unsupported MatMul shapes: {local_specs[a].shape} x {local_specs[b].shape} -> {local_specs[out].shape}")
            params = [_numel(local_specs[out].shape), a4[3], 0, 0, *out4, *a4, *b4, 0]
            lines.append(f"CMD|MATMUL|{value_id[out]}|{_csv([value_id[a], value_id[b]])}|{_csv(params)}")
        elif kernel == "slice":
            src = inputs[0]
            attrs = command.get("attrs", {})
            rank = len(local_specs[src].shape)
            if rank <= 0 or rank > 4:
                raise NotImplementedError("C++ replay Slice supports rank <= 4")
            starts_raw = attrs.get("starts") or [0]
            starts = [int(x) for x in starts_raw]
            axes = [int(x) for x in (attrs.get("axes") or list(range(len(starts))))]
            steps = [int(x) for x in (attrs.get("steps") or [1] * len(starts))]
            if len(axes) < len(starts) or len(steps) < len(starts):
                raise ValueError("Slice starts/axes/steps length mismatch")
            starts4 = [0, 0, 0, 0]
            steps4 = [1, 1, 1, 1]
            for start, axis, step in zip(starts, axes, steps):
                if axis < 0:
                    axis += rank
                if axis < 0 or axis >= rank:
                    raise ValueError(f"invalid Slice axis {axis} for rank {rank}")
                dim = int(local_specs[src].shape[axis])
                if start < 0:
                    start += dim
                start = max(0, min(start, dim))
                if step <= 0:
                    raise NotImplementedError("C++ replay Slice currently supports positive steps")
                padded_axis = axis + (4 - rank)
                starts4[padded_axis] = int(start)
                steps4[padded_axis] = int(step)
            params = [_numel(local_specs[out].shape), rank, 0, 0, *_shape4(local_specs[src].shape), *_shape4(local_specs[out].shape), *starts4, *steps4]
            lines.append(f"CMD|SLICE|{value_id[out]}|{value_id[src]}|{_csv(params)}")
        elif kernel == "transpose":
            src = inputs[0]
            axes = [int(x) for x in command.get("attrs", {}).get("axes", [])]
            replay_rank = len(local_specs[src].shape)
            if replay_rank == 5:
                if int(local_specs[src].shape[0]) != 1 or len(axes) != 5 or axes[0] != 0:
                    raise NotImplementedError("C++ replay rank-5 Transpose requires batch=1 and a fixed leading batch axis")
                replay_rank = 4
                axes = [axis - 1 for axis in axes[1:]]
                if any(axis < 0 or axis >= 4 for axis in axes):
                    raise ValueError("invalid rank-5 Transpose axes after dropping the batch dimension")
            elif replay_rank > 4:
                raise NotImplementedError("C++ replay Transpose supports rank <= 4 or rank-5 with batch=1")
            axes4 = axes + list(range(len(axes), 4))
            params = [_numel(local_specs[out].shape), replay_rank, 0, 0, *_shape4(local_specs[src].shape), *_shape4(local_specs[out].shape), *axes4, 0]
            lines.append(f"CMD|TRANSPOSE|{value_id[out]}|{value_id[src]}|{_csv(params)}")
        elif kernel == "softmax":
            src = inputs[0]
            n, c, h, w = (int(v) for v in local_specs[src].shape)
            axis = int(command.get("attrs", {}).get("axis", 1))
            if axis < 0:
                axis += 4
            groups = c * h * w
            if axis == 1:
                groups = n * h * w
            elif axis == 2:
                groups = n * c * w
            elif axis == 3:
                groups = n * c * h
            lines.append(f"CMD|SOFTMAX|{value_id[out]}|{value_id[src]}|{_csv([n, c, h, w, axis, groups])}")
        else:
            raise NotImplementedError(f"cannot emit C++ replay command for kernel {kernel}")
    lines.append(f"OUTPUT|{value_id[output_name]}")
    return "\n".join(lines)


def _conv_desc_from_specs(input_spec: TensorSpec, weight_spec: TensorSpec, output_spec: TensorSpec, attrs: dict[str, Any]) -> list[int]:
    pads = [int(x) for x in attrs.get("pads", [0, 0, 0, 0])]
    strides = [int(x) for x in attrs.get("strides", [1, 1])]
    dilations = [int(x) for x in attrs.get("dilations", [1, 1])]
    return [
        int(input_spec.shape[0]), int(input_spec.shape[1]), int(input_spec.shape[2]), int(input_spec.shape[3]),
        int(output_spec.shape[1]), int(output_spec.shape[2]), int(output_spec.shape[3]),
        int(weight_spec.shape[2]), int(weight_spec.shape[3]),
        strides[0], strides[1], pads[0], pads[1], dilations[0], dilations[1], int(attrs.get("group", 1)),
    ]


def _concat_constants(input_specs: list[TensorSpec], output_spec: TensorSpec, axis: int) -> list[int]:
    rank = len(output_spec.shape)
    if axis < 0:
        axis += rank
    padded_axis = axis + (4 - rank)
    sizes = [int(spec.shape[axis]) for spec in input_specs]
    sizes += [sizes[-1]] * (8 - len(sizes))
    return [_numel(output_spec.shape), rank, padded_axis, len(input_specs), *_shape4(output_spec.shape), *sizes, 0, 0]


def _binary_constants(op: int, a: TensorSpec, b: TensorSpec, out: TensorSpec) -> list[int]:
    out4 = _shape4(out.shape)
    a4 = _shape4(a.shape)
    b4 = _shape4(b.shape)
    astr = _strides4(a.shape)
    bstr = _strides4(b.shape)
    return [_numel(out.shape), op, len(out.shape), 0, *out4, *a4, *b4, *astr, *bstr]


def _shape4(shape: Sequence[int]) -> list[int]:
    shape_i = [int(x) for x in shape]
    if len(shape_i) == 5:
        if shape_i[0] != 1:
            raise NotImplementedError("C++ replay rank-5 tensors require a leading batch dimension of 1")
        shape_i = shape_i[1:]
    if len(shape_i) > 4:
        raise NotImplementedError("C++ replay tensors support rank <= 4 or rank-5 with batch=1")
    return [1] * (4 - len(shape_i)) + shape_i


def _strides4(shape: Sequence[int]) -> list[int]:
    padded = _shape4(shape)
    strides = [1, 1, 1, 1]
    for i in range(2, -1, -1):
        strides[i] = strides[i + 1] * padded[i + 1]
    return strides


def _csv(values: Sequence[int]) -> str:
    return ",".join(str(int(v)) for v in values)


def _escape_plan_name(name: str) -> str:
    return name.replace("|", "%7C").replace("\n", "%0A")


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    out[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    out[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    out[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou(boxes[i], boxes[rest])
        order = rest[ious <= iou_threshold]
    return keep


def _iou(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(float(box[0]), boxes[:, 0])
    y1 = np.maximum(float(box[1]), boxes[:, 1])
    x2 = np.minimum(float(box[2]), boxes[:, 2])
    y2 = np.minimum(float(box[3]), boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area0 = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
    area1 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area0 + area1 - inter, 1e-7)
