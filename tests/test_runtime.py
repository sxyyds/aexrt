import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import struct

import numpy as np
import pytest
import aexrt.engine as engine_module
from aexrt import Graph, HostDevice, InferenceSession, NativeCppGraph, NativeCppYoloModel, NativeD3D12Device, OnnxInferenceSession, TensorSpec, build_yolo_native_d3d12_graph_package, build_yolo_output0_package, compile_graph_schedule, export_graph_abi, inspect_aexrt_engine, install_aexrt_pipeline_cache, save_aexrt_engine, save_yolo_package, yolo_postprocess
from aexrt.backends.native_d3d12_backend import _pack_winograd_f2x2_3x3_weights


def assert_close(a, b, tol=2e-4):
    if not np.allclose(a, b, atol=tol, rtol=tol):
        raise AssertionError(f"max diff={np.max(np.abs(a-b))}")


def test_native_cpp_fp16_activation_crosses_zero_dispatch_view_when_available(tmp_path):
    if not NativeD3D12Device.probe().available:
        return

    graph = Graph("native_fp16_zero_dispatch_view")
    graph.input("images", TensorSpec((1, 64, 40, 40), "float32"))
    graph.const("w0", np.zeros((64, 64, 1, 1), dtype="float32"))
    graph.const("b0", np.zeros((64,), dtype="float32"))
    graph.node(
        "Conv", "c0", "images", "w0", "b0",
        strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s0", "c0")
    graph.mul("a0", "c0", "s0")
    graph.node("Split", ["view"], "a0", axis=1, split=[64])
    graph.const("w1", np.zeros((64, 64, 3, 3), dtype="float32"))
    graph.const("b1", np.zeros((64,), dtype="float32"))
    graph.node(
        "Conv", "c1", "view", "w1", "b1",
        strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s1", "c1")
    graph.mul("a1", "c1", "s1")
    graph.reshape("output0", "a1", (1, 64, 1600))
    graph.output("output0")

    path = tmp_path / "fp16_zero_dispatch_view.aexrt"
    info = save_aexrt_engine(graph, path, classes=60, max_detections=4)
    assert info["arena_material_storage_precision_counts"]["fp16"] == 1
    assert info["arena_alias_storage_precision_counts"] == {"fp32": 0, "fp16": 1}
    assert info["physical_zero_dispatch_count"] == 1
    zero_record = next(
        record for record in info["physical_dispatch_plan"] if record["zero_dispatch"]
    )
    assert zero_record["logical_indices"] == (1,)
    view_plan = next(
        record
        for record in info["kernel_plan"]
        if record[1] == engine_module.COMMAND_KIND["VIEW"]
    )
    assert view_plan[5] & (
        engine_module.PLAN_FLAG_INPUT_FP16 | engine_module.PLAN_FLAG_OUTPUT_FP16
    ) == 0

    runtime = NativeCppYoloModel(path)
    try:
        assert runtime.run(
            np.zeros(runtime.input_element_count, dtype=np.float32),
            max_detections=1,
        ) == []
    finally:
        runtime.close()


def conv2d_silu_ref(x, weight, bias, strides=(1, 1), pads=(0, 0, 0, 0), dilations=(1, 1), group=1):
    n, c, h, w = x.shape
    oc, icg, kh, kw = weight.shape
    sh, sw = strides
    pt, pl, pb, pr = pads
    dh, dw = dilations
    oh = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1
    ow = (w + pl + pr - dw * (kw - 1) - 1) // sw + 1
    out = np.empty((n, oc, oh, ow), dtype="float32")
    ocg = oc // group
    for bn in range(n):
        for co in range(oc):
            gid = co // ocg
            for yy in range(oh):
                for xx in range(ow):
                    acc = float(bias[co])
                    for ci_local in range(icg):
                        ci = gid * icg + ci_local
                        for ky in range(kh):
                            iy = yy * sh + ky * dh - pt
                            if iy < 0 or iy >= h:
                                continue
                            for kx in range(kw):
                                ix = xx * sw + kx * dw - pl
                                if ix < 0 or ix >= w:
                                    continue
                                acc += float(x[bn, ci, iy, ix]) * float(weight[co, ci_local, ky, kx])
                    out[bn, co, yy, xx] = acc / (1.0 + np.exp(-acc))
    return out


def fold_bn_ref(weight, bias, scale, beta, mean, var, eps=1e-5):
    alpha = scale / np.sqrt(var + eps)
    return weight * alpha.reshape(-1, 1, 1, 1), (bias - mean) * alpha + beta


def maxpool2d_ref(x, kernel=5, pad=2):
    n, c, h, w = x.shape
    out = np.empty_like(x)
    for bn in range(n):
        for ch in range(c):
            for yy in range(h):
                for xx in range(w):
                    m = -np.inf
                    for ky in range(kernel):
                        iy = yy + ky - pad
                        if iy < 0 or iy >= h:
                            continue
                        for kx in range(kernel):
                            ix = xx + kx - pad
                            if ix < 0 or ix >= w:
                                continue
                            m = max(m, float(x[bn, ch, iy, ix]))
                    out[bn, ch, yy, xx] = m
    return out


def test_mlp_numpy_vs_torch():
    rng = np.random.default_rng(0)
    g = Graph("mlp")
    g.input("x", TensorSpec((3, 5), "float32"))
    g.const("w", rng.standard_normal((5, 7)).astype("float32"))
    g.const("b", rng.standard_normal((7,)).astype("float32"))
    g.matmul("h", "x", "w")
    g.add("hb", "h", "b")
    g.gelu("y", "hb")
    g.output("y")
    x = rng.standard_normal((3, 5)).astype("float32")
    y_np = InferenceSession(g, backend="numpy", optimize=True).run({"x": x})["y"]
    y_torch = InferenceSession(g, backend="torch", device="auto", optimize=True).run({"x": x})["y"]
    assert_close(y_np, y_torch)


def test_execution_plan_and_capabilities():
    rng = np.random.default_rng(10)
    g = Graph("plan")
    g.input("x", TensorSpec((2, 3), "float32"))
    g.const("w", rng.standard_normal((3, 4)).astype("float32"))
    g.const("b", np.zeros((4,), dtype="float32"))
    g.matmul("m", "x", "w")
    g.add("y", "m", "b")
    g.output("y")

    session = InferenceSession(g, backend="numpy", optimize=True)
    info = session.info()
    plan = session.execution_plan()

    assert "FusedLinear" in info.capabilities["ops"]
    assert "static_execution_plan" in info.capabilities["features"]
    assert plan is not None
    assert plan.device.device_type == "cpu"
    assert plan.memory.inputs == ("x",)
    assert plan.memory.outputs == ("y",)
    assert plan.memory.constants == ("w", "b")
    buffers = {b.name: b for b in plan.memory.buffers}
    assert buffers["x"].nbytes == 2 * 3 * 4
    assert buffers["w"].nbytes == 3 * 4 * 4


def test_memory_plan_infers_temporaries_and_reuses_allocations():
    g = Graph("memory")
    g.input("x", TensorSpec((2, 4), "float32"))
    g.relu("a", "x")
    g.sigmoid("left", "a")
    g.tanh("b", "x")
    g.relu("right", "b")
    g.output("left")
    g.output("right")

    session = InferenceSession(g, backend="numpy", optimize=False)
    plan = session.execution_plan()
    buffers = {b.name: b for b in plan.memory.buffers}

    assert buffers["a"].shape == (2, 4)
    assert buffers["a"].nbytes == 2 * 4 * 4
    assert buffers["b"].shape == (2, 4)
    assert buffers["a"].allocation_id == buffers["b"].allocation_id


def test_scheduler_builds_arena_fusion_and_tile_plan():
    rng = np.random.default_rng(11)
    g = Graph("schedule")
    g.input("x", TensorSpec((8, 16), "float32"))
    g.const("w", rng.standard_normal((16, 32)).astype("float32"))
    g.matmul("m", "x", "w")
    g.relu("a", "m")
    g.gelu("y", "a")
    g.output("y")

    session = InferenceSession(g, backend="numpy", optimize=False)
    schedule = compile_graph_schedule(session.graph, session.execution_plan())

    assert schedule.arena.alignment == 256
    assert schedule.arena.total_nbytes % 256 == 0
    assert "tile_wave_scheduler_v1" in schedule.algorithm_tags
    matmul = next(k for k in schedule.kernels if k.op == "MatMul")
    assert matmul.category == "matmul"
    assert matmul.tile is not None
    assert matmul.tile.strategy == "portable_baseline"
    assert len(schedule.fusion_groups) == 1
    assert schedule.fusion_groups[0].nodes == ("Relu_1", "Gelu_2")


def test_aexrt_graph_abi_exports_node_list():
    g = Graph("abi")
    g.input("a", TensorSpec((2, 3), "float32"))
    g.input("b", TensorSpec((2, 3), "float32"))
    g.add("sum", "a", "b")
    g.relu("y", "sum")
    g.output("y")

    abi = export_graph_abi(g)
    assert abi["format"] == "aexrt.graph"
    assert abi["version"] == 1
    assert abi["element_count"] == 6
    assert [n["op"] for n in abi["nodes"]] == ["Add", "Relu"]
    assert len(abi["inputs"]) == 2


def test_native_cpp_loads_python_exported_aexrt_json_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    g = Graph("abi_json_to_cpp")
    g.input("a", TensorSpec((2, 3), "float32"))
    g.input("b", TensorSpec((2, 3), "float32"))
    g.add("sum", "a", "b")
    g.relu("y", "sum")
    g.output("y")

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_add_relu.aexrt.json"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g.save_aexrt(path)

    try:
        runtime = NativeCppGraph.load_aexrt(path)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        a = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype="float32")
        b = np.array([[3.0, -4.0, 5.0], [-6.0, 7.0, -8.0]], dtype="float32")
        y = runtime.run({"a": a, "b": b})["y"]
        assert runtime.dispatch_count == 1
        assert runtime.buffer_count == 3
        assert_close(y, np.maximum(a + b, 0))
    finally:
        runtime.close()


def test_native_cpp_fuses_relu_gelu_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    g = Graph("relu_gelu_fusion")
    g.input("x", TensorSpec((2, 3), "float32"))
    g.relu("r", "x")
    g.gelu("y", "r")
    g.output("y")

    try:
        runtime = NativeCppGraph(g)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        x = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype="float32")
        y = runtime.run({"x": x})["y"]
        r = np.maximum(x, 0)
        expected = 0.5 * r * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (r + 0.044715 * np.power(r, 3))))
        assert runtime.dispatch_count == 1
        assert runtime.buffer_count == 2
        assert_close(y, expected, tol=2e-5)
    finally:
        runtime.close()


def test_native_cpp_dynamic_fuses_elementwise_dag_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    g = Graph("dynamic_transformer_elementwise_dag")
    g.input("x", TensorSpec((2, 3), "float32"))
    g.input("gate", TensorSpec((2, 3), "float32"))
    g.add("residual", "x", "gate")
    g.sub("centered", "residual", "gate")
    g.div("scaled", "centered", "gate")
    g.tanh("candidate", "scaled")
    g.sigmoid("weight", "gate")
    g.mul("mixed", "candidate", "weight")
    g.gelu("y", "mixed")
    g.output("y")

    abi = export_graph_abi(g)
    assert [n["op"] for n in abi["nodes"]] == ["Add", "Sub", "Div", "Tanh", "Sigmoid", "Mul", "Gelu"]

    try:
        runtime = NativeCppGraph(g)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        x = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype="float32")
        gate = np.array([[0.5, -1.5, 2.5], [-3.5, 4.5, -5.5]], dtype="float32")
        y = runtime.run({"x": x, "gate": gate})["y"]
        residual = x + gate
        centered = residual - gate
        scaled = centered / gate
        candidate = np.tanh(scaled)
        weight = 1.0 / (1.0 + np.exp(-gate))
        mixed = candidate * weight
        expected = 0.5 * mixed * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (mixed + 0.044715 * np.power(mixed, 3))))
        assert runtime.dispatch_count == 1
        assert runtime.buffer_count == 3
        assert_close(y, expected, tol=2e-5)
    finally:
        runtime.close()


def test_native_cpp_fuses_scalar_constants_from_abi_json_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    g = Graph("dynamic_scalar_constants")
    g.input("x", TensorSpec((2, 3), "float32"))
    g.const("scale", np.array(0.125, dtype="float32"))
    g.const("bias", np.array(-0.25, dtype="float32"))
    g.mul("scaled", "x", "scale")
    g.add("shifted", "scaled", "bias")
    g.tanh("candidate", "shifted")
    g.sigmoid("gate", "shifted")
    g.mul("y", "candidate", "gate")
    g.output("y")

    abi = export_graph_abi(g)
    assert len(abi["inputs"]) == 1
    assert [(c["name"], c["scalar"]) for c in abi["constants"]] == [("scale", 0.125), ("bias", -0.25)]

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_scalar_constants.aexrt.json"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g.save_aexrt(path)

    try:
        runtime = NativeCppGraph.load_aexrt(path)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        x = np.array([[-8.0, -2.0, 0.0], [2.0, 4.0, 8.0]], dtype="float32")
        y = runtime.run({"x": x})["y"]
        shifted = x * np.float32(0.125) + np.float32(-0.25)
        expected = np.tanh(shifted) * (1.0 / (1.0 + np.exp(-shifted)))
        assert runtime.dispatch_count == 1
        assert runtime.buffer_count == 2
        assert_close(y, expected, tol=2e-5)
    finally:
        runtime.close()


def test_native_cpp_runs_three_input_fused_dag_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    g = Graph("dynamic_three_input_dag")
    g.input("x", TensorSpec((2, 3), "float32"))
    g.input("gate", TensorSpec((2, 3), "float32"))
    g.input("residual", TensorSpec((2, 3), "float32"))
    g.add("sum", "x", "residual")
    g.sigmoid("weight", "gate")
    g.mul("mixed", "sum", "weight")
    g.tanh("y", "mixed")
    g.output("y")

    abi = export_graph_abi(g)
    assert len(abi["inputs"]) == 3

    try:
        runtime = NativeCppGraph(g)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        x = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype="float32")
        gate = np.array([[0.5, -1.5, 2.5], [-3.5, 4.5, -5.5]], dtype="float32")
        residual = np.array([[1.0, -0.5, 0.25], [-0.75, 0.125, 2.0]], dtype="float32")
        y = runtime.run({"x": x, "gate": gate, "residual": residual})["y"]
        weight = 1.0 / (1.0 + np.exp(-gate))
        expected = np.tanh((x + residual) * weight)
        assert runtime.dispatch_count == 1
        assert runtime.buffer_count == 4
        assert_close(y, expected, tol=2e-5)
    finally:
        runtime.close()


def test_torch_backend_runs_vision_ops():
    rng = np.random.default_rng(21)
    g = Graph("vision_ops")
    g.input("x", TensorSpec((1, 2, 6, 6), "float32"))
    g.const("w", rng.standard_normal((4, 2, 3, 3)).astype("float32") * 0.1)
    g.const("b", rng.standard_normal((4,)).astype("float32") * 0.1)
    g.node("Conv", "c", "x", "w", "b", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.node("MaxPool", "p", "c", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], ceil_mode=0)
    g.node("Split", ["left", "right"], "p", axis=1, split=[2, 2])
    g.node("Slice", "cut", "left", starts=[1], ends=[5], axes=[3], steps=[1])
    g.node("Resize", "up", "right", sizes=[1, 2, 6, 6], mode="nearest", nearest_mode="floor", coordinate_transformation_mode="asymmetric")
    g.node("Concat", "y", "cut", "up", axis=3)
    g.output("y")

    x = rng.standard_normal((1, 2, 6, 6)).astype("float32")
    y = InferenceSession(g, backend="torch", device="cpu", optimize=False).run({"x": x})["y"]
    assert y.shape == (1, 2, 6, 10)
    assert np.isfinite(y).all()


def test_yolo_postprocess_nms():
    out = np.zeros((1, 6, 4), dtype="float32")
    out[0, :4, 0] = [10, 10, 10, 10]
    out[0, :4, 1] = [11, 10, 10, 10]
    out[0, :4, 2] = [50, 50, 8, 8]
    out[0, :4, 3] = [80, 80, 8, 8]
    out[0, 4:, 0] = [0.9, 0.1]
    out[0, 4:, 1] = [0.8, 0.1]
    out[0, 4:, 2] = [0.1, 0.7]
    out[0, 4:, 3] = [0.2, 0.1]
    dets = yolo_postprocess(out, conf_threshold=0.25, iou_threshold=0.5)
    assert [(d.class_id, round(d.score, 2)) for d in dets] == [(0, 0.9), (1, 0.7)]


def test_yolo_postprocess_channels_last_objectness():
    out = np.zeros((1, 5, 7), dtype="float32")
    out[0, 0, :4] = [10, 10, 10, 10]
    out[0, 0, 4:] = [0.5, 0.9, 0.1]
    out[0, 1, :4] = [50, 50, 8, 8]
    out[0, 1, 4:] = [0.8, 0.1, 0.7]
    dets = yolo_postprocess(out, conf_threshold=0.25, iou_threshold=0.5, layout="channels_last", objectness=True)
    assert [(d.class_id, round(d.score, 2)) for d in dets] == [(1, 0.56), (0, 0.45)]


def test_yolo_package_schema_exports_output0_postprocess():
    package = build_yolo_output0_package((1, 6, 5), source_model="model.onnx", classes=None, max_candidates=8, conf_threshold=0.2)
    assert package["format"] == "aexrt.yolo.package"
    assert package["mode"] == "output0_postprocess"
    assert package["layout"] == "channels_first"
    assert package["objectness"] is False
    assert package["channels"] == 6
    assert package["anchors"] == 5
    assert package["classes"] == 2
    assert package["next_mode"] == "native_d3d12_graph"

    v5 = build_yolo_output0_package((1, 5, 7), source_model="v5.onnx", classes=2, layout="channels_last", objectness=True)
    assert v5["layout"] == "channels_last"
    assert v5["objectness"] is True
    assert v5["channels"] == 7
    assert v5["anchors"] == 5
    assert v5["classes"] == 2


def test_yolo_native_d3d12_graph_package_contains_prepared_metadata():
    rng = np.random.default_rng(40)
    g = Graph("tiny_yolo_native_package")
    g.input("images", TensorSpec((1, 3, 4, 4), "float32"))
    g.const("w", (rng.standard_normal((6, 3, 1, 1)) * 0.1).astype("float32"))
    g.const("b", np.zeros((6,), dtype="float32"))
    g.conv2d("conv", "images", "w", "b")
    g.sigmoid("gate", "conv")
    g.mul("act", "conv", "gate")
    g.reshape("output0", "act", (1, 6, 16))
    g.output("output0")

    package = build_yolo_native_d3d12_graph_package(g, source_model="tiny.onnx", classes=2, embed_constants=True)
    assert package["mode"] == "native_d3d12_graph"
    assert package["version"] == 2
    assert package["yolo"]["layout"] == "channels_first"
    assert package["yolo"]["objectness"] is False
    assert package["yolo"]["anchors"] == 16
    assert package["yolo"]["classes"] == 2
    assert package["prepared_graph"]["upload"]["input_element_count"] == 48
    assert package["prepared_graph"]["command_replay"] is True
    assert package["compiler"]["executable_in_cxx"] is True
    assert package["compiler"]["cxx_kernel_coverage"]["supported_command_count"] >= 2
    assert package["constants"]["w"]["encoding"] == "base64"
    assert any(cmd["kernel"] == "conv2d_silu" for cmd in package["prepared_graph"]["commands"])
    assert package["resource_table"]


def test_aexrt_binary_engine_contains_compiled_sections_and_checksums(tmp_path):
    rng = np.random.default_rng(41)
    g = Graph("tiny_aexrt_engine")
    g.input("images", TensorSpec((1, 32, 20, 20), "float32"))
    g.const("w", (rng.standard_normal((64, 32, 3, 3)) * 0.01).astype("float32"))
    g.const("b", np.zeros((64,), dtype="float32"))
    g.node("Conv", "c", "images", "w", "b", strides=[2, 2], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("s", "c")
    g.mul("act", "c", "s")
    g.reshape("output0", "act", (1, 64, 100))
    g.output("output0")

    path = tmp_path / "tiny.aexrt"
    info = save_aexrt_engine(g, path, source_model="tiny.onnx", classes=60)
    data = path.read_bytes()
    assert data[:8] == b"AEXRTENG"
    assert b"AEXRT_CXX_REPLAY_V1" not in data
    assert b"cxx_replay_text" not in data
    assert b"base64" not in data
    assert info["format"] == "aexrt.engine"
    assert info["section_count"] == 11
    assert info["precision"] == 2
    assert info["kernel_plan_count"] == info["command_count"]
    assert info["dxil_command_count"] == info["command_count"]
    assert info["packed_weight_count"] == 1
    assert info["fusion_group_count"] == 0
    assert info["fusion_plan"] == []
    assert info["dxil_cache_count"] == 0
    assert info["shader_cache_count"] == 0
    assert info["dxbc_cache_count"] == 0
    assert info["pso_cache_count"] == 0
    assert info["constant_precision_counts"] == {"fp32": 0, "fp16": info["constant_count"]}
    assert info["packed_layout_counts"] == {2: 1}
    assert info["arena_slot_count"] >= 2
    assert info["arena_nbytes"] > 0
    conv_plan = info["kernel_plan"][0]
    assert conv_plan[2] == 43
    assert conv_plan[3] == 2
    assert conv_plan[4] != (1 << 32) - 1
    assert conv_plan[5] == 3
    assert inspect_aexrt_engine(path)["source_hash"] == info["source_hash"]

    fp32_info = save_aexrt_engine(g, tmp_path / "tiny_fp32.aexrt", source_model="tiny.onnx", classes=60, precision="fp32")
    assert fp32_info["kernel_plan"][0][2] == 24
    assert fp32_info["kernel_plan"][0][5] == 1

    constants_offset = info["sections"][4]["offset"]
    constant_count = struct.unpack_from("<I", data, constants_offset)[0]
    constant_dtypes = {
        struct.unpack_from("<I", data, constants_offset + 8 + index * 32 + 4)[0]
        for index in range(constant_count)
    }
    assert constant_dtypes == {2}

    packed_offset = info["sections"][6]["offset"]
    _, packed_layout, _, _, packed_elements, _, packed_nbytes = struct.unpack_from(
        "<4IQQQ", data, packed_offset + 8
    )
    assert packed_layout == 2
    assert packed_nbytes == packed_elements * 2

    corrupted = bytearray(data)
    constants = info["sections"][4]
    corrupted[constants["offset"] + constants["size"] - 1] ^= 0x01
    with pytest.raises(ValueError, match="checksum mismatch"):
        inspect_aexrt_engine(corrupted)

    cache = bytearray(197)
    struct.pack_into("<4I", cache, 0, 1, 1, 1, 1)
    struct.pack_into("<IIQQQ", cache, 16, 1, 1, 0x11, 128, 4)
    struct.pack_into("<IIQQQ", cache, 48, 2, 0, 0x22, 192, 5)
    cache[128:132] = b"DXIL"
    cache[192:197] = b"CACHE"
    cached_info = install_aexrt_pipeline_cache(path, cache)
    assert cached_info["section_count"] == 11
    assert cached_info["dxil_cache_count"] == 1
    assert cached_info["shader_cache_count"] == 1
    assert cached_info["dxbc_cache_count"] == 0
    assert cached_info["pso_cache_count"] == 1
    assert cached_info["pipeline_cache_flags"] == 1
    assert [record["kind"] for record in cached_info["pipeline_cache_records"]] == ["dxil", "pso"]

    malformed_cache = bytearray(cache)
    struct.pack_into("<Q", malformed_cache, 16 + 16, 64)
    with pytest.raises(ValueError, match="pipeline-cache record"):
        install_aexrt_pipeline_cache(path, malformed_cache)


@pytest.mark.parametrize(
    ("dim", "in_channels", "out_channels"),
    (
        (20, 128, 64),
        (20, 256, 64),
        (20, 64, 64),
        (10, 256, 64),
        (10, 128, 128),
        (10, 64, 64),
    ),
)
def test_aexrt_binary_engine_plans_measured_winograd_hotspots(tmp_path, dim, in_channels, out_channels):
    graph = Graph(f"winograd_{dim}_{in_channels}_{out_channels}")
    graph.input("images", TensorSpec((1, in_channels, dim, dim), "float32"))
    graph.const("w", np.zeros((out_channels, in_channels, 3, 3), dtype="float32"))
    graph.const("b", np.zeros((out_channels,), dtype="float32"))
    graph.node("Conv", "c", "images", "w", "b", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, out_channels, dim * dim))
    graph.output("output0")

    info = save_aexrt_engine(graph, tmp_path / "winograd.aexrt", classes=out_channels - 4)
    plan = info["kernel_plan"][0]
    assert plan[2:4] == (40, 2)
    assert plan[4] != (1 << 32) - 1
    assert plan[5] == 3
    assert info["packed_layout_counts"] == {3: 1}


@pytest.mark.parametrize(
    ("name", "input_shape", "weight_shape", "strides", "pads", "output_shape", "expected_kernel"),
    (
        (
            "yolov5_frontend_6x6",
            (1, 3, 320, 320),
            (32, 3, 6, 6),
            [2, 2],
            [2, 2, 2, 2],
            (1, 32, 160, 160),
            42,
        ),
        (
            "yolov5_c3_20x20_128x128",
            (1, 128, 20, 20),
            (128, 128, 3, 3),
            [1, 1],
            [1, 1, 1, 1],
            (1, 128, 20, 20),
            40,
        ),
        (
            "yolov5_c3_10x10_256x256",
            (1, 256, 10, 10),
            (256, 256, 3, 3),
            [1, 1],
            [1, 1, 1, 1],
            (1, 256, 10, 10),
            46,
        ),
        (
            "yolov11_stride2_80x80_128x128",
            (1, 128, 80, 80),
            (128, 128, 3, 3),
            [2, 2],
            [1, 1, 1, 1],
            (1, 128, 40, 40),
            33,
        ),
        (
            "yolov11_splitk_10x10_512x64",
            (1, 512, 10, 10),
            (64, 512, 3, 3),
            [1, 1],
            [1, 1, 1, 1],
            (1, 64, 10, 10),
            28,
        ),
    ),
)
def test_aexrt_binary_engine_embeds_exact_model_hotspot_plans(
    tmp_path, name, input_shape, weight_shape, strides, pads, output_shape, expected_kernel
):
    graph = Graph(name)
    graph.input("images", TensorSpec(input_shape, "float32"))
    graph.const("w", np.zeros(weight_shape, dtype="float32"))
    graph.const("b", np.zeros((weight_shape[0],), dtype="float32"))
    graph.node(
        "Conv",
        "c",
        "images",
        "w",
        "b",
        strides=strides,
        pads=pads,
        dilations=[1, 1],
        group=1,
    )
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, output_shape[1], output_shape[2] * output_shape[3]))
    graph.output("output0")

    info = save_aexrt_engine(
        graph,
        tmp_path / f"{name}.aexrt",
        classes=output_shape[1] - 4,
        precision="fp16",
    )
    plan = info["kernel_plan"][0]
    if expected_kernel in {40, 46}:
        assert plan[2:4] == (expected_kernel, 2)
        assert plan[4] != (1 << 32) - 1
        assert plan[5] == 3
        assert info["packed_weight_count"] == 1
        assert info["packed_layout_counts"] == {3 if expected_kernel == 40 else 5: 1}
    elif expected_kernel == 33:
        assert plan[2:4] == (33, 2)
        assert plan[4] != (1 << 32) - 1
        assert plan[5] == 1
        assert info["packed_weight_count"] == 1
        assert info["packed_layout_counts"] == {2: 1}
    else:
        assert plan[2:6] == (expected_kernel, 2, (1 << 32) - 1, 1)
        assert info["packed_weight_count"] == 0
        assert info["packed_layout_counts"] == {}


@pytest.mark.parametrize(
    ("precision", "expected_kernel", "expected_plan_flags", "expected_layout"),
    (("fp16", 44, 3, 2), ("fp32", 39, 1, 1)),
)
def test_aexrt_binary_engine_embeds_packed_weight_for_40x40_hotspot(
    tmp_path, precision, expected_kernel, expected_plan_flags, expected_layout
):
    graph = Graph("packed_pos2_40x40_64x64")
    graph.input("images", TensorSpec((1, 64, 40, 40), "float32"))
    graph.const("w", np.zeros((64, 64, 3, 3), dtype="float32"))
    graph.const("b", np.zeros((64,), dtype="float32"))
    graph.node("Conv", "c", "images", "w", "b", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, 64, 1600))
    graph.output("output0")

    info = save_aexrt_engine(
        graph, tmp_path / f"packed_pos2_{precision}.aexrt", classes=60, precision=precision
    )
    plan = info["kernel_plan"][0]
    assert plan[2] == expected_kernel
    assert plan[5] == expected_plan_flags
    assert plan[4] != (1 << 32) - 1
    assert info["packed_weight_count"] == 1
    assert info["packed_layout_counts"] == {expected_layout: 1}


def test_aexrt_binary_engine_plans_packed_pos2_for_40x40_128x64_hotspot(
    tmp_path,
):
    graph = Graph("packed_pos2_40x40_128x64")
    graph.input("images", TensorSpec((1, 128, 40, 40), "float32"))
    graph.const("w", np.zeros((64, 128, 3, 3), dtype="float32"))
    graph.const("b", np.zeros((64,), dtype="float32"))
    graph.node(
        "Conv",
        "c",
        "images",
        "w",
        "b",
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1,
    )
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, 64, 1600))
    graph.output("output0")

    info = save_aexrt_engine(
        graph,
        tmp_path / "packed_pos2_128x64.aexrt",
        classes=60,
        precision="fp16",
    )
    assert info["kernel_plan"][0][2:6] == (39, 2, 1, 1)
    assert info["packed_weight_count"] == 1
    assert info["packed_layout_counts"] == {2: 1}


@pytest.mark.parametrize(
    ("in_channels", "in_dim", "out_channels", "out_dim"),
    ((3, 320, 16, 160), (16, 160, 32, 80)),
)
def test_aexrt_binary_engine_plans_frontend_direct_pack4(
    tmp_path, in_channels, in_dim, out_channels, out_dim
):
    graph = Graph(f"frontend_direct_pack4_{in_channels}_{out_channels}")
    graph.input("images", TensorSpec((1, in_channels, in_dim, in_dim), "float32"))
    graph.const("w", np.zeros((out_channels, in_channels, 3, 3), dtype="float32"))
    graph.const("b", np.zeros((out_channels,), dtype="float32"))
    graph.node("Conv", "c", "images", "w", "b", strides=[2, 2], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, out_channels, out_dim * out_dim))
    graph.output("output0")

    info = save_aexrt_engine(graph, tmp_path / "frontend.aexrt", classes=out_channels - 4)
    assert info["kernel_plan"][0][2] == 24


@pytest.mark.parametrize(("in_channels", "expected"), ((32, True), (64, True), (68, False), (128, False), (256, False)))
def test_aexrt_binary_engine_plans_stride2_conv1x1_fusion(tmp_path, in_channels, expected):
    graph = Graph(f"stride2_conv1x1_fusion_{in_channels}")
    graph.input("images", TensorSpec((1, in_channels, 20, 20), "float32"))
    graph.const("w0", np.zeros((64, in_channels, 3, 3), dtype="float32"))
    graph.const("b0", np.zeros((64,), dtype="float32"))
    graph.const("w1", np.zeros((64, 64, 1, 1), dtype="float32"))
    graph.const("b1", np.zeros((64,), dtype="float32"))
    graph.node("Conv", "c0", "images", "w0", "b0", strides=[2, 2], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    graph.sigmoid("s0", "c0")
    graph.mul("a0", "c0", "s0")
    graph.node("Conv", "c1", "a0", "w1", "b1", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    graph.sigmoid("s1", "c1")
    graph.mul("a1", "c1", "s1")
    graph.reshape("output0", "a1", (1, 64, 100))
    graph.output("output0")

    info = save_aexrt_engine(graph, tmp_path / f"stride2_conv1x1_{in_channels}.aexrt", classes=60)
    groups = [group for group in info["fusion_plan"] if group["kind"] == 6]
    assert bool(groups) is expected
    if expected:
        assert len(groups) == 1
        assert (groups[0]["start"], groups[0]["end"], groups[0]["precision"]) == (0, 1, 1)


def test_aexrt_binary_engine_packs_exact_fp16_paired_winograd(tmp_path):
    graph = Graph("paired_winograd_10x10_256x64")
    graph.input("images", TensorSpec((1, 256, 10, 10), "float32"))
    graph.const("w0", np.zeros((64, 256, 3, 3), dtype="float32"))
    graph.const("b0", np.zeros((64,), dtype="float32"))
    graph.const("w1", np.zeros((64, 256, 3, 3), dtype="float32"))
    graph.const("b1", np.zeros((64,), dtype="float32"))
    for suffix in ("0", "1"):
        graph.node(
            "Conv", f"c{suffix}", "images", f"w{suffix}", f"b{suffix}",
            strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1,
        )
        graph.sigmoid(f"s{suffix}", f"c{suffix}")
        graph.mul(f"a{suffix}", f"c{suffix}", f"s{suffix}")
    graph.node("Concat", "cat", "a0", "a1", axis=1)
    graph.reshape("output0", "cat", (1, 128, 100))
    graph.output("output0")

    info = save_aexrt_engine(graph, tmp_path / "paired_winograd.aexrt", classes=124)
    groups = [group for group in info["fusion_plan"] if group["kind"] == 1]
    assert groups == [{
        "kind": 1,
        "start": 0,
        "end": 1,
        "precision": 2,
        "kernel": 40,
        "flags": 3,
        "aux0": 0,
        "aux1": 0,
    }]
    assert info["packed_weight_count"] == 2
    assert info["packed_layout_counts"] == {3: 2}


def _make_fp16_concat_conv1x1_graph(name, *, insert_unrelated_command=False):
    graph = Graph(name)
    graph.input("images", TensorSpec((1, 8, 20, 20), "float32"))
    for suffix in ("0", "1"):
        graph.const(f"w{suffix}", np.zeros((32, 8, 1, 1), dtype="float32"))
        graph.const(f"b{suffix}", np.zeros((32,), dtype="float32"))
        graph.node(
            "Conv", f"c{suffix}", "images", f"w{suffix}", f"b{suffix}",
            strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
        )
        graph.sigmoid(f"s{suffix}", f"c{suffix}")
        graph.mul(f"a{suffix}", f"c{suffix}", f"s{suffix}")
    graph.node("Concat", "cat", "a0", "a1", axis=1)
    if insert_unrelated_command:
        graph.relu("unrelated", "a0")
    graph.const("wf", np.zeros((64, 64, 1, 1), dtype="float32"))
    graph.const("bf", np.zeros((64,), dtype="float32"))
    graph.node(
        "Conv", "cf", "cat", "wf", "bf",
        strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
    )
    graph.sigmoid("sf", "cf")
    graph.mul("act", "cf", "sf")
    graph.reshape("output0", "act", (1, 64, 400))
    graph.output("output0")
    return graph


def test_aexrt_binary_engine_plans_native_fp16_direct_concat_conv1x1(tmp_path):
    graph = _make_fp16_concat_conv1x1_graph("native_fp16_direct_concat")
    original = engine_module._uses_native_fp16_concat_conv1x1
    engine_module._uses_native_fp16_concat_conv1x1 = lambda command: True
    try:
        info = save_aexrt_engine(graph, tmp_path / "direct_concat.aexrt", classes=60)
    finally:
        engine_module._uses_native_fp16_concat_conv1x1 = original
    concat_plans = [record for record in info["kernel_plan"] if record[1] == 3]
    assert len(concat_plans) == 1
    assert concat_plans[0][2:4] == (3, 2)
    assert concat_plans[0][4] != (1 << 32) - 1
    assert concat_plans[0][5] == 7
    assert info["packed_layout_counts"] == {4: 1}


def test_aexrt_replay_keeps_shared_concat_materialized():
    graph = _make_fp16_concat_conv1x1_graph("shared_concat")
    graph.relu("cat_side", "cat")

    package = build_yolo_native_d3d12_graph_package(graph, classes=60)
    replay = engine_module._parse_replay_text(str(package["cxx_replay_text"]))
    producers = {
        int(command["output"]): index
        for index, command in enumerate(replay["commands"])
    }

    assert any(command["kind"] == "CONCAT" for command in replay["commands"])
    for command in replay["commands"]:
        for value_id in command["inputs"]:
            value = replay["values"][value_id]
            if value["flags"] & (engine_module.VALUE_INPUT | engine_module.VALUE_CONSTANT):
                continue
            assert value_id in producers


def test_aexrt_binary_engine_plans_native_fp16_late_concat_without_duplicate_pack(tmp_path):
    graph = _make_fp16_concat_conv1x1_graph(
        "native_fp16_late_concat", insert_unrelated_command=True
    )
    original = engine_module._uses_native_fp16_conv1x1
    engine_module._uses_native_fp16_conv1x1 = (
        lambda params: len(params) >= 5 and int(params[1]) == int(params[4]) == 64
    )
    try:
        info = save_aexrt_engine(graph, tmp_path / "late_concat.aexrt", classes=60)
    finally:
        engine_module._uses_native_fp16_conv1x1 = original
    assert not [record for record in info["kernel_plan"] if record[1] == 3]
    conv_plans = [record for record in info["kernel_plan"] if record[1] == 2 and record[2] == 45]
    assert len(conv_plans) == 1
    late_groups = [group for group in info["fusion_plan"] if group["kind"] == 3]
    assert len(late_groups) == 1
    assert late_groups[0]["precision"] == 2
    assert late_groups[0]["kernel"] == 3
    assert late_groups[0]["flags"] == 7
    assert info["packed_weight_count"] == 1
    assert info["packed_layout_counts"] == {4: 1}


def test_aexrt_native_fp16_concat_conv1x1_gate_rejects_more_than_eight_branches():
    command = {
        "inputs": list(range(9)) + [9, 10],
        "params": [1, 20, 20, 64, 1] + [8] * 8 + [0],
    }
    assert not engine_module._uses_native_fp16_concat_conv1x1(command)


@pytest.mark.parametrize(
    ("height", "channels", "out_channels"),
    (
        (80, (32, 32), 64),
        (40, (64, 64), 128),
        (40, (256, 256), 128),
        (40, (64, 64, 64), 128),
    ),
)
def test_aexrt_native_fp16_concat_conv1x1_gate_selects_measured_hotspots(
    height, channels, out_channels,
):
    command = {
        "inputs": list(range(len(channels))) + [10, 11],
        "params": [1, height, height, out_channels, 1, *channels],
    }

    assert engine_module._uses_native_fp16_concat_conv1x1(command)
    command["params"][2] = 20
    assert not engine_module._uses_native_fp16_concat_conv1x1(command)


def test_aexrt_native_fp16_direct_concat_gate_accepts_v11_three_branch_shape():
    command = {
        "inputs": [0, 1, 2, 10, 11],
        "params": [1, 20, 20, 256, 1, 128, 128, 128],
    }

    assert engine_module._uses_native_fp16_concat_conv1x1(command)
    assert engine_module._uses_native_fp16_residual_concat_conv1x1(command)


def test_aexrt_engine_omits_unreferenced_empty_constant_and_loads_natively(tmp_path):
    g = Graph("aexrt_engine_unused_empty_constant")
    g.input("images", TensorSpec((1, 8, 4, 4), "float32"))
    g.const("w", np.zeros((8, 8, 1, 1), dtype="float32"))
    g.const("b", np.zeros((8,), dtype="float32"))
    g.const("unused_empty", np.empty((0,), dtype="float32"))
    g.node("Conv", "c", "images", "w", "b", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.reshape("output0", "c", (1, 8, 16))
    g.output("output0")

    path = tmp_path / "unused_empty_constant.aexrt"
    engine_info = save_aexrt_engine(g, path, classes=4)
    inspected = inspect_aexrt_engine(path)
    assert engine_info["constant_count"] == 2
    assert inspected["constant_count"] == 2

    if not NativeD3D12Device.probe().available:
        return
    runtime = NativeCppYoloModel(path)
    try:
        assert runtime.constant_count == 2
        assert runtime.prepared_command_count == runtime.supported_prepared_command_count
    finally:
        runtime.close()


def test_native_cpp_runs_aexrt_engine_with_memory_arena_when_available(tmp_path):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    g = Graph("aexrt_engine_nonzero")
    g.input("images", TensorSpec((1, 64, 10, 10), "float32"))
    weight = np.zeros((64, 64, 1, 1), dtype="float32")
    weight[np.arange(64), np.arange(64), 0, 0] = 1.0
    g.const("w", weight)
    g.const("b", np.zeros((64,), dtype="float32"))
    g.node("Conv", "c", "images", "w", "b", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.reshape("output0", "c", (1, 64, 100))
    g.output("output0")

    path = tmp_path / "nonzero.aexrt"
    engine_info = save_aexrt_engine(g, path, classes=60, max_detections=4)
    assert engine_info["memory_plan_version"] == 2
    assert 0 < engine_info["arena_slot_count"] < engine_info["arena_value_count"]
    runtime = NativeCppYoloModel(path)
    try:
        x = np.zeros((1, 64, 10, 10), dtype="float32")
        x[0, 0, 5, 5] = 10.0
        x[0, 1, 5, 5] = 10.0
        x[0, 2, 5, 5] = 4.0
        x[0, 3, 5, 5] = 4.0
        x[0, 4, 5, 5] = 0.9
        detections = runtime.run(x, max_detections=4)
        assert len(detections) == 1
        assert detections[0].class_id == 0
        assert abs(detections[0].score - 0.9) < 1e-6
        assert_close(np.asarray(detections[0].xyxy), np.asarray([8.0, 8.0, 12.0, 12.0]), tol=1e-6)
    finally:
        runtime.close()


def test_native_cpp_aexrt_physical_plan_mismatch_fails_closed(tmp_path):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    g = Graph("aexrt_invalid_physical_fusion")
    g.input("images", TensorSpec((1, 64, 10, 10), "float32"))
    weight = np.zeros((64, 64, 1, 1), dtype="float32")
    weight[np.arange(64), np.arange(64), 0, 0] = 1.0
    g.const("w", weight)
    g.const("b", np.zeros((64,), dtype="float32"))
    g.node(
        "Conv",
        "c",
        "images",
        "w",
        "b",
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        dilations=[1, 1],
        group=1,
    )
    g.reshape("output0", "c", (1, 64, 100))
    g.output("output0")

    valid_path = tmp_path / "valid.aexrt"
    valid_info = save_aexrt_engine(g, valid_path, classes=60, max_detections=4)
    assert valid_info["command_count"] == 2
    data = valid_path.read_bytes()
    sections = {
        section_type: data[section["offset"] : section["offset"] + section["size"]]
        for section_type, section in valid_info["sections"].items()
    }
    first_plan = valid_info["kernel_plan"][0]
    malformed_record = {
        "stable_id": 0xA3E7_0000_0000_0001,
        "logical_start": 0,
        "logical_end": 1,
        "execution_index": 0,
        "logical_indices": (0, 1),
        "kernel": first_plan[2],
        "precision": first_plan[3],
        "flags": first_plan[5],
        "fusion_kind": engine_module.FUSION_PAIRED_CONV3X3_SILU,
    }
    sections[engine_module.SECTION_PHYSICAL_DISPATCH_PLAN] = (
        engine_module._encode_physical_dispatch_plan(
            [malformed_record], valid_info["command_count"]
        )
    )
    # This test targets physical-plan validation specifically.  Drop the
    # optional V2 barrier plan because its dispatch-count ABI correctly rejects
    # the mutated physical timeline during generic inspection.
    del sections[engine_module.SECTION_ARENA_BARRIER_PLAN]
    malformed_path = tmp_path / "malformed.aexrt"
    malformed_path.write_bytes(engine_module._build_container(sections))

    malformed_info = inspect_aexrt_engine(malformed_path)
    assert malformed_info["physical_dispatch_count"] == 1
    runtime = NativeCppYoloModel(malformed_path)
    try:
        with pytest.raises(RuntimeError, match="run failed"):
            runtime.run(np.zeros(runtime.input_element_count, dtype=np.float32), 4)
    finally:
        runtime.close()


def test_native_cpp_sm62_fp16_engine_matches_sm5_engine_when_available(tmp_path, monkeypatch):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(20260712)
    g = Graph("native_fp16_stride2_numeric")
    g.input("images", TensorSpec((1, 32, 20, 20), "float32"))
    weight = np.zeros((64, 32, 3, 3), dtype="float32")
    input_values = rng.uniform(0.75, 1.25, size=32).astype("float32")
    for output_channel, target in enumerate((10.0, 10.0, 4.0, 4.0, 1.5)):
        kernel = rng.uniform(0.5, 1.5, size=32).astype("float32")
        kernel *= target / float(np.dot(kernel, input_values))
        weight[output_channel, :, 1, 1] = kernel
    g.const("w", weight)
    g.const("b", np.zeros((64,), dtype="float32"))
    g.node("Conv", "c", "images", "w", "b", strides=[2, 2], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("s", "c")
    g.mul("act", "c", "s")
    g.reshape("output0", "act", (1, 64, 100))
    g.output("output0")

    fp16_path = tmp_path / "native_fp16_stride2.aexrt"
    fp32_path = tmp_path / "sm5_stride2.aexrt"
    fp16_info = save_aexrt_engine(g, fp16_path, classes=60, conf_threshold=0.25, max_detections=4)
    fp32_info = save_aexrt_engine(g, fp32_path, classes=60, conf_threshold=0.25, max_detections=4, precision="fp32")
    fp16_plan = fp16_info["kernel_plan"][0]
    assert fp16_plan[2] == 43
    assert fp16_plan[3] == 2
    assert fp16_plan[4] != (1 << 32) - 1
    assert fp16_plan[5] == 3
    assert fp32_info["kernel_plan"][0][2] == 24
    input_data = np.zeros((1, 32, 20, 20), dtype="float32")
    input_data[0, :, 10, 10] = input_values

    monkeypatch.setenv("AEXRT_NATIVE_D3D12_DISABLE_NATIVE_FP16_3X3", "1")
    monkeypatch.setenv("AEXRT_NATIVE_D3D12_FORCE_NATIVE_FP16_3X3", "1")
    fallback = NativeCppYoloModel(fp32_path)
    try:
        capabilities = fallback.capabilities
        reference = fallback.run(input_data, max_detections=4)
        assert fallback.tileflow_native_fp16_count == 0
    finally:
        fallback.close()

    assert capabilities["highest_shader_model"] >= 0x51
    assert isinstance(capabilities["native_fp16_supported"], bool)
    assert isinstance(capabilities["dxc_available"], bool)
    fp16_capable = capabilities["native_fp16_supported"] and capabilities["dxc_available"] and capabilities["highest_shader_model"] >= 0x62
    native_fp16 = NativeCppYoloModel(fp16_path)
    try:
        actual = native_fp16.run(input_data, max_detections=4)
        assert native_fp16.tileflow_native_fp16_count == (1 if fp16_capable else 0)
        pipeline_cache = native_fp16.export_pipeline_cache()
    finally:
        native_fp16.close()

    assert len(reference) == len(actual) == 1
    assert reference[0].class_id == actual[0].class_id
    assert abs(reference[0].score - actual[0].score) < 0.01
    assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=0.01)

    if fp16_capable:
        cached_info = install_aexrt_pipeline_cache(fp16_path, pipeline_cache)
        assert cached_info["dxil_cache_count"] >= 1
        assert cached_info["pso_cache_count"] >= 1
        cached = NativeCppYoloModel(fp16_path)
        try:
            cached.run(input_data, max_detections=4)
            assert cached.dxil_cache_hits >= 1
            assert cached.pso_cache_hits >= 1
        finally:
            cached.close()


def test_native_cpp_v11_stride2_position_pair_matches_kernel32_when_available(tmp_path):
    if not NativeD3D12Device.probe().available:
        return

    rng = np.random.default_rng(20260717)
    in_channels = out_channels = 256
    input_values = rng.uniform(0.75, 1.25, size=in_channels).astype("float32")
    weight = np.zeros((out_channels, in_channels, 3, 3), dtype="float32")
    for output_channel, target in ((0, 10.0), (1, 10.0), (2, 4.0), (3, 4.0)):
        kernel = rng.uniform(0.5, 1.5, size=in_channels).astype("float32")
        kernel *= target / float(np.dot(kernel, input_values))
        weight[output_channel, :, 1, 1] = kernel
    split0 = rng.uniform(0.5, 1.5, size=128).astype("float32")
    split1 = rng.uniform(0.5, 1.5, size=128).astype("float32")
    split0 *= 5.0 / float(np.dot(split0, input_values[:128]))
    split1 *= -3.5 / float(np.dot(split1, input_values[128:]))
    weight[255, :128, 1, 1] = split0
    weight[255, 128:, 1, 1] = split1

    graph = Graph("v11_stride2_position_pair_numeric")
    graph.input("images", TensorSpec((1, in_channels, 40, 40), "float32"))
    graph.const("w", weight)
    graph.const("b", np.zeros((out_channels,), dtype="float32"))
    graph.node(
        "Conv",
        "c",
        "images",
        "w",
        "b",
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1,
    )
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, out_channels, 400))
    graph.output("output0")

    optimized_path = tmp_path / "stride2_position_pair.aexrt"
    reference_path = tmp_path / "stride2_kernel32.aexrt"
    optimized_info = save_aexrt_engine(
        graph, optimized_path, classes=252, max_detections=4
    )
    reference_info = save_aexrt_engine(
        graph,
        reference_path,
        classes=252,
        max_detections=4,
        precision="fp32",
    )
    assert optimized_info["kernel_plan"][0][2:6] == (47, 2, 1, 3)
    assert reference_info["kernel_plan"][0][2] == 32

    input_data = np.zeros((1, in_channels, 40, 40), dtype="float32")
    input_data[0, :, 20, 20] = input_values
    reference_model = NativeCppYoloModel(reference_path)
    optimized_model = NativeCppYoloModel(optimized_path)
    try:
        reference = reference_model.run(input_data, max_detections=4)
        actual = optimized_model.run(input_data, max_detections=4)
        capabilities = optimized_model.capabilities
        fp16_capable = (
            capabilities["native_fp16_supported"]
            and capabilities["dxc_available"]
            and capabilities["highest_shader_model"] >= 0x62
        )
        assert optimized_model.tileflow_native_fp16_count == (1 if fp16_capable else 0)
        pipeline_cache = optimized_model.export_pipeline_cache()
    finally:
        reference_model.close()
        optimized_model.close()

    assert len(reference) == len(actual) == 1
    assert reference[0].class_id == actual[0].class_id == 251
    assert abs(reference[0].score - actual[0].score) < 0.02
    assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=0.02)

    if fp16_capable:
        cached_info = install_aexrt_pipeline_cache(optimized_path, pipeline_cache)
        assert cached_info["dxil_cache_count"] >= 1
        assert cached_info["pso_cache_count"] >= 1
        cached = NativeCppYoloModel(optimized_path)
        try:
            cached.run(input_data, max_detections=4)
            assert cached.dxil_cache_hits >= 1
            assert cached.pso_cache_hits >= 1
        finally:
            cached.close()


def test_native_cpp_position_pair_engine_rejects_malformed_abi(tmp_path):
    if not NativeD3D12Device.probe().available:
        return

    graph = Graph("position_pair_malformed_abi")
    graph.input("images", TensorSpec((1, 256, 40, 40), "float32"))
    graph.const("w", np.zeros((256, 256, 3, 3), dtype="float32"))
    graph.const("b", np.zeros((256,), dtype="float32"))
    graph.node(
        "Conv",
        "c",
        "images",
        "w",
        "b",
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1,
    )
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, 256, 400))
    graph.output("output0")

    valid_path = tmp_path / "position_pair_valid.aexrt"
    valid_info = save_aexrt_engine(
        graph, valid_path, classes=252, max_detections=4
    )
    assert valid_info["kernel_plan"][0][2:6] == (47, 2, 1, 3)
    data = valid_path.read_bytes()
    original_sections = {
        section_type: data[section["offset"] : section["offset"] + section["size"]]
        for section_type, section in valid_info["sections"].items()
    }

    for mutation in (
        "wrong_layout",
        "wrong_kernel_flags",
        "wrong_physical_flags",
        "near_shape",
    ):
        sections = dict(original_sections)
        if mutation == "wrong_layout":
            packed = bytearray(sections[engine_module.SECTION_PACKED_WEIGHTS])
            struct.pack_into(
                "<I",
                packed,
                12,
                engine_module.PACKED_LAYOUT_CONV3X3_OC4_FP16,
            )
            sections[engine_module.SECTION_PACKED_WEIGHTS] = bytes(packed)
        elif mutation == "wrong_kernel_flags":
            kernel_plan = bytearray(sections[engine_module.SECTION_KERNEL_PLAN])
            struct.pack_into(
                "<I",
                kernel_plan,
                28,
                engine_module.PLAN_FLAG_AUTHORITATIVE
                | engine_module.PLAN_FLAG_DXIL
                | engine_module.PLAN_FLAG_INPUT_FP16,
            )
            sections[engine_module.SECTION_KERNEL_PLAN] = bytes(kernel_plan)
        elif mutation == "wrong_physical_flags":
            physical = bytearray(
                sections[engine_module.SECTION_PHYSICAL_DISPATCH_PLAN]
            )
            struct.pack_into(
                "<I",
                physical,
                engine_module.PHYSICAL_DISPATCH_PLAN_HEADER_SIZE + 36,
                engine_module.PLAN_FLAG_AUTHORITATIVE
                | engine_module.PLAN_FLAG_DXIL
                | engine_module.PLAN_FLAG_OUTPUT_FP16,
            )
            sections[engine_module.SECTION_PHYSICAL_DISPATCH_PLAN] = bytes(physical)
        else:
            commands = bytearray(sections[engine_module.SECTION_COMMANDS])
            input_count = struct.unpack_from("<I", commands, 16)[0]
            params_offset = 8 + 24 + input_count * 4
            struct.pack_into("<I", commands, params_offset + 3 * 4, 39)
            assert struct.unpack_from("<I", commands, params_offset + 3 * 4)[0] == 39
            sections[engine_module.SECTION_COMMANDS] = bytes(commands)

        malformed_path = tmp_path / f"position_pair_{mutation}.aexrt"
        malformed_path.write_bytes(engine_module._build_container(sections))
        malformed_info = inspect_aexrt_engine(malformed_path)
        assert malformed_info["kernel_plan"][0][2] == 47
        if mutation == "wrong_layout":
            assert malformed_info["packed_layout_counts"] == {
                engine_module.PACKED_LAYOUT_CONV3X3_OC4_FP16: 1
            }
        elif mutation == "wrong_kernel_flags":
            assert malformed_info["kernel_plan"][0][5] == 7
        elif mutation == "wrong_physical_flags":
            owner = next(
                record
                for record in malformed_info["physical_dispatch_plan"]
                if record["logical_indices"] == (0,)
            )
            assert owner["flags"] == 11
        if mutation in {"wrong_layout", "near_shape"}:
            with pytest.raises(RuntimeError, match="failed to compile"):
                NativeCppYoloModel(malformed_path)
        else:
            # I/O flags are schema-valid plan bits.  Cross-section dtype/flag
            # coherence is intentionally checked by prepared execution rather
            # than by the binary section reader.
            malformed_model = NativeCppYoloModel(malformed_path)
            try:
                with pytest.raises(RuntimeError, match="run failed"):
                    malformed_model.run(
                        np.zeros(malformed_model.input_element_count, dtype=np.float32),
                        max_detections=1,
                    )
            finally:
                malformed_model.close()


@pytest.mark.parametrize("rejected_kernel", (48, 49, 50))
def test_native_cpp_rejects_unpromoted_v11_cmd9_kernel_ids(
    tmp_path, rejected_kernel
):
    if not NativeD3D12Device.probe().available:
        return

    graph = Graph(f"rejected_v11_cmd9_kernel_{rejected_kernel}")
    graph.input("images", TensorSpec((1, 128, 80, 80), "float32"))
    graph.const("w", np.zeros((128, 128, 3, 3), dtype="float32"))
    graph.const("b", np.zeros((128,), dtype="float32"))
    graph.node(
        "Conv",
        "c",
        "images",
        "w",
        "b",
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1,
    )
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, 128, 1600))
    graph.output("output0")

    valid_path = tmp_path / "stable_kernel33.aexrt"
    valid_info = save_aexrt_engine(
        graph, valid_path, classes=124, max_detections=4
    )
    assert valid_info["kernel_plan"][0][2] == 33
    data = valid_path.read_bytes()
    sections = {
        section_type: data[section["offset"] : section["offset"] + section["size"]]
        for section_type, section in valid_info["sections"].items()
    }

    commands = bytearray(sections[engine_module.SECTION_COMMANDS])
    struct.pack_into("<I", commands, 24, rejected_kernel)
    sections[engine_module.SECTION_COMMANDS] = bytes(commands)

    kernel_plan = bytearray(sections[engine_module.SECTION_KERNEL_PLAN])
    struct.pack_into("<I", kernel_plan, 16, rejected_kernel)
    sections[engine_module.SECTION_KERNEL_PLAN] = bytes(kernel_plan)

    physical = bytearray(sections[engine_module.SECTION_PHYSICAL_DISPATCH_PLAN])
    struct.pack_into(
        "<I",
        physical,
        engine_module.PHYSICAL_DISPATCH_PLAN_HEADER_SIZE + 28,
        rejected_kernel,
    )
    sections[engine_module.SECTION_PHYSICAL_DISPATCH_PLAN] = bytes(physical)

    rejected_path = tmp_path / f"rejected_kernel_{rejected_kernel}.aexrt"
    rejected_path.write_bytes(engine_module._build_container(sections))
    rejected_info = inspect_aexrt_engine(rejected_path)
    assert rejected_info["kernel_plan"][0][2] == rejected_kernel
    with pytest.raises(RuntimeError, match="failed to compile"):
        NativeCppYoloModel(rejected_path)


def test_native_cpp_v11_stride2_pos2_matches_oc8_plan_when_available(
    tmp_path, monkeypatch
):
    if not NativeD3D12Device.probe().available:
        return

    rng = np.random.default_rng(20260719)
    in_channels = out_channels = 128
    input_values = rng.uniform(0.75, 1.25, size=in_channels).astype("float32")
    weight = np.zeros((out_channels, in_channels, 3, 3), dtype="float32")
    for output_channel, target in enumerate((10.0, 10.0, 4.0, 4.0, 1.5)):
        kernel = rng.uniform(0.5, 1.5, size=in_channels).astype("float32")
        kernel *= target / float(np.dot(kernel, input_values))
        weight[output_channel, :, 1, 1] = kernel

    graph = Graph("v11_stride2_pos2_numeric")
    graph.input("images", TensorSpec((1, in_channels, 80, 80), "float32"))
    graph.const("w", weight)
    graph.const("b", np.zeros((out_channels,), dtype="float32"))
    graph.node(
        "Conv",
        "c",
        "images",
        "w",
        "b",
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1,
    )
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, out_channels, 1600))
    graph.output("output0")

    optimized_path = tmp_path / "stride2_pos2.aexrt"
    optimized_info = save_aexrt_engine(
        graph, optimized_path, classes=out_channels - 4, max_detections=4
    )
    assert optimized_info["kernel_plan"][0][2] == 33

    stable_conv_algorithm = engine_module._stable_conv_algorithm

    def oc8_plan(params, *, silu):
        desc = tuple(int(value) for value in params[:16])
        if silu and desc == (
            1, 128, 80, 80, 128, 40, 40, 3, 3, 2, 2, 1, 1, 1, 1, 1
        ):
            return 34
        return stable_conv_algorithm(params, silu=silu)

    monkeypatch.setattr(engine_module, "_stable_conv_algorithm", oc8_plan)
    reference_path = tmp_path / "stride2_oc8.aexrt"
    reference_info = save_aexrt_engine(
        graph, reference_path, classes=out_channels - 4, max_detections=4
    )
    assert reference_info["kernel_plan"][0][2] == 34

    input_data = np.zeros((1, in_channels, 80, 80), dtype="float32")
    input_data[0, :, 40, 40] = input_values
    reference_model = NativeCppYoloModel(reference_path)
    optimized_model = NativeCppYoloModel(optimized_path)
    try:
        reference = reference_model.run(input_data, max_detections=4)
        actual = optimized_model.run(input_data, max_detections=4)
    finally:
        optimized_model.close()
        reference_model.close()

    assert len(reference) == len(actual) == 1
    assert reference[0].class_id == actual[0].class_id
    assert abs(reference[0].score - actual[0].score) < 1e-5
    assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=1e-5)


def test_native_cpp_fp16_conv1x1_oc8_pos2_matches_sm5_when_available(tmp_path, monkeypatch):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(20260716)
    dim = 40
    channels = 64
    input_values = rng.uniform(0.75, 1.25, size=channels).astype("float32")
    weight = np.zeros((channels, channels, 1, 1), dtype="float32")
    for output_channel, target in enumerate((10.0, 10.0, 4.0, 4.0, 1.5)):
        kernel = rng.uniform(0.5, 1.5, size=channels).astype("float32")
        kernel *= target / float(np.dot(kernel, input_values))
        weight[output_channel, :, 0, 0] = kernel

    graph = Graph("native_fp16_conv1x1_oc8_pos2_numeric")
    graph.input("images", TensorSpec((1, channels, dim, dim), "float32"))
    graph.const("w", weight)
    graph.const("b", np.zeros((channels,), dtype="float32"))
    graph.node(
        "Conv", "c", "images", "w", "b",
        strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, channels, dim * dim))
    graph.output("output0")

    optimized_path = tmp_path / "native_fp16_conv1x1.aexrt"
    optimized_info = save_aexrt_engine(
        graph, optimized_path, classes=channels - 4, conf_threshold=0.25, max_detections=4
    )
    assert optimized_info["kernel_plan"][0][2:6] == (45, 2, 1, 3)
    assert optimized_info["packed_layout_counts"] == {4: 1}

    monkeypatch.setattr(engine_module, "_uses_native_fp16_conv1x1", lambda params: False)
    reference_path = tmp_path / "sm5_conv1x1.aexrt"
    reference_info = save_aexrt_engine(
        graph, reference_path, classes=channels - 4, conf_threshold=0.25, max_detections=4
    )
    assert reference_info["kernel_plan"][0][2] == 1
    assert reference_info["packed_weight_count"] == 0

    input_data = np.zeros((1, channels, dim, dim), dtype="float32")
    input_data[0, :, dim // 2, dim // 2] = input_values
    reference_model = NativeCppYoloModel(reference_path)
    optimized_model = NativeCppYoloModel(optimized_path)
    cache = b""
    try:
        capabilities = optimized_model.capabilities
        fp16_capable = (
            capabilities["native_fp16_supported"]
            and capabilities["dxc_available"]
            and capabilities["highest_shader_model"] >= 0x62
        )
        reference = reference_model.run(input_data, max_detections=4)
        actual = optimized_model.run(input_data, max_detections=4)
        assert optimized_model.tileflow_native_fp16_count == (1 if fp16_capable else 0)
        if fp16_capable:
            cache = optimized_model.export_pipeline_cache()
    finally:
        optimized_model.close()
        reference_model.close()

    assert len(reference) == len(actual) == 1
    assert reference[0].class_id == actual[0].class_id == 0
    assert abs(reference[0].score - actual[0].score) < 0.01
    assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=0.02)

    if cache:
        cached_info = install_aexrt_pipeline_cache(optimized_path, cache)
        assert cached_info["dxil_cache_count"] >= 1
        assert cached_info["pso_cache_count"] >= 1
        cached = NativeCppYoloModel(optimized_path)
        try:
            cached.run(input_data, max_detections=4)
            assert cached.dxil_cache_hits >= 1
            assert cached.pso_cache_hits >= 1
        finally:
            cached.close()


@pytest.mark.parametrize("mode", ("direct", "residual", "late"))
def test_native_cpp_fp16_concat_conv1x1_oc8_pos2_matches_sm5_when_available(
    tmp_path, monkeypatch, mode
):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    fused_residual = mode == "residual"
    late_concat = mode == "late"
    rng = np.random.default_rng(20260717 + (1 if fused_residual else 2 if late_concat else 0))
    dim = 20
    input_channels = 8
    branch_channels = 32
    output_channels = 64
    input_values = rng.uniform(0.75, 1.25, size=input_channels).astype("float32")

    def branch_weight(target):
        result = np.zeros((branch_channels, input_channels, 1, 1), dtype="float32")
        for output_channel in range(branch_channels):
            kernel = rng.uniform(0.5, 1.5, size=input_channels).astype("float32")
            kernel *= target / float(np.dot(kernel, input_values))
            result[output_channel, :, 0, 0] = kernel
        return result

    graph = Graph(f"native_fp16_concat_conv1x1_{mode}")
    graph.input("images", TensorSpec((1, input_channels, dim, dim), "float32"))
    if fused_residual:
        graph.const("wp", branch_weight(0.5))
        graph.const("bp", np.zeros((branch_channels,), dtype="float32"))
        graph.const("wr", branch_weight(0.5))
        graph.const("br", np.zeros((branch_channels,), dtype="float32"))
        graph.node(
            "Conv", "primary", "images", "wp", "bp",
            strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
        )
        graph.node(
            "Conv", "residual", "images", "wr", "br",
            strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
        )
        graph.add("branch0", "primary", "residual")
    else:
        graph.const("w0", branch_weight(1.0))
        graph.const("b0", np.zeros((branch_channels,), dtype="float32"))
        graph.node(
            "Conv", "branch0", "images", "w0", "b0",
            strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
        )
    graph.const("w1", branch_weight(1.0))
    graph.const("b1", np.zeros((branch_channels,), dtype="float32"))
    graph.node(
        "Conv", "branch1", "images", "w1", "b1",
        strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
    )
    graph.node("Concat", "cat", "branch0", "branch1", axis=1)
    if late_concat:
        graph.relu("unrelated", "branch1")

    final_weight = np.zeros((output_channels, branch_channels * 2, 1, 1), dtype="float32")
    for output_channel, target in enumerate((10.0, 10.0, 4.0, 4.0, 1.5)):
        kernel = rng.uniform(0.5, 1.5, size=branch_channels * 2).astype("float32")
        kernel *= target / float(kernel.sum())
        final_weight[output_channel, :, 0, 0] = kernel
    graph.const("wf", final_weight)
    graph.const("bf", np.zeros((output_channels,), dtype="float32"))
    graph.node(
        "Conv", "cf", "cat", "wf", "bf",
        strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
    )
    graph.sigmoid("sf", "cf")
    graph.mul("act", "cf", "sf")
    graph.reshape("output0", "act", (1, output_channels, dim * dim))
    graph.output("output0")

    optimized_path = tmp_path / f"native_concat_{mode}.aexrt"
    if late_concat:
        monkeypatch.setattr(
            engine_module,
            "_uses_native_fp16_conv1x1",
            lambda params: len(params) >= 5 and int(params[1]) == int(params[4]) == 64,
        )
    else:
        monkeypatch.setattr(
            engine_module, "_uses_native_fp16_concat_conv1x1", lambda command: True
        )
    optimized_info = save_aexrt_engine(
        graph, optimized_path, classes=output_channels - 4, conf_threshold=0.25, max_detections=4
    )
    if late_concat:
        optimized_plan = [
            record for record in optimized_info["kernel_plan"] if record[1] == 2 and record[2] == 45
        ]
        late_groups = [group for group in optimized_info["fusion_plan"] if group["kind"] == 3]
        assert len(optimized_plan) == len(late_groups) == 1
        assert (late_groups[0]["kernel"], late_groups[0]["flags"]) == (3, 7)
    else:
        optimized_plan = [record for record in optimized_info["kernel_plan"] if record[1] == 3]
        assert len(optimized_plan) == 1
        assert optimized_plan[0][2:4] == (3, 2)
        assert optimized_plan[0][5] == 7
    assert optimized_info["packed_layout_counts"] == {4: 1}

    if late_concat:
        monkeypatch.setattr(engine_module, "_uses_native_fp16_conv1x1", lambda params: False)
    else:
        monkeypatch.setattr(
            engine_module, "_uses_native_fp16_concat_conv1x1", lambda command: False
        )
    reference_path = tmp_path / f"sm5_concat_{mode}.aexrt"
    reference_info = save_aexrt_engine(
        graph, reference_path, classes=output_channels - 4, conf_threshold=0.25, max_detections=4
    )
    if late_concat:
        reference_plan = [record for record in reference_info["kernel_plan"] if record[1] == 2]
        assert reference_plan[-1][2] == 1
    else:
        reference_plan = [record for record in reference_info["kernel_plan"] if record[1] == 3]
        assert len(reference_plan) == 1
        assert reference_plan[0][2] == 1

    input_data = np.zeros((1, input_channels, dim, dim), dtype="float32")
    input_data[0, :, dim // 2, dim // 2] = input_values
    reference_model = NativeCppYoloModel(reference_path)
    optimized_model = NativeCppYoloModel(optimized_path)
    cache = b""
    try:
        capabilities = optimized_model.capabilities
        fp16_capable = (
            capabilities["native_fp16_supported"]
            and capabilities["dxc_available"]
            and capabilities["highest_shader_model"] >= 0x62
        )
        reference = reference_model.run(input_data, max_detections=4)
        actual = optimized_model.run(input_data, max_detections=4)
        if fused_residual:
            assert optimized_model.concat_residual_conv1x1_fusion_count == 1
            assert reference_model.concat_residual_conv1x1_fusion_count == 1
        if fp16_capable:
            cache = optimized_model.export_pipeline_cache()
    finally:
        optimized_model.close()
        reference_model.close()

    assert len(reference) == len(actual) == 1
    assert reference[0].class_id == actual[0].class_id == 0
    assert abs(reference[0].score - actual[0].score) < 0.015
    assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=0.03)
    if cache and not fused_residual:
        cached_info = install_aexrt_pipeline_cache(optimized_path, cache)
        assert cached_info["dxil_cache_count"] >= 1
        assert cached_info["pso_cache_count"] >= 1


def test_native_cpp_position_owned_winograd_residual_concat_matches_separate_path(
    tmp_path, monkeypatch
):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    channels = 128
    dim = 10
    graph = Graph("position_owned_winograd_residual_concat")
    graph.input("images", TensorSpec((1, channels, dim, dim), "float32"))
    zero_1x1 = np.zeros((channels, channels, 1, 1), dtype="float32")
    zero_bias = np.zeros((channels,), dtype="float32")
    for name in ("branch", "pre"):
        graph.const(f"w_{name}", zero_1x1)
        graph.const(f"b_{name}", zero_bias)
        graph.node(
            "Conv", f"c_{name}", "images", f"w_{name}", f"b_{name}",
            strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
        )
        graph.sigmoid(f"s_{name}", f"c_{name}")
        graph.mul(name, f"c_{name}", f"s_{name}")

    graph.const(
        "w_tail", np.zeros((channels, channels, 3, 3), dtype="float32")
    )
    graph.const("b_tail", zero_bias)
    graph.node(
        "Conv", "c_tail", "pre", "w_tail", "b_tail",
        strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s_tail", "c_tail")
    graph.mul("tail", "c_tail", "s_tail")
    graph.add("residual", "images", "tail")
    graph.node("Concat", "cat", "residual", "branch", axis=1)

    final_weight = np.zeros((256, 256, 1, 1), dtype="float32")
    for output_channel, target in enumerate((10.0, 10.0, 4.0, 4.0, 1.5)):
        final_weight[output_channel, :channels, 0, 0] = target / channels
    graph.const("w_final", final_weight)
    graph.const("b_final", np.zeros((256,), dtype="float32"))
    graph.node(
        "Conv", "c_final", "cat", "w_final", "b_final",
        strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s_final", "c_final")
    graph.mul("act", "c_final", "s_final")
    graph.reshape("output0", "act", (1, 256, dim * dim))
    graph.output("output0")

    monkeypatch.setenv("AEXRT_NATIVE_D3D12_PROFILE_REPLAY", "1")
    optimized_path = tmp_path / "position_owned.aexrt"
    optimized_info = save_aexrt_engine(
        graph, optimized_path, classes=252, conf_threshold=0.25, max_detections=4
    )
    owners = [
        record
        for record in optimized_info["physical_dispatch_plan"]
        if record["fusion_kind"]
        == engine_module.FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2
    ]
    assert len(owners) == 1
    assert owners[0]["logical_indices"] == (2, 3, 4)
    assert optimized_info["packed_layout_counts"] == {3: 1, 4: 1}

    monkeypatch.setattr(
        engine_module,
        "_position_owned_winograd_residual_cv2_groups",
        lambda replay, fusion_plan: [],
    )
    reference_path = tmp_path / "position_separate.aexrt"
    reference_info = save_aexrt_engine(
        graph, reference_path, classes=252, conf_threshold=0.25, max_detections=4
    )
    assert not [
        record
        for record in reference_info["physical_dispatch_plan"]
        if record["fusion_kind"]
        == engine_module.FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2
    ]

    input_data = np.zeros((1, channels, dim, dim), dtype="float32")
    input_data[0, :, dim // 2, dim // 2] = 1.0
    reference_model = NativeCppYoloModel(reference_path)
    optimized_model = NativeCppYoloModel(optimized_path)
    cache = b""
    try:
        reference = reference_model.run(input_data, max_detections=4)
        actual = optimized_model.run(input_data, max_detections=4)
        labels = [label for label, _ in optimized_model.profile_events()]
        assert sum("POSITION_WINOGRAD_RESIDUAL_CONCAT1X1" in label for label in labels) == 1
        if (
            optimized_model.capabilities["native_fp16_supported"]
            and optimized_model.capabilities["dxc_available"]
            and optimized_model.capabilities["highest_shader_model"] >= 0x62
        ):
            cache = optimized_model.export_pipeline_cache()
    finally:
        optimized_model.close()
        reference_model.close()

    assert len(reference) == len(actual) == 1
    assert reference[0].class_id == actual[0].class_id == 0
    assert abs(reference[0].score - actual[0].score) < 0.002
    assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=0.002)

    if cache:
        cached_info = install_aexrt_pipeline_cache(optimized_path, cache)
        assert cached_info["dxil_cache_count"] >= 1
        assert cached_info["pso_cache_count"] >= 1
        cached = NativeCppYoloModel(optimized_path)
        try:
            cached.run(input_data, max_detections=4)
            assert cached.dxil_cache_hits >= 1
            assert cached.pso_cache_hits >= 1
        finally:
            cached.close()


def test_native_cpp_position_winograd_residual_branch_concat_matches_separate_path(
    tmp_path, monkeypatch
):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    channels = 128
    input_channels = 256
    dim = 20
    graph = Graph("position_winograd_residual_branch_concat")
    graph.input("images", TensorSpec((1, input_channels, dim, dim), "float32"))

    identity = np.zeros((channels, input_channels, 1, 1), dtype="float32")
    identity[np.arange(channels), np.arange(channels), 0, 0] = 1.0
    graph.const("w_residual", identity)
    graph.const("b_residual", np.zeros((channels,), dtype="float32"))
    graph.node(
        "Conv", "c_residual", "images", "w_residual", "b_residual",
        strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s_residual", "c_residual")
    graph.mul("residual", "c_residual", "s_residual")

    identity_128 = np.zeros((channels, channels, 1, 1), dtype="float32")
    identity_128[np.arange(channels), np.arange(channels), 0, 0] = 1.0
    graph.const("w_pre", identity_128)
    graph.const("b_pre", np.zeros((channels,), dtype="float32"))
    graph.node(
        "Conv", "c_pre", "residual", "w_pre", "b_pre",
        strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s_pre", "c_pre")
    graph.mul("pre", "c_pre", "s_pre")

    tail_weight = np.zeros((channels, channels, 3, 3), dtype="float32")
    tail_weight[np.arange(channels), np.arange(channels), 1, 1] = 0.125
    graph.const("w_tail", tail_weight)
    graph.const("b_tail", np.zeros((channels,), dtype="float32"))
    graph.node(
        "Conv", "c_tail", "pre", "w_tail", "b_tail",
        strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s_tail", "c_tail")
    graph.mul("tail", "c_tail", "s_tail")
    graph.add("sum", "residual", "tail")

    branch_weight = np.zeros((channels, input_channels, 1, 1), dtype="float32")
    branch_weight[np.arange(channels), channels + np.arange(channels), 0, 0] = 0.5
    graph.const("w_branch", branch_weight)
    graph.const(
        "b_branch", np.linspace(-0.025, 0.025, channels, dtype="float32")
    )
    graph.node(
        "Conv", "c_branch", "images", "w_branch", "b_branch",
        strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s_branch", "c_branch")
    graph.mul("branch", "c_branch", "s_branch")
    graph.node("Concat", "cat", "sum", "branch", axis=1)

    final_weight = np.zeros((256, 256, 1, 1), dtype="float32")
    for output_channel, target in enumerate((12.0, 12.0, 5.0, 5.0, 2.0)):
        final_weight[output_channel, :channels, 0, 0] = target / (2 * channels)
        final_weight[output_channel, channels:, 0, 0] = target / (2 * channels)
    graph.const("w_final", final_weight)
    graph.const("b_final", np.zeros((256,), dtype="float32"))
    graph.node(
        "Conv", "c_final", "cat", "w_final", "b_final",
        strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s_final", "c_final")
    graph.mul("act", "c_final", "s_final")
    graph.reshape("output0", "act", (1, 256, dim * dim))
    graph.output("output0")

    monkeypatch.setenv("AEXRT_NATIVE_D3D12_PROFILE_REPLAY", "1")
    optimized_path = tmp_path / "position_owned_branch.aexrt"
    optimized_info = save_aexrt_engine(
        graph, optimized_path, classes=252, conf_threshold=0.25, max_detections=4
    )
    owners = [
        record
        for record in optimized_info["physical_dispatch_plan"]
        if record["fusion_kind"]
        == engine_module.FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2
        and len(record["logical_indices"]) == 4
    ]
    assert len(owners) == 1
    assert owners[0]["logical_indices"] == (2, 3, 4, 5)
    assert owners[0]["execution_index"] == 2

    monkeypatch.setattr(
        engine_module,
        "_position_owned_winograd_residual_branch_cv2_groups",
        lambda replay: [],
    )
    reference_path = tmp_path / "position_branch_separate.aexrt"
    reference_info = save_aexrt_engine(
        graph, reference_path, classes=252, conf_threshold=0.25, max_detections=4
    )
    assert not [
        record
        for record in reference_info["physical_dispatch_plan"]
        if record["fusion_kind"]
        == engine_module.FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2
    ]

    input_data = np.zeros((1, input_channels, dim, dim), dtype="float32")
    input_data[0, :channels, dim // 2, dim // 2] = 1.0
    input_data[0, channels:, dim // 2, dim // 2] = 0.75
    reference_model = NativeCppYoloModel(reference_path)
    optimized_model = NativeCppYoloModel(optimized_path)
    cache = b""
    try:
        reference = reference_model.run(input_data, max_detections=4)
        actual = optimized_model.run(input_data, max_detections=4)
        labels = [label for label, _ in optimized_model.profile_events()]
        assert sum(
            "POSITION_WINOGRAD_RESIDUAL_BRANCH_CONCAT1X1" in label
            for label in labels
        ) == 1
        if (
            optimized_model.capabilities["native_fp16_supported"]
            and optimized_model.capabilities["dxc_available"]
            and optimized_model.capabilities["highest_shader_model"] >= 0x62
        ):
            cache = optimized_model.export_pipeline_cache()
    finally:
        optimized_model.close()
        reference_model.close()

    assert len(reference) == len(actual) == 1
    assert reference[0].class_id == actual[0].class_id == 0
    assert abs(reference[0].score - actual[0].score) < 0.01
    assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=0.03)

    if cache:
        cached_info = install_aexrt_pipeline_cache(optimized_path, cache)
        assert cached_info["dxil_cache_count"] >= 1
        assert cached_info["pso_cache_count"] >= 1
        cached = NativeCppYoloModel(optimized_path)
        try:
            cached.run(input_data, max_detections=4)
            assert cached.dxil_cache_hits >= 1
            assert cached.pso_cache_hits >= 1
        finally:
            cached.close()


def test_native_cpp_fp16_pos2_dense_edges_match_fp32_pos2_when_available(tmp_path, monkeypatch):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(20260714)
    dim = 40
    channels = 64
    graph = Graph("native_fp16_pos2_dense_edges")
    graph.input("images", TensorSpec((1, channels, dim, dim), "float32"))
    input_values = rng.uniform(0.75, 1.25, size=channels).astype("float32")
    weight = rng.uniform(-1.0e-4, 1.0e-4, size=(channels, channels, 3, 3)).astype("float32")
    signs = np.where(np.arange(channels) % 2 == 0, 1.0, -1.0).astype("float32")
    targets = (10.0, 10.0, 4.0, 4.0, 1.5)
    for output_channel, target in enumerate(targets):
        center = rng.uniform(0.5, 1.5, size=channels).astype("float32")
        center *= target / float(np.dot(center, input_values))
        weight[output_channel, :, 1, 1] = center
        for ky in range(3):
            for kx in range(3):
                if (ky, kx) == (1, 1):
                    continue
                dense = rng.uniform(0.5, 1.5, size=channels).astype("float32") * signs
                dense *= 0.04 / float(np.sum(np.abs(dense * input_values)))
                weight[output_channel, :, ky, kx] = dense
                signs = -signs

    graph.const("w", weight)
    graph.const("b", np.zeros((channels,), dtype="float32"))
    graph.node(
        "Conv", "c", "images", "w", "b",
        strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, channels, dim * dim))
    graph.output("output0")

    optimized_path = tmp_path / "fp16_pos2.aexrt"
    optimized_info = save_aexrt_engine(
        graph, optimized_path, classes=channels - 4, conf_threshold=0.25, max_detections=4
    )
    assert optimized_info["kernel_plan"][0][2:6] == (44, 2, 1, 3)

    monkeypatch.setattr(
        engine_module, "_uses_native_fp16_pos2_40x40_64x64", lambda params: False
    )
    reference_path = tmp_path / "fp32_pos2.aexrt"
    reference_info = save_aexrt_engine(
        graph, reference_path, classes=channels - 4, conf_threshold=0.25, max_detections=4
    )
    assert reference_info["kernel_plan"][0][2] == 39

    reference_model = NativeCppYoloModel(reference_path)
    optimized_model = NativeCppYoloModel(optimized_path)
    try:
        capabilities = optimized_model.capabilities
        fp16_capable = (
            capabilities["native_fp16_supported"]
            and capabilities["dxc_available"]
            and capabilities["highest_shader_model"] >= 0x62
        )
        edge_positions = (
            (0, 0), (0, 1), (0, 38), (0, 39),
            (1, 0), (1, 1), (20, 38), (20, 39),
            (39, 0), (39, 39),
        )
        for y, x in edge_positions:
            input_data = np.zeros((1, channels, dim, dim), dtype="float32")
            input_data[0, :, y, x] = input_values
            reference = reference_model.run(input_data, max_detections=4)
            actual = optimized_model.run(input_data, max_detections=4)
            assert len(reference) == len(actual) == 1, (y, x, reference, actual)
            assert reference[0].class_id == actual[0].class_id == 0
            assert abs(reference[0].score - actual[0].score) < 0.01
            assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=0.02)
        assert optimized_model.tileflow_native_fp16_count == (1 if fp16_capable else 0)
    finally:
        optimized_model.close()
        reference_model.close()


def test_native_cpp_fp16_paired_winograd_dense_edges_match_two_id40_when_available(
    tmp_path, monkeypatch
):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(20260715)
    dim = 10
    in_channels = 256
    out_channels = 64
    input_values = rng.uniform(0.75, 1.25, size=in_channels).astype("float32")
    signs = np.where(np.arange(in_channels) % 2 == 0, 1.0, -1.0).astype("float32")

    def dense_weight(targets):
        nonlocal signs
        weight = rng.uniform(
            -1.0e-4, 1.0e-4, size=(out_channels, in_channels, 3, 3)
        ).astype("float32")
        for output_channel, target in targets.items():
            center = rng.uniform(0.5, 1.5, size=in_channels).astype("float32")
            center *= target / float(np.dot(center, input_values))
            weight[output_channel, :, 1, 1] = center
            for ky in range(3):
                for kx in range(3):
                    if (ky, kx) == (1, 1):
                        continue
                    dense = rng.uniform(0.5, 1.5, size=in_channels).astype("float32") * signs
                    dense *= 0.04 / float(np.sum(np.abs(dense * input_values)))
                    weight[output_channel, :, ky, kx] = dense
                    signs = -signs
        return weight

    graph = Graph("native_fp16_paired_winograd_dense_edges")
    graph.input("images", TensorSpec((1, in_channels, dim, dim), "float32"))
    graph.const("w0", dense_weight({0: 10.0, 1: 10.0, 2: 4.0, 3: 4.0}))
    graph.const("b0", np.zeros((out_channels,), dtype="float32"))
    graph.const("w1", dense_weight({0: 1.5}))
    graph.const("b1", np.zeros((out_channels,), dtype="float32"))
    for suffix in ("0", "1"):
        graph.node(
            "Conv", f"c{suffix}", "images", f"w{suffix}", f"b{suffix}",
            strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1,
        )
        graph.sigmoid(f"s{suffix}", f"c{suffix}")
        graph.mul(f"a{suffix}", f"c{suffix}", f"s{suffix}")
    graph.node("Concat", "cat", "a0", "a1", axis=1)
    graph.reshape("output0", "cat", (1, 128, dim * dim))
    graph.output("output0")

    optimized_path = tmp_path / "paired_winograd.aexrt"
    optimized_info = save_aexrt_engine(
        graph, optimized_path, classes=124, conf_threshold=0.25, max_detections=4
    )
    paired_groups = [group for group in optimized_info["fusion_plan"] if group["kind"] == 1]
    assert len(paired_groups) == 1
    assert optimized_info["packed_layout_counts"] == {3: 2}

    build_fusion_plan = engine_module._build_fusion_plan

    def without_paired_winograd(*args, **kwargs):
        return [group for group in build_fusion_plan(*args, **kwargs) if group["kind"] != 1]

    monkeypatch.setattr(engine_module, "_build_fusion_plan", without_paired_winograd)
    reference_path = tmp_path / "two_id40.aexrt"
    reference_info = save_aexrt_engine(
        graph, reference_path, classes=124, conf_threshold=0.25, max_detections=4
    )
    assert not [group for group in reference_info["fusion_plan"] if group["kind"] == 1]
    assert reference_info["kernel_plan"][0][2] == reference_info["kernel_plan"][1][2] == 40

    reference_model = NativeCppYoloModel(reference_path)
    optimized_model = NativeCppYoloModel(optimized_path)
    cache = b""
    try:
        capabilities = optimized_model.capabilities
        fp16_capable = (
            capabilities["native_fp16_supported"]
            and capabilities["dxc_available"]
            and capabilities["highest_shader_model"] >= 0x62
        )
        edge_positions = ((0, 0), (0, 1), (1, 0), (8, 8), (8, 9), (9, 8), (9, 9))
        for y, x in edge_positions:
            input_data = np.zeros((1, in_channels, dim, dim), dtype="float32")
            input_data[0, :, y, x] = input_values
            reference = reference_model.run(input_data, max_detections=4)
            actual = optimized_model.run(input_data, max_detections=4)
            assert len(reference) == len(actual) == 1, (y, x, reference, actual)
            assert reference[0].class_id == actual[0].class_id == 60
            assert abs(reference[0].score - actual[0].score) < 0.03
            assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=0.05)
        assert optimized_model.paired_conv3x3_fusion_count == 1
        if fp16_capable:
            cache = optimized_model.export_pipeline_cache()
    finally:
        optimized_model.close()
        reference_model.close()

    if cache:
        cached_info = install_aexrt_pipeline_cache(optimized_path, cache)
        assert cached_info["dxil_cache_count"] >= 1
        assert cached_info["pso_cache_count"] >= 1
        cached = NativeCppYoloModel(optimized_path)
        try:
            cached.run(np.zeros((1, in_channels, dim, dim), dtype="float32"), max_detections=4)
            assert cached.dxil_cache_hits >= 1
            assert cached.pso_cache_hits >= 1
        finally:
            cached.close()


def test_native_cpp_fp16_winograd_c3_residual_tail_matches_standalone_and_reloads_cache_when_available(
    tmp_path, monkeypatch
):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(20260718)
    dim = 20
    channels = 64
    input_values = np.zeros((channels,), dtype="float32")
    input_values[:5] = (10.0, 10.0, 4.0, 4.0, 0.9)

    first_weight = np.zeros((channels, channels, 1, 1), dtype="float32")
    first_weight[np.arange(channels), np.arange(channels), 0, 0] = 0.25
    first_linear = input_values * 0.25
    first_features = first_linear / (1.0 + np.exp(-first_linear))

    tail_weight = np.zeros((channels, channels, 3, 3), dtype="float32")
    for output_channel, target in enumerate((0.5, 0.5, 0.2, 0.2, 0.3)):
        center = rng.uniform(0.5, 1.5, size=5).astype("float32")
        center *= target / float(np.dot(center, first_features[:5]))
        tail_weight[output_channel, :5, 1, 1] = center
        for ky in range(3):
            for kx in range(3):
                if (ky, kx) != (1, 1):
                    tail_weight[output_channel, :5, ky, kx] = rng.uniform(
                        -0.002, 0.002, size=5
                    )

    graph = Graph("native_fp16_winograd_c3_residual_tail_numeric")
    graph.input("images", TensorSpec((1, channels, dim, dim), "float32"))
    graph.const("w0", first_weight)
    graph.const("b0", np.zeros((channels,), dtype="float32"))
    graph.const("w1", tail_weight)
    graph.const("b1", np.zeros((channels,), dtype="float32"))
    graph.node(
        "Conv", "c0", "images", "w0", "b0",
        strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s0", "c0")
    graph.mul("a0", "c0", "s0")
    graph.node(
        "Conv", "c1", "a0", "w1", "b1",
        strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1,
    )
    graph.sigmoid("s1", "c1")
    graph.mul("a1", "c1", "s1")
    graph.add("residual", "images", "a1")
    graph.reshape("output0", "residual", (1, channels, dim * dim))
    graph.output("output0")

    # Keep the front 1x1 on SM5 so the native FP16 count isolates the tail ID40.
    monkeypatch.setattr(engine_module, "_uses_native_fp16_conv1x1", lambda params: False)
    optimized_path = tmp_path / "fp16_winograd_c3_tail.aexrt"
    optimized_info = save_aexrt_engine(
        graph,
        optimized_path,
        classes=channels - 4,
        conf_threshold=0.25,
        max_detections=4,
    )
    tail_groups = [group for group in optimized_info["fusion_plan"] if group["kind"] == 2]
    assert tail_groups == [
        {
            "kind": 2,
            "start": 1,
            "end": 2,
            "precision": 2,
            "kernel": 40,
            "flags": 3,
            "aux0": 0,
            "aux1": 0,
        }
    ]
    assert optimized_info["kernel_plan"][0][2] == 1
    tail_plan = optimized_info["kernel_plan"][1]
    assert tail_plan[2] == 40
    assert tail_plan[3] == 2
    assert tail_plan[4] != (1 << 32) - 1
    assert tail_plan[5] == 3
    assert optimized_info["packed_layout_counts"] == {3: 1}

    build_fusion_plan = engine_module._build_fusion_plan

    def without_c3_residual_tail(*args, **kwargs):
        return [group for group in build_fusion_plan(*args, **kwargs) if group["kind"] != 2]

    monkeypatch.setattr(engine_module, "_build_fusion_plan", without_c3_residual_tail)
    reference_path = tmp_path / "fp16_winograd_standalone_add.aexrt"
    reference_info = save_aexrt_engine(
        graph,
        reference_path,
        classes=channels - 4,
        conf_threshold=0.25,
        max_detections=4,
    )
    assert not [group for group in reference_info["fusion_plan"] if group["kind"] == 2]
    assert reference_info["kernel_plan"][1][2] == 40
    assert reference_info["packed_layout_counts"] == {3: 1}

    input_data = np.zeros((1, channels, dim, dim), dtype="float32")
    input_data[0, :, dim // 2, dim // 2] = input_values
    reference_model = NativeCppYoloModel(reference_path)
    optimized_model = NativeCppYoloModel(optimized_path)
    cache = b""
    try:
        capabilities = optimized_model.capabilities
        fp16_capable = (
            capabilities["native_fp16_supported"]
            and capabilities["dxc_available"]
            and capabilities["highest_shader_model"] >= 0x62
        )
        reference = reference_model.run(input_data, max_detections=4)
        actual = optimized_model.run(input_data, max_detections=4)
        assert optimized_model.c2f_tail_residual_fusion_count == 1
        assert reference_model.c2f_tail_residual_fusion_count == 0
        assert optimized_model.tileflow_native_fp16_count == (1 if fp16_capable else 0)
        assert reference_model.tileflow_native_fp16_count == (1 if fp16_capable else 0)
        if fp16_capable:
            cache = optimized_model.export_pipeline_cache()
    finally:
        optimized_model.close()
        reference_model.close()

    assert len(reference) == len(actual) == 1
    assert reference[0].class_id == actual[0].class_id == 0
    assert abs(reference[0].score - actual[0].score) < 0.01
    assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=0.02)

    if cache:
        cached_info = install_aexrt_pipeline_cache(optimized_path, cache)
        assert cached_info["dxil_cache_count"] >= 1
        assert cached_info["pso_cache_count"] >= 1
        cached = NativeCppYoloModel(optimized_path)
        try:
            cached_result = cached.run(input_data, max_detections=4)
            assert cached.c2f_tail_residual_fusion_count == 1
            assert cached.tileflow_native_fp16_count == 1
            assert cached.dxil_cache_hits >= 1
            assert cached.pso_cache_hits >= 1
        finally:
            cached.close()
        assert len(cached_result) == len(actual) == 1
        assert cached_result[0].class_id == actual[0].class_id
        assert abs(cached_result[0].score - actual[0].score) < 1e-6
        assert_close(np.asarray(cached_result[0].xyxy), np.asarray(actual[0].xyxy), tol=1e-6)


@pytest.mark.parametrize(
    ("dim", "in_channels", "out_channels", "optimized_algorithm", "reference_algorithm"),
    (
        (20, 256, 64, 40, 7),
        (20, 64, 64, 40, 10),
        (10, 128, 128, 40, 11),
        (10, 256, 256, 46, 8),
        (10, 64, 64, 40, 2),
        (40, 64, 64, 44, 39),
    ),
)
def test_native_cpp_hotspot_plan_matches_reference_when_available(
    tmp_path, monkeypatch, dim, in_channels, out_channels, optimized_algorithm, reference_algorithm
):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(20260713)
    graph = Graph(f"winograd_{dim}x{dim}_{in_channels}x{out_channels}_numeric")
    graph.input("images", TensorSpec((1, in_channels, dim, dim), "float32"))
    weight = np.zeros((out_channels, in_channels, 3, 3), dtype="float32")
    input_values = rng.uniform(0.75, 1.25, size=in_channels).astype("float32")
    for output_channel, target in enumerate((10.0, 10.0, 4.0, 4.0, 1.5)):
        kernel = rng.uniform(0.5, 1.5, size=in_channels).astype("float32")
        kernel *= target / float(np.dot(kernel, input_values))
        weight[output_channel, :, 1, 1] = kernel
    graph.const("w", weight)
    graph.const("b", np.zeros((out_channels,), dtype="float32"))
    graph.node("Conv", "c", "images", "w", "b", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, out_channels, dim * dim))
    graph.output("output0")

    optimized_path = tmp_path / "optimized.aexrt"
    classes = out_channels - 4
    optimized_info = save_aexrt_engine(graph, optimized_path, classes=classes)
    assert optimized_info["kernel_plan"][0][2] == optimized_algorithm
    if optimized_algorithm == 46:
        assert optimized_info["packed_layout_counts"] == {5: 1}

    stable_conv_algorithm = engine_module._stable_conv_algorithm

    def reference_plan(params, *, silu):
        shape = (in_channels, dim, dim, out_channels, dim, dim)
        if silu and tuple(int(value) for value in params[1:7]) == shape:
            return reference_algorithm
        return stable_conv_algorithm(params, silu=silu)

    monkeypatch.setattr(engine_module, "_stable_conv_algorithm", reference_plan)
    if optimized_algorithm == 44:
        monkeypatch.setattr(
            engine_module, "_uses_native_fp16_pos2_40x40_64x64", lambda params: False
        )
    reference_path = tmp_path / "reference.aexrt"
    reference_info = save_aexrt_engine(graph, reference_path, classes=classes)
    assert reference_info["kernel_plan"][0][2] == reference_algorithm

    input_data = np.zeros((1, in_channels, dim, dim), dtype="float32")
    input_data[0, :, dim // 2, dim // 2] = input_values
    reference_model = NativeCppYoloModel(reference_path)
    optimized_model = NativeCppYoloModel(optimized_path)
    try:
        reference = reference_model.run(input_data, max_detections=4)
        actual = optimized_model.run(input_data, max_detections=4)
    finally:
        optimized_model.close()
        reference_model.close()

    assert len(reference) == len(actual) == 1
    assert reference[0].class_id == actual[0].class_id
    tolerance = 0.01 if optimized_algorithm == 44 else 0.001
    assert abs(reference[0].score - actual[0].score) < tolerance
    assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=tolerance)


@pytest.mark.parametrize(
    ("in_channels", "in_dim", "out_channels", "out_dim"),
    ((3, 320, 16, 160), (16, 160, 32, 80)),
)
def test_native_cpp_frontend_direct_pack4_matches_generic_when_available(
    tmp_path, monkeypatch, in_channels, in_dim, out_channels, out_dim
):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(20260714)
    graph = Graph(f"frontend_direct_pack4_{in_channels}_{out_channels}_numeric")
    graph.input("images", TensorSpec((1, in_channels, in_dim, in_dim), "float32"))
    weight = np.zeros((out_channels, in_channels, 3, 3), dtype="float32")
    input_values = rng.uniform(0.75, 1.25, size=in_channels).astype("float32")
    for output_channel, target in enumerate((10.0, 10.0, 4.0, 4.0, 1.5)):
        kernel = rng.uniform(0.5, 1.5, size=in_channels).astype("float32")
        kernel *= target / float(np.dot(kernel, input_values))
        weight[output_channel, :, 1, 1] = kernel
    graph.const("w", weight)
    graph.const("b", np.zeros((out_channels,), dtype="float32"))
    graph.node("Conv", "c", "images", "w", "b", strides=[2, 2], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, out_channels, out_dim * out_dim))
    graph.output("output0")

    classes = out_channels - 4
    optimized_path = tmp_path / "frontend_direct.aexrt"
    optimized_info = save_aexrt_engine(graph, optimized_path, classes=classes)
    assert optimized_info["kernel_plan"][0][2] == 24

    stable_conv_algorithm = engine_module._stable_conv_algorithm

    def generic_plan(params, *, silu):
        shape = (in_channels, in_dim, in_dim, out_channels, out_dim, out_dim, 3, 3, 2, 2)
        if silu and tuple(int(value) for value in params[1:11]) == shape:
            return 0
        return stable_conv_algorithm(params, silu=silu)

    monkeypatch.setattr(engine_module, "_stable_conv_algorithm", generic_plan)
    reference_path = tmp_path / "frontend_generic.aexrt"
    reference_info = save_aexrt_engine(graph, reference_path, classes=classes)
    assert reference_info["kernel_plan"][0][2] == 0

    input_data = np.zeros((1, in_channels, in_dim, in_dim), dtype="float32")
    input_data[0, :, in_dim // 2, in_dim // 2] = input_values
    reference_model = NativeCppYoloModel(reference_path)
    optimized_model = NativeCppYoloModel(optimized_path)
    try:
        reference = reference_model.run(input_data, max_detections=4)
        actual = optimized_model.run(input_data, max_detections=4)
    finally:
        optimized_model.close()
        reference_model.close()

    assert len(reference) == len(actual) == 1
    assert reference[0].class_id == actual[0].class_id
    assert abs(reference[0].score - actual[0].score) < 0.001
    assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=0.001)


def test_native_cpp_stride2_conv1x1_plan_matches_unfused_when_available(tmp_path, monkeypatch):
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(20260715)
    graph = Graph("stride2_conv1x1_fusion_numeric")
    graph.input("images", TensorSpec((1, 32, 20, 20), "float32"))
    weight0 = np.zeros((64, 32, 3, 3), dtype="float32")
    input_values = rng.uniform(0.75, 1.25, size=32).astype("float32")
    for output_channel, target in enumerate((10.0, 10.0, 4.0, 4.0, 1.5)):
        kernel = rng.uniform(0.5, 1.5, size=32).astype("float32")
        kernel *= target / float(np.dot(kernel, input_values))
        weight0[output_channel, :, 1, 1] = kernel
    weight1 = np.zeros((64, 64, 1, 1), dtype="float32")
    weight1[np.arange(64), np.arange(64), 0, 0] = 1.0
    graph.const("w0", weight0)
    graph.const("b0", np.zeros((64,), dtype="float32"))
    graph.const("w1", weight1)
    graph.const("b1", np.zeros((64,), dtype="float32"))
    graph.node("Conv", "c0", "images", "w0", "b0", strides=[2, 2], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    graph.sigmoid("s0", "c0")
    graph.mul("a0", "c0", "s0")
    graph.node("Conv", "c1", "a0", "w1", "b1", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    graph.sigmoid("s1", "c1")
    graph.mul("a1", "c1", "s1")
    graph.reshape("output0", "a1", (1, 64, 100))
    graph.output("output0")

    fused_path = tmp_path / "fused.aexrt"
    fused_info = save_aexrt_engine(graph, fused_path, classes=60, precision="fp32")
    assert any(group["kind"] == 6 for group in fused_info["fusion_plan"])

    build_fusion_plan = engine_module._build_fusion_plan

    def without_stride2_conv1x1(*args, **kwargs):
        return [group for group in build_fusion_plan(*args, **kwargs) if group["kind"] != 6]

    monkeypatch.setattr(engine_module, "_build_fusion_plan", without_stride2_conv1x1)
    reference_path = tmp_path / "unfused.aexrt"
    reference_info = save_aexrt_engine(graph, reference_path, classes=60, precision="fp32")
    assert not any(group["kind"] == 6 for group in reference_info["fusion_plan"])

    monkeypatch.setenv("AEXRT_NATIVE_D3D12_PROFILE_REPLAY", "1")
    input_data = np.zeros((1, 32, 20, 20), dtype="float32")
    input_data[0, :, 10, 10] = input_values
    reference_model = NativeCppYoloModel(reference_path)
    fused_model = NativeCppYoloModel(fused_path)
    try:
        reference = reference_model.run(input_data, max_detections=4)
        actual = fused_model.run(input_data, max_detections=4)
        assert not any("STRIDE2_CONV1X1_SUPERBLOCK" in label for label, _ in reference_model.profile_events())
        assert any("STRIDE2_CONV1X1_SUPERBLOCK" in label for label, _ in fused_model.profile_events())
    finally:
        fused_model.close()
        reference_model.close()

    assert len(reference) == len(actual) == 1
    assert reference[0].class_id == actual[0].class_id
    assert abs(reference[0].score - actual[0].score) < 0.001
    assert_close(np.asarray(reference[0].xyxy), np.asarray(actual[0].xyxy), tol=0.001)


def test_native_cpp_yolo_package_run_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolo_package.aexyolo.json"))
    package = build_yolo_output0_package((1, 6, 5), source_model="synthetic.onnx", classes=2, max_candidates=8, conf_threshold=0.25, iou_threshold=0.5)
    save_yolo_package(package, path)

    try:
        runtime = NativeCppYoloModel(path)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        out = np.zeros((1, 6, 5), dtype="float32")
        out[0, :4, 0] = [10, 10, 10, 10]
        out[0, :4, 1] = [11, 10, 10, 10]
        out[0, :4, 2] = [50, 50, 8, 8]
        out[0, :4, 3] = [80, 80, 8, 8]
        out[0, :4, 4] = [70, 70, 8, 8]
        out[0, 4:, 0] = [0.9, 0.1]
        out[0, 4:, 1] = [0.8, 0.1]
        out[0, 4:, 2] = [0.1, 0.7]
        out[0, 4:, 3] = [0.2, 0.1]
        out[0, 4:, 4] = [0.4, 0.1]

        dets = runtime.run(out, max_detections=4)
        assert runtime.input_element_count == 30
        assert runtime.class_count == 2
        assert runtime.anchor_count == 5
        assert [(d.class_id, round(d.score, 2)) for d in dets] == [(0, 0.9), (1, 0.7), (0, 0.4)]
    finally:
        runtime.close()

    path_v5 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolo_package_v5_layout.aexyolo.json"))
    package_v5 = build_yolo_output0_package(
        (1, 5, 7),
        source_model="synthetic_v5.onnx",
        classes=2,
        layout="channels_last",
        objectness=True,
        max_candidates=8,
        conf_threshold=0.25,
        iou_threshold=0.5,
    )
    save_yolo_package(package_v5, path_v5)
    runtime_v5 = NativeCppYoloModel(path_v5)
    try:
        out_v5 = np.zeros((1, 5, 7), dtype="float32")
        out_v5[0, 0, :4] = [10, 10, 10, 10]
        out_v5[0, 0, 4:] = [0.5, 0.9, 0.1]
        out_v5[0, 1, :4] = [50, 50, 8, 8]
        out_v5[0, 1, 4:] = [0.8, 0.1, 0.7]
        dets_v5 = runtime_v5.run(out_v5, max_detections=4)
        assert runtime_v5.input_element_count == 35
        assert runtime_v5.class_count == 2
        assert runtime_v5.channel_count == 7
        assert runtime_v5.anchor_count == 5
        assert runtime_v5.output_layout == 2
        assert runtime_v5.has_objectness is True
        assert [(d.class_id, round(d.score, 2)) for d in dets_v5] == [(1, 0.56), (0, 0.45)]
    finally:
        runtime_v5.close()


def test_native_cpp_loads_native_d3d12_graph_yolo_package_metadata_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    g = Graph("tiny_yolo_native_cpp_package")
    g.input("images", TensorSpec((1, 3, 4, 4), "float32"))
    g.const("w", np.ones((6, 3, 1, 1), dtype="float32") * 0.01)
    g.const("b", np.zeros((6,), dtype="float32"))
    g.conv2d("conv", "images", "w", "b")
    g.sigmoid("gate", "conv")
    g.mul("act", "conv", "gate")
    g.reshape("output0", "act", (1, 6, 16))
    g.output("output0")

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolo_native_graph.aexyolo.json"))
    save_yolo_package(build_yolo_native_d3d12_graph_package(g, source_model="tiny.onnx", classes=2), path)

    try:
        runtime = NativeCppYoloModel(path)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        assert runtime.package_mode == 2
        assert runtime.executable is True
        assert runtime.input_element_count == 48
        assert runtime.class_count == 2
        assert runtime.anchor_count == 16
        assert runtime.graph_node_count == 4
        assert runtime.graph_value_count >= 7
        assert runtime.constant_count == 2
        assert runtime.prepared_command_count >= 3
        assert runtime.supported_prepared_command_count >= 2
        assert runtime.unsupported_prepared_command_count == 0
        dets = runtime.run(np.zeros((1, 3, 4, 4), dtype="float32"), max_detections=8)
        assert dets == []
    finally:
        runtime.close()


def test_native_cpp_yolo_exposes_c2f_concat_residual_plan_stats_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(117)
    g = Graph("tiny_yolo_native_cpp_c2f_stats")
    g.input("images", TensorSpec((1, 4, 5, 5), "float32"))
    for name, value in {
        "w1": (rng.standard_normal((2, 2, 3, 3)) * 0.03).astype("float32"),
        "b1": (rng.standard_normal((2,)) * 0.01).astype("float32"),
        "w2": (rng.standard_normal((2, 2, 3, 3)) * 0.03).astype("float32"),
        "b2": (rng.standard_normal((2,)) * 0.01).astype("float32"),
        "w3": (rng.standard_normal((6, 6, 1, 1)) * 0.03).astype("float32"),
        "b3": np.zeros((6,), dtype="float32"),
    }.items():
        g.const(name, value)
    g.node("Split", ["left", "right"], "images", axis=1, split=[2, 2])
    g.node("Conv", "c1", "right", "w1", "b1", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("s1", "c1")
    g.mul("a1", "c1", "s1")
    g.node("Conv", "c2", "a1", "w2", "b2", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("s2", "c2")
    g.mul("a2", "c2", "s2")
    g.add("res", "right", "a2")
    g.node("Concat", "cat", "left", "right", "res", axis=1)
    g.node("Conv", "cv2", "cat", "w3", "b3", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.sigmoid("cv2s", "cv2")
    g.mul("act", "cv2", "cv2s")
    g.reshape("output0", "act", (1, 6, 25))
    g.output("output0")

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolo_native_graph_c2f_stats.aexyolo.json"))
    save_yolo_package(build_yolo_native_d3d12_graph_package(g, source_model="tiny.onnx", classes=2), path)

    try:
        runtime = NativeCppYoloModel(path)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        assert runtime.concat_residual_conv1x1_fusion_count == 0
        dets = runtime.run(np.zeros((1, 4, 5, 5), dtype="float32"), max_detections=4)
        assert dets == []
        assert runtime.concat_residual_conv1x1_fusion_count == 1
        assert runtime.c2f_tail_residual_fusion_count == 0
        assert runtime.prepared_skipped_command_count >= 1
        assert runtime.tileflow_3x3_spatial_count >= 2
    finally:
        runtime.close()

    old_direct = os.environ.get("AEXRT_NATIVE_D3D12_ENABLE_DIRECT_C2F_RESIDUAL_ADD")
    old_disable_direct = os.environ.pop("AEXRT_NATIVE_D3D12_DISABLE_DIRECT_C2F_RESIDUAL_ADD", None)
    os.environ["AEXRT_NATIVE_D3D12_ENABLE_DIRECT_C2F_RESIDUAL_ADD"] = "1"
    try:
        runtime_direct = NativeCppYoloModel(path)
        try:
            dets = runtime_direct.run(np.zeros((1, 4, 5, 5), dtype="float32"), max_detections=4)
            assert dets == []
            assert runtime_direct.concat_residual_conv1x1_fusion_count == 0
            assert runtime_direct.c2f_tail_residual_fusion_count == 1
        finally:
            runtime_direct.close()
    finally:
        if old_direct is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_ENABLE_DIRECT_C2F_RESIDUAL_ADD", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_ENABLE_DIRECT_C2F_RESIDUAL_ADD"] = old_direct
        if old_disable_direct is not None:
            os.environ["AEXRT_NATIVE_D3D12_DISABLE_DIRECT_C2F_RESIDUAL_ADD"] = old_disable_direct


def test_native_cpp_yolo_exposes_paired_conv3x3_plan_stats_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(213)
    g = Graph("tiny_yolo_native_cpp_paired_conv3x3_stats")
    g.input("images", TensorSpec((1, 4, 20, 20), "float32"))
    for name, value in {
        "w0": (rng.standard_normal((4, 4, 3, 3)) * 0.03).astype("float32"),
        "b0": np.zeros((4,), dtype="float32"),
        "w1": (rng.standard_normal((4, 4, 3, 3)) * 0.03).astype("float32"),
        "b1": np.zeros((4,), dtype="float32"),
    }.items():
        g.const(name, value)
    g.node("Conv", "c0", "images", "w0", "b0", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("s0", "c0")
    g.mul("a0", "c0", "s0")
    g.node("Conv", "c1", "images", "w1", "b1", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("s1", "c1")
    g.mul("a1", "c1", "s1")
    g.node("Concat", "cat", "a0", "a1", axis=1)
    g.reshape("output0", "cat", (1, 8, 400))
    g.output("output0")

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolo_native_graph_paired_conv3x3.aexyolo.json"))
    save_yolo_package(build_yolo_native_d3d12_graph_package(g, source_model="tiny_pair.onnx", classes=4), path)

    old_disable = os.environ.pop("AEXRT_NATIVE_D3D12_DISABLE_PAIRED_CONV3X3_SILU", None)
    try:
        runtime = NativeCppYoloModel(path)
    except RuntimeError as e:
        if old_disable is not None:
            os.environ["AEXRT_NATIVE_D3D12_DISABLE_PAIRED_CONV3X3_SILU"] = old_disable
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        dets = runtime.run(np.zeros((1, 4, 20, 20), dtype="float32"), max_detections=4)
        assert dets == []
        assert runtime.paired_conv3x3_fusion_count == 1
    finally:
        runtime.close()
        if old_disable is not None:
            os.environ["AEXRT_NATIVE_D3D12_DISABLE_PAIRED_CONV3X3_SILU"] = old_disable


def test_native_cpp_yolo_uses_exact_shape_implicit_gemm_plans_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(74)
    g40 = Graph("tiny_yolo_native_cpp_shape40_implicit_gemm")
    g40.input("images", TensorSpec((1, 64, 40, 40), "float32"))
    g40.const("w", (rng.standard_normal((64, 64, 3, 3)) * 0.005).astype("float32"))
    g40.const("b", np.zeros((64,), dtype="float32"))
    g40.node("Conv", "c", "images", "w", "b", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g40.sigmoid("s", "c")
    g40.mul("act", "c", "s")
    g40.reshape("output0", "act", (1, 64, 1600))
    g40.output("output0")

    path40 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolo_native_graph_shape40.aexyolo.json"))
    save_yolo_package(build_yolo_native_d3d12_graph_package(g40, source_model="tiny_shape40.onnx", classes=60), path40)

    old_enable_exact_40 = os.environ.get("AEXRT_NATIVE_D3D12_ENABLE_EXACT_40X40_64X64_IMPLICIT_GEMM_3X3")
    old_disable_40 = os.environ.pop("AEXRT_NATIVE_D3D12_DISABLE_SHAPE40_IMPLICIT_GEMM_3X3", None)
    old_disable_exact_40 = os.environ.pop("AEXRT_NATIVE_D3D12_DISABLE_EXACT_40X40_64X64_IMPLICIT_GEMM_3X3", None)
    os.environ["AEXRT_NATIVE_D3D12_ENABLE_EXACT_40X40_64X64_IMPLICIT_GEMM_3X3"] = "1"
    try:
        runtime40 = NativeCppYoloModel(path40)
    except RuntimeError as e:
        if old_enable_exact_40 is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_ENABLE_EXACT_40X40_64X64_IMPLICIT_GEMM_3X3", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_ENABLE_EXACT_40X40_64X64_IMPLICIT_GEMM_3X3"] = old_enable_exact_40
        if old_disable_40 is not None:
            os.environ["AEXRT_NATIVE_D3D12_DISABLE_SHAPE40_IMPLICIT_GEMM_3X3"] = old_disable_40
        if old_disable_exact_40 is not None:
            os.environ["AEXRT_NATIVE_D3D12_DISABLE_EXACT_40X40_64X64_IMPLICIT_GEMM_3X3"] = old_disable_exact_40
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        dets = runtime40.run(np.zeros((1, 64, 40, 40), dtype="float32"), max_detections=4)
        assert dets == []
        assert runtime40.tileflow_3x3_implicit_gemm_40x40_count == 1
        assert runtime40.tileflow_3x3_implicit_gemm_20x20_count == 0
        assert runtime40.tileflow_3x3_implicit_gemm_10x10_count == 0
        assert runtime40.tileflow_3x3_exact_40x40_64x64_count == 1
        assert runtime40.tileflow_3x3_exact_20x20_64x64_count == 0
        assert runtime40.tileflow_3x3_exact_10x10_128x128_count == 0
    finally:
        runtime40.close()
        if old_enable_exact_40 is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_ENABLE_EXACT_40X40_64X64_IMPLICIT_GEMM_3X3", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_ENABLE_EXACT_40X40_64X64_IMPLICIT_GEMM_3X3"] = old_enable_exact_40
        if old_disable_40 is not None:
            os.environ["AEXRT_NATIVE_D3D12_DISABLE_SHAPE40_IMPLICIT_GEMM_3X3"] = old_disable_40
        if old_disable_exact_40 is not None:
            os.environ["AEXRT_NATIVE_D3D12_DISABLE_EXACT_40X40_64X64_IMPLICIT_GEMM_3X3"] = old_disable_exact_40

    g = Graph("tiny_yolo_native_cpp_shape20_implicit_gemm")
    g.input("images", TensorSpec((1, 64, 20, 20), "float32"))
    g.const("w", (rng.standard_normal((64, 64, 3, 3)) * 0.01).astype("float32"))
    g.const("b", np.zeros((64,), dtype="float32"))
    g.node("Conv", "c", "images", "w", "b", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("s", "c")
    g.mul("act", "c", "s")
    g.reshape("output0", "act", (1, 64, 400))
    g.output("output0")

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolo_native_graph_shape20.aexyolo.json"))
    save_yolo_package(build_yolo_native_d3d12_graph_package(g, source_model="tiny_shape20.onnx", classes=60), path)

    old_disable = os.environ.pop("AEXRT_NATIVE_D3D12_DISABLE_SHAPE20_IMPLICIT_GEMM_3X3", None)
    try:
        runtime = NativeCppYoloModel(path)
    except RuntimeError as e:
        if old_disable is not None:
            os.environ["AEXRT_NATIVE_D3D12_DISABLE_SHAPE20_IMPLICIT_GEMM_3X3"] = old_disable
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        dets = runtime.run(np.zeros((1, 64, 20, 20), dtype="float32"), max_detections=4)
        assert dets == []
        assert runtime.tileflow_3x3_implicit_gemm_20x20_count == 1
        assert runtime.tileflow_3x3_implicit_gemm_10x10_count == 0
        assert runtime.tileflow_3x3_exact_20x20_64x64_count == 1
        assert runtime.tileflow_3x3_exact_10x10_128x128_count == 0
    finally:
        runtime.close()
        if old_disable is not None:
            os.environ["AEXRT_NATIVE_D3D12_DISABLE_SHAPE20_IMPLICIT_GEMM_3X3"] = old_disable

    g10 = Graph("tiny_yolo_native_cpp_shape10_implicit_gemm")
    g10.input("images", TensorSpec((1, 128, 10, 10), "float32"))
    g10.const("w", (rng.standard_normal((128, 128, 3, 3)) * 0.005).astype("float32"))
    g10.const("b", np.zeros((128,), dtype="float32"))
    g10.node("Conv", "c", "images", "w", "b", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g10.sigmoid("s", "c")
    g10.mul("act", "c", "s")
    g10.reshape("output0", "act", (1, 128, 100))
    g10.output("output0")

    path10 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolo_native_graph_shape10.aexyolo.json"))
    save_yolo_package(build_yolo_native_d3d12_graph_package(g10, source_model="tiny_shape10.onnx", classes=124), path10)

    old_disable_10 = os.environ.pop("AEXRT_NATIVE_D3D12_DISABLE_SHAPE10_IMPLICIT_GEMM_3X3", None)
    try:
        runtime10 = NativeCppYoloModel(path10)
    except RuntimeError as e:
        if old_disable_10 is not None:
            os.environ["AEXRT_NATIVE_D3D12_DISABLE_SHAPE10_IMPLICIT_GEMM_3X3"] = old_disable_10
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        dets = runtime10.run(np.zeros((1, 128, 10, 10), dtype="float32"), max_detections=4)
        assert dets == []
        assert runtime10.tileflow_3x3_implicit_gemm_20x20_count == 0
        assert runtime10.tileflow_3x3_implicit_gemm_10x10_count == 1
        assert runtime10.tileflow_3x3_exact_20x20_64x64_count == 0
        assert runtime10.tileflow_3x3_exact_10x10_128x128_count == 1
    finally:
        runtime10.close()
        if old_disable_10 is not None:
            os.environ["AEXRT_NATIVE_D3D12_DISABLE_SHAPE10_IMPLICIT_GEMM_3X3"] = old_disable_10


def test_native_cpp_yolo_experimental_concat_conv1x1_gemm_compiles_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    rng = np.random.default_rng(83)
    g = Graph("tiny_yolo_native_cpp_concat_conv1x1_gemm")
    g.input("images", TensorSpec((1, 64, 10, 10), "float32"))
    g.const("w", (rng.standard_normal((64, 64, 1, 1)) * 0.01).astype("float32"))
    g.const("b", np.zeros((64,), dtype="float32"))
    g.node("Split", ["left", "right"], "images", axis=1, split=[32, 32])
    g.node("Concat", "cat", "left", "right", axis=1)
    g.node("Conv", "c", "cat", "w", "b", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.sigmoid("s", "c")
    g.mul("act", "c", "s")
    g.reshape("output0", "act", (1, 64, 100))
    g.output("output0")

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolo_native_graph_concat_gemm.aexyolo.json"))
    package = build_yolo_native_d3d12_graph_package(g, source_model="tiny_concat_gemm.onnx", classes=60)
    assert package["compiler"]["cxx_kernel_coverage"]["required_kernel_counts"]["concat_conv1x1"] == 1
    save_yolo_package(package, path)

    old = os.environ.get("AEXRT_NATIVE_D3D12_ENABLE_EXPERIMENTAL_CONCAT_CONV1X1_GEMM")
    os.environ["AEXRT_NATIVE_D3D12_ENABLE_EXPERIMENTAL_CONCAT_CONV1X1_GEMM"] = "1"
    try:
        runtime = NativeCppYoloModel(path)
        try:
            assert runtime.run(np.zeros((1, 64, 10, 10), dtype="float32"), max_detections=4) == []
        finally:
            runtime.close()
    finally:
        if old is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_ENABLE_EXPERIMENTAL_CONCAT_CONV1X1_GEMM", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_ENABLE_EXPERIMENTAL_CONCAT_CONV1X1_GEMM"] = old


def test_native_cpp_native_graph_yolo_gpu_postprocess_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    g = Graph("tiny_yolo_native_cpp_gpu_postprocess")
    g.input("images", TensorSpec((1, 6, 5), "float32"))
    g.reshape("output0", "images", (1, 6, 5))
    g.output("output0")

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolo_native_graph_gpu_post.aexyolo.json"))
    save_yolo_package(
        build_yolo_native_d3d12_graph_package(
            g,
            source_model="synthetic.onnx",
            classes=2,
            max_candidates=8,
            max_detections=4,
            conf_threshold=0.25,
            iou_threshold=0.5,
        ),
        path,
    )

    try:
        runtime = NativeCppYoloModel(path)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        out = np.zeros((1, 6, 5), dtype="float32")
        out[0, :4, 0] = [10, 10, 10, 10]
        out[0, :4, 1] = [11, 10, 10, 10]
        out[0, :4, 2] = [50, 50, 8, 8]
        out[0, :4, 3] = [80, 80, 8, 8]
        out[0, :4, 4] = [70, 70, 8, 8]
        out[0, 4:, 0] = [0.9, 0.1]
        out[0, 4:, 1] = [0.8, 0.1]
        out[0, 4:, 2] = [0.1, 0.7]
        out[0, 4:, 3] = [0.2, 0.1]
        out[0, 4:, 4] = [0.4, 0.1]

        dets = runtime.run(out, max_detections=4)
        assert [(d.class_id, round(d.score, 2), tuple(round(x, 1) for x in d.xyxy)) for d in dets] == [
            (0, 0.9, (5.0, 5.0, 15.0, 15.0)),
            (1, 0.7, (46.0, 46.0, 54.0, 54.0)),
            (0, 0.4, (66.0, 66.0, 74.0, 74.0)),
        ]
    finally:
        runtime.close()


def test_native_cpp_native_graph_multi_axis_slice_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    g = Graph("tiny_yolo_native_cpp_focus_slice")
    g.input("images", TensorSpec((1, 2, 4, 4), "float32"))
    g.node("Slice", "focus", "images", starts=[0, 0], ends=[4, 4], axes=[2, 3], steps=[2, 2])
    g.reshape("output0", "focus", (1, 8, 1))
    g.output("output0")

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolo_native_graph_focus_slice.aexyolo.json"))
    save_yolo_package(
        build_yolo_native_d3d12_graph_package(
            g,
            source_model="synthetic_focus.onnx",
            classes=4,
            max_candidates=4,
            max_detections=2,
        ),
        path,
    )

    try:
        runtime = NativeCppYoloModel(path)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise

    try:
        assert runtime.executable is True
        assert runtime.prepared_command_count >= 2
        _ = runtime.run(np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4), max_detections=1)
    finally:
        runtime.close()


def test_native_cpp_compile_onnx_constant_slice_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except Exception:
        return

    def const_node(name: str, values: list[int]):
        tensor = numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=f"{name}_value")
        return helper.make_node("Constant", [], [name], value=tensor)

    nodes = [
        const_node("starts", [0, 0]),
        const_node("ends", [4, 4]),
        const_node("axes", [2, 3]),
        const_node("steps", [2, 2]),
        helper.make_node("Slice", ["images", "starts", "ends", "axes", "steps"], ["focus"]),
        const_node("shape", [1, 8, 1]),
        helper.make_node("Reshape", ["focus", "shape"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "constant_slice_yolo_like",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 2, 4, 4])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 8, 1])],
        value_info=[helper.make_tensor_value_info("focus", TensorProto.FLOAT, [1, 2, 2, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_constant_slice_yolo_like.onnx"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(model, path)

    try:
        runtime = NativeCppYoloModel(path)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise
    try:
        assert runtime.executable is True
        assert runtime.class_count == 4
        assert runtime.channel_count == 8
        assert runtime.anchor_count == 1
        assert runtime.output_layout == 1
        _ = runtime.run(np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4), max_detections=1)
    finally:
        runtime.close()


def test_native_cpp_compile_yolov5_style_onnx_without_value_info_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except Exception:
        return

    initializers = [
        numpy_helper.from_array(np.zeros((24, 3, 1, 1), dtype=np.float32), name="weight"),
        numpy_helper.from_array(np.zeros((24,), dtype=np.float32), name="bias"),
        numpy_helper.from_array(np.asarray([1, 3, 8, 8, 8], dtype=np.int64), name="detect_shape"),
        numpy_helper.from_array(np.asarray([2.0], dtype=np.float32), name="exponent"),
        numpy_helper.from_array(np.asarray([1, -1, 8], dtype=np.int64), name="output_shape"),
    ]
    nodes = [
        helper.make_node("Conv", ["images", "weight", "bias"], ["conv"]),
        helper.make_node("Reshape", ["conv", "detect_shape"], ["detect5d"]),
        helper.make_node("Transpose", ["detect5d"], ["detect5d_t"], perm=[0, 1, 3, 4, 2]),
        helper.make_node("Sigmoid", ["detect5d_t"], ["scores"]),
        helper.make_node("Split", ["scores"], ["xy", "wh", "obj_cls"], axis=4, split=[2, 2, 4]),
        helper.make_node("Pow", ["wh", "exponent"], ["wh2"]),
        helper.make_node("Concat", ["xy", "wh2", "obj_cls"], ["detect"], axis=4),
        helper.make_node("Reshape", ["detect", "output_shape"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "yolov5_no_value_info",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 192, 8])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    helper.set_model_props(model, {"names": "{0: 'a', 1: 'b', 2: 'c'}"})
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "native", "test_yolov5_no_value_info.onnx"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(model, path)

    try:
        runtime = NativeCppYoloModel(path)
    except RuntimeError as e:
        if "aexrt_native_cpp.dll not found" in str(e):
            return
        raise
    try:
        assert runtime.executable is True
        assert runtime.prepared_command_count == runtime.supported_prepared_command_count
        assert runtime.output_layout == 2
        assert runtime.has_objectness is True
        assert runtime.class_count == 3
        assert runtime.channel_count == 8
        assert runtime.anchor_count == 192
        _ = runtime.run(np.zeros((1, 3, 8, 8), dtype=np.float32), max_detections=1)
    finally:
        runtime.close()


def test_unsupported_op_rejected_by_backend():
    g = Graph("bad_op")
    g.input("x", TensorSpec((1,), "float32"))
    g.node("MadeUpAcceleratorOp", "y", "x")
    g.output("y")

    try:
        InferenceSession(g, backend="numpy", optimize=False)
    except NotImplementedError as e:
        assert "MadeUpAcceleratorOp" in str(e)
    else:
        raise AssertionError("unsupported op should be rejected during session creation")


def test_auto_backend_creates_execution_plan():
    g = Graph("auto")
    g.input("x", TensorSpec((1, 2), "float32"))
    g.relu("y", "x")
    g.output("y")
    session = InferenceSession(g, backend="auto", device="auto", optimize=True)
    assert session.execution_plan() is not None
    assert session.info().name in {"torch", "directml", "numpy"}


def test_host_device_upload_download_roundtrip():
    device = HostDevice()
    x = np.arange(6, dtype="float32").reshape(2, 3)
    buffer = device.upload(x, label="x")
    y = device.download(buffer)
    assert buffer.nbytes == x.nbytes
    assert buffer.dtype == "float32"
    assert buffer.shape == (2, 3)
    assert_close(x, y)


def test_native_d3d12_probe_is_not_directml_bridge():
    info = NativeD3D12Device.probe()
    assert info.api == "aexrt_native_d3d12"
    assert info.device_type == "d3d12"
    assert info.available in {True, False}


def test_native_d3d12_upload_download_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    device = NativeD3D12Device()
    x = np.arange(16, dtype="float32").reshape(4, 4)
    buffer = device.upload(x, label="native_roundtrip")
    y = device.download(buffer)
    assert buffer.nbytes == x.nbytes
    assert buffer.dtype == "float32"
    assert buffer.shape == (4, 4)
    assert device.info().api == "aexrt_native_d3d12"
    assert_close(x, y)


def test_native_d3d12_relu_dispatch_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    device = NativeD3D12Device()
    x = np.array([-2.0, -0.5, 0.0, 3.0, 9.0], dtype="float32")
    y_buffer = device.dispatch_relu_float32(device.upload(x, label="relu_x"), x.size)
    y = device.download(y_buffer)
    assert_close(y, np.maximum(x, 0))


def test_native_d3d12_relu_dispatch_into_reuses_output_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    device = NativeD3D12Device()
    x1 = np.array([-2.0, 1.0, 3.0, -4.0], dtype="float32")
    x2 = np.array([5.0, -6.0, -7.0, 8.0], dtype="float32")
    output = device.allocate_uav(x1.nbytes, dtype="float32", shape=x1.shape, label="relu_out")
    device.dispatch_relu_float32_into(device.upload(x1, label="relu_x1"), output, x1.size)
    y1 = device.download(output)
    device.dispatch_relu_float32_into(device.upload(x2, label="relu_x2"), output, x2.size)
    y2 = device.download(output)
    assert_close(y1, np.maximum(x1, 0))
    assert_close(y2, np.maximum(x2, 0))


def test_native_d3d12_prepared_relu_dispatch_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    device = NativeD3D12Device()
    x = np.array([-3.0, 1.0, -2.0, 4.0], dtype="float32")
    x_buffer = device.upload(x, label="prepared_relu_x")
    y_buffer = device.allocate_uav(x.nbytes, dtype="float32", shape=x.shape, label="prepared_relu_y")
    dispatch = device.prepare_relu_float32_dispatch(x_buffer, y_buffer, x.size)
    device.execute_relu_float32_dispatch(dispatch)
    device.execute_relu_float32_dispatch(dispatch)
    y = device.download(y_buffer)
    assert dispatch.op == "Relu"
    assert_close(y, np.maximum(x, 0))


def test_native_d3d12_conv2d_silu_dispatch_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(77)
    device = NativeD3D12Device()
    x = (rng.standard_normal((1, 2, 5, 5)) * 0.25).astype("float32")
    w = (rng.standard_normal((3, 2, 3, 3)) * 0.1).astype("float32")
    b = (rng.standard_normal((3,)) * 0.05).astype("float32")
    expected = conv2d_silu_ref(x, w, b, strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1), group=1)
    xb = device.upload(x, label="conv_silu_x")
    wb = device.upload(w, label="conv_silu_w")
    bb = device.upload(b, label="conv_silu_b")
    yb = device.allocate_uav(expected.nbytes, dtype="float32", shape=expected.shape, label="conv_silu_y")
    desc = {
        "batch": 1,
        "in_channels": 2,
        "in_h": 5,
        "in_w": 5,
        "out_channels": 3,
        "out_h": 5,
        "out_w": 5,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_top": 1,
        "pad_left": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "groups": 1,
    }
    dispatch = device.prepare_conv2d_silu_float32_dispatch(xb, wb, bb, yb, desc)
    device.execute_conv2d_silu_float32_dispatch(dispatch)
    y = device.download(yb)
    assert dispatch.op == "Conv2D+SiLU"
    assert_close(y, expected, tol=3e-5)


def test_native_d3d12_conv2d_silu_upload_ring_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(79)
    device = NativeD3D12Device()
    x = (rng.standard_normal((1, 2, 5, 5)) * 0.25).astype("float32")
    w = (rng.standard_normal((3, 2, 3, 3)) * 0.1).astype("float32")
    b = (rng.standard_normal((3,)) * 0.05).astype("float32")
    expected = conv2d_silu_ref(x, w, b, strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1), group=1)
    wb = device.upload(w, label="conv_silu_ring_w")
    bb = device.upload(b, label="conv_silu_ring_b")
    yb = device.allocate_uav(expected.nbytes, dtype="float32", shape=expected.shape, label="conv_silu_ring_y")
    desc = {
        "batch": 1,
        "in_channels": 2,
        "in_h": 5,
        "in_w": 5,
        "out_channels": 3,
        "out_h": 5,
        "out_w": 5,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_top": 1,
        "pad_left": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "groups": 1,
    }
    dispatch = device.prepare_conv2d_silu_upload_float32_dispatch(wb, bb, yb, desc, ring_size=2)
    device.execute_conv2d_silu_upload_float32_dispatch(dispatch, x)
    y = device.download(yb)
    assert dispatch.op == "Conv2D+SiLU+UploadRing"
    assert_close(y, expected, tol=3e-5)


def test_native_d3d12_yolo_decode_filter_stays_on_gpu_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    device = NativeD3D12Device()
    out = np.zeros((6, 4), dtype="float32")
    out[:4, 0] = [10, 10, 10, 10]
    out[:4, 1] = [11, 10, 10, 10]
    out[:4, 2] = [50, 50, 8, 8]
    out[:4, 3] = [80, 80, 8, 8]
    out[4:, 0] = [0.9, 0.1]
    out[4:, 1] = [0.8, 0.1]
    out[4:, 2] = [0.1, 0.7]
    out[4:, 3] = [0.2, 0.1]
    yolo_buffer = device.upload(out, label="yolo_head")
    detections, counter = device.allocate_yolo_detection_buffers(8, label="yolo_gpu")
    device.dispatch_yolo_decode_filter_float32(
        yolo_buffer,
        detections,
        counter,
        anchors=4,
        channels=6,
        classes=2,
        max_detections=8,
        conf_threshold=0.25,
    )
    dets = device.download_yolo_topk(detections, counter)
    count = dets.shape[0]
    assert count == 3
    scores = sorted([round(float(x), 2) for x in dets[:, 4]], reverse=True)
    assert scores == [0.9, 0.8, 0.7]


def test_native_d3d12_yolo_decode_nms_topk_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    device = NativeD3D12Device()
    out = np.zeros((6, 5), dtype="float32")
    out[:4, 0] = [10, 10, 10, 10]
    out[:4, 1] = [11, 10, 10, 10]
    out[:4, 2] = [50, 50, 8, 8]
    out[:4, 3] = [80, 80, 8, 8]
    out[:4, 4] = [70, 70, 8, 8]
    out[4:, 0] = [0.9, 0.1]
    out[4:, 1] = [0.8, 0.1]
    out[4:, 2] = [0.1, 0.7]
    out[4:, 3] = [0.2, 0.1]
    out[4:, 4] = [0.4, 0.2]
    yolo_buffer = device.upload(out, label="yolo_head_nms")
    candidates, candidate_counter, keep, detections, counter = device.allocate_yolo_nms_buffers(8, 4, label="yolo_gpu_nms")
    device.dispatch_yolo_decode_nms_float32(
        yolo_buffer,
        candidates,
        candidate_counter,
        keep,
        detections,
        counter,
        anchors=5,
        channels=6,
        classes=2,
        max_candidates=8,
        max_detections=4,
        conf_threshold=0.25,
        iou_threshold=0.5,
    )
    dets = device.download_yolo_topk(detections, counter)
    count = dets.shape[0]
    assert count == 3
    assert [round(float(x), 2) for x in dets[:, 4]] == [0.9, 0.7, 0.4]
    assert [int(x) for x in dets[:, 5]] == [0, 1, 0]


def test_native_d3d12_layout_primitives_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    device = NativeD3D12Device()

    x = np.arange(1 * 6 * 5, dtype="float32").reshape(1, 6, 5)
    xb = device.upload(x, label="rank3_layout_x")
    sliced = device.allocate_uav(1 * 2 * 5 * 4, dtype="float32", shape=(1, 2, 5), label="rank3_slice")
    device.dispatch_slice_float32_into(xb, sliced, axis=1, start=2)
    assert_close(device.download(sliced), x[:, 2:4, :])

    left = device.upload(x[:, :2, :], label="rank3_concat_left")
    right = device.upload(x[:, 2:, :], label="rank3_concat_right")
    cat_axis1 = device.allocate_uav(x.nbytes, dtype="float32", shape=x.shape, label="rank3_concat_axis1")
    device.dispatch_concat_float32_into([left, right], cat_axis1, axis=1)
    assert_close(device.download(cat_axis1), x)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype="float32")
    b = np.array([10, 20, 30], dtype="float32")
    yb = device.allocate_uav(a.nbytes, dtype="float32", shape=a.shape, label="broadcast_add")
    device.dispatch_binary_broadcast_float32_into("Add", device.upload(a), device.upload(b), yb)
    assert_close(device.download(yb), a + b)

    ab = device.upload(a, label="batched_add_a")
    bb = device.upload(b, label="batched_add_b")
    batched = device.allocate_uav(a.nbytes, dtype="float32", shape=a.shape, label="batched_broadcast_add")
    device.begin_batch()
    device.dispatch_binary_broadcast_float32_into("Add", ab, bb, batched)
    device.end_batch()
    assert_close(device.download(batched), a + b)

    replay_input = device.upload(a, label="prepared_graph_input")
    replay_bias = device.upload(b, label="prepared_graph_bias")
    replay_out = device.allocate_uav(a.nbytes, dtype="float32", shape=a.shape, label="prepared_graph_out")
    device.begin_prepared_graph()
    device.dispatch_binary_broadcast_float32_into("Add", replay_input, replay_bias, replay_out)
    prepared = device.end_prepared_graph()
    device.execute_prepared_graph(prepared)
    assert_close(device.download(replay_out), a + b)
    a2 = a * np.float32(2.0)
    device.upload_into(replay_input, a2)
    device.execute_prepared_graph(prepared)
    assert_close(device.download(replay_out), a2 + b)

    logits = np.zeros((1, 16, 4, 5), dtype="float32")
    for bin_idx in range(16):
        logits[:, bin_idx, :, :] = np.float32(bin_idx) / np.float32(16.0)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    expected_dfl = (probs * np.arange(16, dtype="float32").reshape(1, 16, 1, 1)).sum(axis=1, keepdims=True)
    dfl_out = device.allocate_uav(expected_dfl.nbytes, dtype="float32", shape=expected_dfl.shape, label="dfl_project")
    device.dispatch_dfl_project_float32_into(device.upload(probs, label="dfl_probs"), dfl_out)
    assert_close(device.download(dfl_out), expected_dfl, tol=2e-5)


def test_native_d3d12_channel_views_feed_concat_and_conv_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(81)
    device = NativeD3D12Device()
    x = np.arange(1 * 4 * 3 * 3, dtype="float32").reshape(1, 4, 3, 3)
    xb = device.upload(x, label="channel_view_x")
    left = device.create_buffer_view(xb, element_offset=0, element_count=2 * 3 * 3, dtype="float32", shape=(1, 2, 3, 3), label="left_view")
    right = device.create_buffer_view(xb, element_offset=2 * 3 * 3, element_count=2 * 3 * 3, dtype="float32", shape=(1, 2, 3, 3), label="right_view")

    assert device.buffer_info(right)["element_offset"] == 18
    assert_close(device.download(right), x[:, 2:, :, :])

    cat = device.allocate_uav(x.nbytes, dtype="float32", shape=x.shape, label="channel_view_concat")
    device.dispatch_concat_float32_into([left, right], cat, axis=1)
    assert_close(device.download(cat), x)

    w = (rng.standard_normal((3, 2, 1, 1)) * 0.1).astype("float32")
    b = (rng.standard_normal((3,)) * 0.05).astype("float32")
    expected = np.empty((1, 3, 3, 3), dtype="float32")
    for oc in range(3):
        expected[:, oc] = b[oc]
        for ic in range(2):
            expected[:, oc] += x[:, 2 + ic] * w[oc, ic, 0, 0]
    yb = device.allocate_uav(expected.nbytes, dtype="float32", shape=expected.shape, label="channel_view_conv_y")
    desc = {
        "batch": 1,
        "in_channels": 2,
        "in_h": 3,
        "in_w": 3,
        "out_channels": 3,
        "out_h": 3,
        "out_w": 3,
        "kernel_h": 1,
        "kernel_w": 1,
        "stride_h": 1,
        "stride_w": 1,
        "pad_top": 0,
        "pad_left": 0,
        "dilation_h": 1,
        "dilation_w": 1,
        "groups": 1,
    }
    device.dispatch_conv2d_float32_into(right, device.upload(w), device.upload(b), yb, desc)
    assert_close(device.download(yb), expected, tol=2e-5)


def test_native_d3d12_fuses_concat_1x1_conv_silu_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(83)
    x = (rng.standard_normal((1, 4, 4, 4)) * 0.2).astype("float32")
    w = (rng.standard_normal((5, 4, 1, 1)) * 0.1).astype("float32")
    b = (rng.standard_normal((5,)) * 0.03).astype("float32")
    expected = conv2d_silu_ref(np.concatenate([x[:, :2], x[:, 2:]], axis=1), w, b)

    g = Graph("native_concat_conv1x1_silu")
    g.input("x", TensorSpec(x.shape, "float32"))
    g.const("w", w)
    g.const("b", b)
    g.node("Split", ["left", "right"], "x", axis=1, split=[2, 2])
    g.node("Concat", "cat", "left", "right", axis=1)
    g.node("Conv", "c", "cat", "w", "b", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.sigmoid("s", "c")
    g.mul("y", "c", "s")
    g.output("y")

    session = InferenceSession(g, backend="native_d3d12", device="d3d12", output_numpy=True)
    y = session.run({"x": x})["y"]
    assert "fused_channel_concat_conv1x1" in session.backend.info().capabilities["features"]
    assert session.backend._fused_concat_conv1x1_count == 1
    assert_close(y, expected, tol=3e-5)


def test_native_d3d12_fuses_concat_1x1_conv_silu_fp16_superblock_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(90)
    x = (rng.standard_normal((1, 4, 4, 4)) * 0.2).astype("float32")
    w = (rng.standard_normal((5, 4, 1, 1)) * 0.1).astype("float32")
    b = (rng.standard_normal((5,)) * 0.03).astype("float32")
    expected = conv2d_silu_ref(np.concatenate([x[:, :2], x[:, 2:]], axis=1), w, b)

    g = Graph("native_concat_conv1x1_silu_fp16_superblock")
    g.input("x", TensorSpec(x.shape, "float32"))
    g.const("w", w)
    g.const("b", b)
    g.node("Split", ["left", "right"], "x", axis=1, split=[2, 2])
    g.node("Concat", "cat", "left", "right", axis=1)
    g.node("Conv", "c", "cat", "w", "b", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.sigmoid("s", "c")
    g.mul("act", "c", "s")
    g.identity("y", "act")
    g.output("y")

    old = os.environ.get("AEXRT_NATIVE_D3D12_FP16_SUPERBLOCK")
    os.environ["AEXRT_NATIVE_D3D12_FP16_SUPERBLOCK"] = "1"
    try:
        session = InferenceSession(g, backend="native_d3d12", device="d3d12", output_numpy=True)
        y = session.run({"x": x})["y"]
        assert len(session.backend._fp16_superblock_constants) == 1
        assert session.backend._fused_concat_conv1x1_count == 1
        assert_close(y, expected, tol=4e-4)
    finally:
        if old is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_FP16_SUPERBLOCK", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_FP16_SUPERBLOCK"] = old


def test_native_d3d12_fuses_concat_1x1_conv_silu_int8_superblock_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(91)
    x = (rng.standard_normal((1, 4, 4, 4)) * 0.05).astype("float32")
    w = (rng.standard_normal((5, 4, 1, 1)) * 0.04).astype("float32")
    b = (rng.standard_normal((5,)) * 0.01).astype("float32")
    expected = conv2d_silu_ref(np.concatenate([x[:, :2], x[:, 2:]], axis=1), w, b)

    g = Graph("native_concat_conv1x1_silu_int8_superblock")
    g.input("x", TensorSpec(x.shape, "float32"))
    g.const("w", w)
    g.const("b", b)
    g.node("Split", ["left", "right"], "x", axis=1, split=[2, 2])
    g.node("Concat", "cat", "left", "right", axis=1)
    g.node("Conv", "c", "cat", "w", "b", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.sigmoid("s", "c")
    g.mul("act", "c", "s")
    g.identity("y", "act")
    g.output("y")

    old = os.environ.get("AEXRT_NATIVE_D3D12_INT8_SUPERBLOCK")
    old_scale = os.environ.get("AEXRT_NATIVE_D3D12_INT8_ACT_SCALE")
    os.environ["AEXRT_NATIVE_D3D12_INT8_SUPERBLOCK"] = "1"
    os.environ["AEXRT_NATIVE_D3D12_INT8_ACT_SCALE"] = "0.001"
    try:
        session = InferenceSession(g, backend="native_d3d12", device="d3d12", output_numpy=True)
        y = session.run({"x": x})["y"]
        assert len(session.backend._int8_superblock_constants) == 1
        assert session.backend._fused_concat_conv1x1_count == 1
        assert_close(y, expected, tol=8e-4)
    finally:
        if old is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_INT8_SUPERBLOCK", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_INT8_SUPERBLOCK"] = old
        if old_scale is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_INT8_ACT_SCALE", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_INT8_ACT_SCALE"] = old_scale


def test_native_d3d12_fuses_sppf_tail_superblock_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(92)
    x = (rng.standard_normal((1, 3, 5, 5)) * 0.2).astype("float32")
    p1 = maxpool2d_ref(x)
    p2 = maxpool2d_ref(p1)
    p3 = maxpool2d_ref(p2)
    w = (rng.standard_normal((4, 12, 1, 1)) * 0.05).astype("float32")
    b = (rng.standard_normal((4,)) * 0.02).astype("float32")
    expected = conv2d_silu_ref(np.concatenate([x, p1, p2, p3], axis=1), w, b)

    g = Graph("native_sppf_tail_superblock")
    g.input("x", TensorSpec(x.shape, "float32"))
    g.const("w", w)
    g.const("b", b)
    g.node("MaxPool", "p1", "x", kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2], dilations=[1, 1], ceil_mode=0)
    g.node("MaxPool", "p2", "p1", kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2], dilations=[1, 1], ceil_mode=0)
    g.node("MaxPool", "p3", "p2", kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2], dilations=[1, 1], ceil_mode=0)
    g.node("Concat", "cat", "x", "p1", "p2", "p3", axis=1)
    g.node("Conv", "c", "cat", "w", "b", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.sigmoid("s", "c")
    g.mul("y", "c", "s")
    g.output("y")

    old = os.environ.get("AEXRT_NATIVE_D3D12_SPPF_SUPERBLOCK")
    os.environ["AEXRT_NATIVE_D3D12_SPPF_SUPERBLOCK"] = "1"
    try:
        session = InferenceSession(g, backend="native_d3d12", device="d3d12", output_numpy=True)
        y = session.run({"x": x})["y"]
        assert session.backend._fused_sppf_tail_count == 1
        assert_close(y, expected, tol=3e-5)
    finally:
        if old is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_SPPF_SUPERBLOCK", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_SPPF_SUPERBLOCK"] = old


def test_native_d3d12_fuses_c2f_residual_tail_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(84)
    x = (rng.standard_normal((1, 4, 4, 4)) * 0.2).astype("float32")
    w = (rng.standard_normal((5, 6, 1, 1)) * 0.1).astype("float32")
    b = (rng.standard_normal((5,)) * 0.03).astype("float32")
    left = x[:, :2]
    right = x[:, 2:]
    branch = right * np.float32(0.25)
    residual = right + branch
    expected = conv2d_silu_ref(np.concatenate([left, right, residual], axis=1), w, b)

    g = Graph("native_c2f_residual_tail")
    g.input("x", TensorSpec(x.shape, "float32"))
    g.const("scale", np.full((1, 2, 1, 1), 0.25, dtype="float32"))
    g.const("w", w)
    g.const("b", b)
    g.node("Split", ["left", "right"], "x", axis=1, split=[2, 2])
    g.mul("branch", "right", "scale")
    g.add("residual", "right", "branch")
    g.node("Concat", "cat", "left", "right", "residual", axis=1)
    g.node("Conv", "c", "cat", "w", "b", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.sigmoid("s", "c")
    g.mul("y", "c", "s")
    g.output("y")

    old = os.environ.get("AEXRT_NATIVE_D3D12_C2F_SUPERBLOCK")
    old_tiled = os.environ.get("AEXRT_NATIVE_D3D12_C2F_SUPERBLOCK_TILED")
    os.environ["AEXRT_NATIVE_D3D12_C2F_SUPERBLOCK"] = "1"
    os.environ["AEXRT_NATIVE_D3D12_C2F_SUPERBLOCK_TILED"] = "1"
    try:
        session = InferenceSession(g, backend="native_d3d12", device="d3d12", output_numpy=True)
        y = session.run({"x": x})["y"]
        assert session.backend._fused_concat_conv1x1_count == 1
        assert session.backend._fused_c2f_residual_tail_count == 1
        assert_close(y, expected, tol=3e-5)
    finally:
        if old is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_C2F_SUPERBLOCK", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_C2F_SUPERBLOCK"] = old
        if old_tiled is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_C2F_SUPERBLOCK_TILED", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_C2F_SUPERBLOCK_TILED"] = old_tiled


def test_native_d3d12_compiles_full_c2f_superblock_by_default_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(94)
    x = (rng.standard_normal((1, 4, 5, 5)) * 0.2).astype("float32")
    w_cv1 = (rng.standard_normal((4, 4, 1, 1)) * 0.08).astype("float32")
    b_cv1 = (rng.standard_normal((4,)) * 0.02).astype("float32")
    w1 = (rng.standard_normal((2, 2, 3, 3)) * 0.08).astype("float32")
    b1 = (rng.standard_normal((2,)) * 0.02).astype("float32")
    w2 = (rng.standard_normal((2, 2, 3, 3)) * 0.08).astype("float32")
    b2 = (rng.standard_normal((2,)) * 0.02).astype("float32")
    w_cv2 = (rng.standard_normal((5, 6, 1, 1)) * 0.08).astype("float32")
    b_cv2 = (rng.standard_normal((5,)) * 0.02).astype("float32")

    cv1 = conv2d_silu_ref(x, w_cv1, b_cv1)
    left = cv1[:, :2]
    right = cv1[:, 2:]
    h1 = conv2d_silu_ref(right, w1, b1, pads=(1, 1, 1, 1))
    h2 = conv2d_silu_ref(h1, w2, b2, pads=(1, 1, 1, 1))
    residual = right + h2
    expected = conv2d_silu_ref(np.concatenate([left, right, residual], axis=1), w_cv2, b_cv2)

    g = Graph("native_full_c2f_superblock")
    g.input("x", TensorSpec(x.shape, "float32"))
    for name, value in {
        "w_cv1": w_cv1,
        "b_cv1": b_cv1,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w_cv2": w_cv2,
        "b_cv2": b_cv2,
    }.items():
        g.const(name, value)
    g.node("Conv", "cv1_c", "x", "w_cv1", "b_cv1", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.sigmoid("cv1_s", "cv1_c")
    g.mul("cv1", "cv1_c", "cv1_s")
    g.node("Split", ["left", "right"], "cv1", axis=1, split=[2, 2])
    g.node("Conv", "m0_c1", "right", "w1", "b1", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("m0_s1", "m0_c1")
    g.mul("m0_h1", "m0_c1", "m0_s1")
    g.node("Conv", "m0_c2", "m0_h1", "w2", "b2", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("m0_s2", "m0_c2")
    g.mul("m0_h2", "m0_c2", "m0_s2")
    g.add("m0_out", "right", "m0_h2")
    g.node("Concat", "cat", "left", "right", "m0_out", axis=1)
    g.node("Conv", "cv2_c", "cat", "w_cv2", "b_cv2", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.sigmoid("cv2_s", "cv2_c")
    g.mul("y", "cv2_c", "cv2_s")
    g.output("y")

    session = InferenceSession(g, backend="native_d3d12", device="d3d12", output_numpy=True)
    y = session.run({"x": x})["y"]
    assert session.backend._compiled_c2f_superblock_count == 1
    assert session.backend._fused_c2f_residual_tail_count == 1
    assert session.backend._fused_concat_conv1x1_count == 1
    assert_close(y, expected, tol=4e-5)


def test_native_d3d12_fuses_c2f_bottleneck_front_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(85)
    x = (rng.standard_normal((1, 4, 5, 5)) * 0.2).astype("float32")
    right = x[:, 2:]
    w1 = (rng.standard_normal((2, 2, 3, 3)) * 0.08).astype("float32")
    b1 = (rng.standard_normal((2,)) * 0.02).astype("float32")
    w2 = (rng.standard_normal((2, 2, 3, 3)) * 0.08).astype("float32")
    b2 = (rng.standard_normal((2,)) * 0.02).astype("float32")
    h1 = conv2d_silu_ref(right, w1, b1, pads=(1, 1, 1, 1))
    h2 = conv2d_silu_ref(h1, w2, b2, pads=(1, 1, 1, 1))
    expected = right + h2

    g = Graph("native_c2f_bottleneck_front")
    g.input("x", TensorSpec(x.shape, "float32"))
    g.const("w1", w1)
    g.const("b1", b1)
    g.const("w2", w2)
    g.const("b2", b2)
    g.node("Split", ["left", "right"], "x", axis=1, split=[2, 2])
    g.node("Conv", "c1", "right", "w1", "b1", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("s1", "c1")
    g.mul("a1", "c1", "s1")
    g.node("Conv", "c2", "a1", "w2", "b2", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("s2", "c2")
    g.mul("a2", "c2", "s2")
    g.add("y", "right", "a2")
    g.output("y")

    old = os.environ.get("AEXRT_NATIVE_D3D12_C2F_FRONT")
    os.environ["AEXRT_NATIVE_D3D12_C2F_FRONT"] = "1"
    try:
        session = InferenceSession(g, backend="native_d3d12", device="d3d12", output_numpy=True)
        y = session.run({"x": x})["y"]
        assert session.backend._fused_c2f_bottleneck_count == 1
        assert_close(y, expected, tol=6e-5)
    finally:
        if old is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_C2F_FRONT", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_C2F_FRONT"] = old


def test_native_d3d12_conv1x1_linear_fast_path_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(82)
    device = NativeD3D12Device()
    x = (rng.standard_normal((1, 3, 4, 4)) * 0.25).astype("float32")
    w = (rng.standard_normal((5, 3, 1, 1)) * 0.1).astype("float32")
    b = (rng.standard_normal((5,)) * 0.05).astype("float32")
    expected = np.empty((1, 5, 4, 4), dtype="float32")
    for oc in range(5):
        expected[:, oc] = b[oc]
        for ic in range(3):
            expected[:, oc] += x[:, ic] * w[oc, ic, 0, 0]

    xb = device.upload(x, label="conv1x1_x")
    wb = device.upload(w, label="conv1x1_w")
    bb = device.upload(b, label="conv1x1_b")
    yb = device.allocate_uav(expected.nbytes, dtype="float32", shape=expected.shape, label="conv1x1_y")
    desc = {
        "batch": 1,
        "in_channels": 3,
        "in_h": 4,
        "in_w": 4,
        "out_channels": 5,
        "out_h": 4,
        "out_w": 4,
        "kernel_h": 1,
        "kernel_w": 1,
        "stride_h": 1,
        "stride_w": 1,
        "pad_top": 0,
        "pad_left": 0,
        "dilation_h": 1,
        "dilation_w": 1,
        "groups": 1,
    }
    device.dispatch_conv2d_float32_into(xb, wb, bb, yb, desc)
    assert_close(device.download(yb), expected, tol=2e-5)


def test_native_d3d12_session_run_yolo_gpu_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    head = np.zeros((6, 5), dtype="float32")
    head[:4, 0] = [10, 10, 10, 10]
    head[:4, 1] = [11, 10, 10, 10]
    head[:4, 2] = [50, 50, 8, 8]
    head[:4, 3] = [80, 80, 8, 8]
    head[:4, 4] = [70, 70, 8, 8]
    head[4:, 0] = [0.9, 0.1]
    head[4:, 1] = [0.8, 0.1]
    head[4:, 2] = [0.1, 0.7]
    head[4:, 3] = [0.2, 0.1]
    head[4:, 4] = [0.4, 0.2]

    g = Graph("native_session_yolo_gpu")
    g.input("head", TensorSpec(head.shape, "float32"))
    g.relu("y", "head")
    g.output("y")

    session = InferenceSession(g, backend="native_d3d12", optimize=False)
    dets = session.run_yolo_gpu(
        {"head": head},
        output_name="y",
        classes=2,
        max_candidates=8,
        max_detections=4,
        conf_threshold=0.25,
        iou_threshold=0.5,
    )
    assert dets.shape == (3, 6)
    assert [round(float(x), 2) for x in dets[:, 4]] == [0.9, 0.7, 0.4]
    assert [int(x) for x in dets[:, 5]] == [0, 1, 0]
    assert session.backend._prepared_yolo_graph_plan is not None


def test_native_d3d12_fuses_yolo_detect_head_tail_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    classes = 2
    scale_shapes = [(2, 2), (1, 1), (1, 1)]
    boxes = []
    scores = []
    for si, (h, w) in enumerate(scale_shapes):
        b = np.full((1, 64, h, w), -8.0, dtype="float32")
        for dim in range(4):
            b[:, dim * 16 + 2, :, :] = 8.0
        c = np.full((1, classes, h, w), -8.0, dtype="float32")
        c[:, si % classes, :, :] = 7.0 - si
        boxes.append(b)
        scores.append(c)

    anchors = []
    strides = []
    for stride, (h, w) in zip([8.0, 16.0, 32.0], scale_shapes):
        for y in range(h):
            for x in range(w):
                anchors.append((x + 0.5, y + 0.5))
                strides.append(stride)
    anchor_arr = np.asarray(anchors, dtype="float32").T.reshape(1, 2, -1)
    stride_arr = np.asarray(strides, dtype="float32").reshape(1, -1)

    g = Graph("native_yolo_detect_head_tail_superblock")
    for i, (b, c) in enumerate(zip(boxes, scores)):
        g.input(f"b{i}", TensorSpec(b.shape, "float32"))
        g.input(f"c{i}", TensorSpec(c.shape, "float32"))
        g.concat(f"raw{i}", f"b{i}", f"c{i}", axis=1)
        g.reshape(f"r{i}", f"raw{i}", [1, 64 + classes, -1])
    g.concat("all", "r0", "r1", "r2", axis=2)
    g.node("Split", ["box", "cls"], "all", axis=1, split=[64, classes])
    g.reshape("dfl_r", "box", [1, 4, 16, -1])
    g.sigmoid("cls_s", "cls")
    g.transpose("dfl_t", "dfl_r", [0, 2, 1, 3])
    g.softmax("dfl_sm", "dfl_t", axis=1)
    g.const("dfl_w", np.arange(16, dtype="float32").reshape(1, 16, 1, 1))
    g.node("Conv", "dfl_c", "dfl_sm", "dfl_w", name="/model.22/dfl/conv/Conv", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.reshape("dfl", "dfl_c", [1, 4, -1])
    g.node("Slice", "lt", "dfl", starts=[0], ends=[2], axes=[1], steps=[1])
    g.node("Slice", "rb", "dfl", starts=[2], ends=[4], axes=[1], steps=[1])
    g.const("anchor0", anchor_arr)
    g.const("anchor1", anchor_arr)
    g.const("two", np.asarray(2.0, dtype="float32"))
    g.const("stride", stride_arr)
    g.sub("xy0", "anchor0", "lt")
    g.add("xy1", "anchor1", "rb")
    g.add("xy_sum", "xy0", "xy1")
    g.sub("wh", "xy1", "xy0")
    g.div("xy", "xy_sum", "two")
    g.concat("xywh", "xy", "wh", axis=1)
    g.mul("scaled", "xywh", "stride")
    g.concat("output0", "scaled", "cls_s", axis=1)
    g.output("output0")

    feed = {f"b{i}": boxes[i] for i in range(3)}
    feed.update({f"c{i}": scores[i] for i in range(3)})
    old = os.environ.get("AEXRT_NATIVE_D3D12_YOLO_HEAD_SUPERBLOCK")
    try:
        os.environ.pop("AEXRT_NATIVE_D3D12_YOLO_HEAD_SUPERBLOCK", None)
        fused_session = InferenceSession(g, backend="native_d3d12", device="d3d12", optimize=False)
        fused = fused_session.run_yolo_gpu(feed, output_name="output0", classes=classes, max_candidates=16, max_detections=8, conf_threshold=0.01)
        os.environ["AEXRT_NATIVE_D3D12_YOLO_HEAD_SUPERBLOCK"] = "0"
        baseline_session = InferenceSession(g, backend="native_d3d12", device="d3d12", optimize=False)
        baseline = baseline_session.run_yolo_gpu(feed, output_name="output0", classes=classes, max_candidates=16, max_detections=8, conf_threshold=0.01)
        assert fused_session.backend._fused_yolo_detect_head_count == 1
        assert baseline_session.backend._fused_yolo_detect_head_count == 0
        assert fused.shape == baseline.shape
        assert_close(fused, baseline, tol=8e-5)
    finally:
        if old is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_YOLO_HEAD_SUPERBLOCK", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_YOLO_HEAD_SUPERBLOCK"] = old


def test_native_d3d12_backend_runs_single_relu_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    g = Graph("native_relu")
    g.input("x", TensorSpec((2, 3), "float32"))
    g.relu("y", "x")
    g.output("y")
    x = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype="float32")
    y = InferenceSession(g, backend="native_d3d12", optimize=False).run({"x": x})["y"]
    assert_close(y, np.maximum(x, 0))


def test_native_d3d12_backend_runs_relu_gelu_graph_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    g = Graph("native_relu_gelu")
    g.input("x", TensorSpec((2, 3), "float32"))
    g.relu("r", "x")
    g.gelu("y", "r")
    g.output("y")
    x = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype="float32")
    y = InferenceSession(g, backend="native_d3d12", optimize=False).run({"x": x})["y"]
    r = np.maximum(x, 0)
    expected = 0.5 * r * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (r + 0.044715 * np.power(r, 3))))
    assert_close(y, expected, tol=2e-5)


def test_native_d3d12_backend_runs_add_relu_graph_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    g = Graph("native_add_relu")
    g.input("a", TensorSpec((2, 3), "float32"))
    g.input("b", TensorSpec((2, 3), "float32"))
    g.add("sum", "a", "b")
    g.relu("y", "sum")
    g.output("y")
    a = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype="float32")
    b = np.array([[3.0, -4.0, 5.0], [-6.0, 7.0, -8.0]], dtype="float32")
    y = InferenceSession(g, backend="native_d3d12", optimize=False).run({"a": a, "b": b})["y"]
    assert_close(y, np.maximum(a + b, 0))


def test_native_d3d12_backend_fuses_conv_sigmoid_mul_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(78)
    x = (rng.standard_normal((1, 2, 5, 5)) * 0.25).astype("float32")
    w = (rng.standard_normal((3, 2, 3, 3)) * 0.1).astype("float32")
    b = (rng.standard_normal((3,)) * 0.05).astype("float32")
    g = Graph("native_conv_silu")
    g.input("x", TensorSpec(x.shape, "float32"))
    g.const("w", w)
    g.const("b", b)
    g.conv2d("c", "x", "w", "b", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("s", "c")
    g.mul("y", "c", "s")
    g.output("y")
    y = InferenceSession(g, backend="native_d3d12", optimize=False).run({"x": x})["y"]
    expected = conv2d_silu_ref(x, w, b, strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1), group=1)
    assert_close(y, expected, tol=3e-5)


def test_native_d3d12_conv3x3_winograd_silu_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(86)
    device = NativeD3D12Device()
    x = (rng.standard_normal((1, 3, 5, 7)) * 0.2).astype("float32")
    w = (rng.standard_normal((4, 3, 3, 3)) * 0.08).astype("float32")
    b = (rng.standard_normal((4,)) * 0.02).astype("float32")
    expected = conv2d_silu_ref(x, w, b, strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1), group=1)
    xb = device.upload(x, label="winograd_x")
    wb = device.upload(w, label="winograd_w")
    bb = device.upload(b, label="winograd_b")
    yb = device.allocate_uav(expected.nbytes, dtype="float32", shape=expected.shape, label="winograd_y")
    desc = {
        "batch": 1,
        "in_channels": 3,
        "in_h": 5,
        "in_w": 7,
        "out_channels": 4,
        "out_h": 5,
        "out_w": 7,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_top": 1,
        "pad_left": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "groups": 1,
    }
    old = os.environ.get("AEXRT_NATIVE_D3D12_WINOGRAD")
    os.environ["AEXRT_NATIVE_D3D12_WINOGRAD"] = "1"
    try:
        device.dispatch_conv2d_silu_float32_into(xb, wb, bb, yb, desc)
        assert_close(device.download(yb), expected, tol=8e-5)
    finally:
        if old is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_WINOGRAD", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_WINOGRAD"] = old


def test_native_d3d12_conv3x3_winograd_packed_silu_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(87)
    device = NativeD3D12Device()
    x = (rng.standard_normal((1, 3, 6, 6)) * 0.2).astype("float32")
    w = (rng.standard_normal((4, 3, 3, 3)) * 0.08).astype("float32")
    b = (rng.standard_normal((4,)) * 0.02).astype("float32")
    packed_w = _pack_winograd_f2x2_3x3_weights(w)
    expected = conv2d_silu_ref(x, w, b, strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1), group=1)
    xb = device.upload(x, label="winograd_packed_x")
    wb = device.upload(packed_w, label="winograd_packed_u")
    bb = device.upload(b, label="winograd_packed_b")
    yb = device.allocate_uav(expected.nbytes, dtype="float32", shape=expected.shape, label="winograd_packed_y")
    desc = {
        "batch": 1,
        "in_channels": 3,
        "in_h": 6,
        "in_w": 6,
        "out_channels": 4,
        "out_h": 6,
        "out_w": 6,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_top": 1,
        "pad_left": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "groups": 1,
    }
    old = os.environ.get("AEXRT_NATIVE_D3D12_WINOGRAD_PACKED")
    os.environ["AEXRT_NATIVE_D3D12_WINOGRAD_PACKED"] = "1"
    try:
        device.dispatch_conv2d_silu_float32_into(xb, wb, bb, yb, desc)
        assert_close(device.download(yb), expected, tol=8e-5)
    finally:
        if old is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_WINOGRAD_PACKED", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_WINOGRAD_PACKED"] = old


def test_native_d3d12_conv3x3_winograd_packed_oc4_silu_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(89)
    device = NativeD3D12Device()
    x = (rng.standard_normal((1, 3, 6, 6)) * 0.2).astype("float32")
    w = (rng.standard_normal((5, 3, 3, 3)) * 0.08).astype("float32")
    b = (rng.standard_normal((5,)) * 0.02).astype("float32")
    packed_w = _pack_winograd_f2x2_3x3_weights(w)
    expected = conv2d_silu_ref(x, w, b, strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1), group=1)
    xb = device.upload(x, label="winograd_oc4_x")
    wb = device.upload(packed_w, label="winograd_oc4_u")
    bb = device.upload(b, label="winograd_oc4_b")
    yb = device.allocate_uav(expected.nbytes, dtype="float32", shape=expected.shape, label="winograd_oc4_y")
    desc = {
        "batch": 1,
        "in_channels": 3,
        "in_h": 6,
        "in_w": 6,
        "out_channels": 5,
        "out_h": 6,
        "out_w": 6,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_top": 1,
        "pad_left": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "groups": 1,
    }
    old_packed = os.environ.get("AEXRT_NATIVE_D3D12_WINOGRAD_PACKED")
    old_oc4 = os.environ.get("AEXRT_NATIVE_D3D12_WINOGRAD_OC4")
    os.environ["AEXRT_NATIVE_D3D12_WINOGRAD_PACKED"] = "1"
    os.environ["AEXRT_NATIVE_D3D12_WINOGRAD_OC4"] = "1"
    try:
        device.dispatch_conv2d_silu_float32_into(xb, wb, bb, yb, desc)
        assert_close(device.download(yb), expected, tol=8e-5)
    finally:
        if old_packed is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_WINOGRAD_PACKED", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_WINOGRAD_PACKED"] = old_packed
        if old_oc4 is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_WINOGRAD_OC4", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_WINOGRAD_OC4"] = old_oc4


def test_native_d3d12_backend_uses_persistent_winograd_packed_weights_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(88)
    x = (rng.standard_normal((1, 2, 6, 6)) * 0.25).astype("float32")
    w = (rng.standard_normal((3, 2, 3, 3)) * 0.1).astype("float32")
    b = (rng.standard_normal((3,)) * 0.05).astype("float32")
    g = Graph("native_conv_silu_packed_winograd")
    g.input("x", TensorSpec(x.shape, "float32"))
    g.const("w", w)
    g.const("b", b)
    g.conv2d("c", "x", "w", "b", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.sigmoid("s", "c")
    g.mul("act", "c", "s")
    g.identity("y", "act")
    g.output("y")
    old = os.environ.get("AEXRT_NATIVE_D3D12_WINOGRAD_PACKED")
    os.environ["AEXRT_NATIVE_D3D12_WINOGRAD_PACKED"] = "1"
    try:
        session = InferenceSession(g, backend="native_d3d12", optimize=False)
        y = session.run({"x": x})["y"]
        expected = conv2d_silu_ref(x, w, b, strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1), group=1)
        assert len(session.backend._winograd_packed_constants) == 1
        assert_close(y, expected, tol=8e-5)
    finally:
        if old is None:
            os.environ.pop("AEXRT_NATIVE_D3D12_WINOGRAD_PACKED", None)
        else:
            os.environ["AEXRT_NATIVE_D3D12_WINOGRAD_PACKED"] = old


def test_native_d3d12_backend_fuses_conv_bn_silu_chain_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(80)
    x = (rng.standard_normal((1, 2, 5, 5)) * 0.25).astype("float32")
    w0 = (rng.standard_normal((3, 2, 3, 3)) * 0.1).astype("float32")
    b0 = (rng.standard_normal((3,)) * 0.05).astype("float32")
    scale0 = (rng.random((3,)) * 0.5 + 0.75).astype("float32")
    beta0 = (rng.standard_normal((3,)) * 0.05).astype("float32")
    mean0 = (rng.standard_normal((3,)) * 0.03).astype("float32")
    var0 = (rng.random((3,)) * 0.2 + 0.8).astype("float32")
    w1 = (rng.standard_normal((4, 3, 1, 1)) * 0.1).astype("float32")
    b1 = (rng.standard_normal((4,)) * 0.05).astype("float32")

    g = Graph("native_conv_bn_silu_chain")
    g.input("x", TensorSpec(x.shape, "float32"))
    for name, value in {
        "w0": w0,
        "b0": b0,
        "scale0": scale0,
        "beta0": beta0,
        "mean0": mean0,
        "var0": var0,
        "w1": w1,
        "b1": b1,
    }.items():
        g.const(name, value)
    g.conv2d("c0", "x", "w0", "b0", strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    g.batchnorm("bn0", "c0", "scale0", "beta0", "mean0", "var0", epsilon=1e-5)
    g.sigmoid("s0", "bn0")
    g.mul("h0", "bn0", "s0")
    g.conv2d("c1", "h0", "w1", "b1", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    g.sigmoid("s1", "c1")
    g.mul("y", "c1", "s1")
    g.output("y")

    fw0, fb0 = fold_bn_ref(w0, b0, scale0, beta0, mean0, var0, eps=1e-5)
    h0 = conv2d_silu_ref(x, fw0.astype("float32"), fb0.astype("float32"), strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1), group=1)
    expected = conv2d_silu_ref(h0, w1, b1, strides=(1, 1), pads=(0, 0, 0, 0), dilations=(1, 1), group=1)

    session = InferenceSession(g, backend="native_d3d12", optimize=False)
    y = session.run({"x": x})["y"]
    assert session.backend._conv_silu_plan["prepared"].op == "Conv2D+SiLU+ChainUploadRing"
    assert_close(y, expected, tol=4e-5)


def test_native_d3d12_backend_fuses_deeper_conv_silu_chain_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return
    rng = np.random.default_rng(81)
    x = (rng.standard_normal((1, 2, 5, 5)) * 0.25).astype("float32")
    channels = [2, 3, 3, 4, 2]
    weights = []
    biases = []
    for i in range(4):
        weights.append((rng.standard_normal((channels[i + 1], channels[i], 1, 1)) * 0.1).astype("float32"))
        biases.append((rng.standard_normal((channels[i + 1],)) * 0.05).astype("float32"))

    g = Graph("native_deep_conv_silu_chain")
    g.input("x", TensorSpec(x.shape, "float32"))
    last = "x"
    expected = x
    for i, (w, b) in enumerate(zip(weights, biases)):
        g.const(f"w{i}", w)
        g.const(f"b{i}", b)
        g.conv2d(f"c{i}", last, f"w{i}", f"b{i}", strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
        g.sigmoid(f"s{i}", f"c{i}")
        out = "y" if i == 3 else f"h{i}"
        g.mul(out, f"c{i}", f"s{i}")
        last = out
        expected = conv2d_silu_ref(expected, w, b, strides=(1, 1), pads=(0, 0, 0, 0), dilations=(1, 1), group=1)
    g.output("y")

    session = InferenceSession(g, backend="native_d3d12", optimize=False)
    y = session.run({"x": x})["y"]
    assert session.backend._conv_silu_plan["prepared"].op == "Conv2D+SiLU+ChainUploadRing"
    assert len(session.backend._conv_silu_plan["prepared"].descs) == 4
    assert_close(y, expected, tol=4e-5)


def test_native_d3d12_backend_prepares_when_available():
    info = NativeD3D12Device.probe()
    if not info.available:
        return

    g = Graph("native_prepare")
    g.input("x", TensorSpec((1, 4), "float32"))
    g.relu("y", "x")
    g.output("y")

    session = InferenceSession(g, backend="native_d3d12", optimize=False)
    backend_info = session.info()
    assert backend_info.name == "native_d3d12"
    assert backend_info.capabilities["api"] == "aexrt_native_d3d12"
    assert backend_info.capabilities["directml"] is False
    assert session.execution_plan() is not None
    assert session.schedule() is not None
    assert "lifetime_colored_arena_v1" in session.schedule().algorithm_tags


def test_auto_d3d12_does_not_fallback_to_directml_or_cpu():
    info = NativeD3D12Device.probe()
    if info.available:
        return

    g = Graph("d3d12_entry")
    g.input("x", TensorSpec((1,), "float32"))
    g.relu("y", "x")
    g.output("y")

    try:
        InferenceSession(g, backend="auto", device="d3d12", optimize=False)
    except RuntimeError as e:
        assert "native D3D12 runtime is not built" in str(e)
    else:
        raise AssertionError("device='d3d12' must not silently fall back to DirectML or CPU")


def test_attention_numpy_vs_torch():
    rng = np.random.default_rng(1)
    g = Graph("attn")
    for name in ["q", "k", "v"]:
        g.input(name, TensorSpec((2, 4, 8), "float32"))
    g.sdpa("y", "q", "k", "v", causal=True)
    g.output("y")
    feed = {name: rng.standard_normal((2, 4, 8)).astype("float32") for name in ["q", "k", "v"]}
    y_np = InferenceSession(g, backend="numpy", optimize=True).run(feed)["y"]
    y_torch = InferenceSession(g, backend="torch", device="auto", optimize=True).run(feed)["y"]
    assert_close(y_np, y_torch, tol=5e-4)


def test_rope_numpy_vs_torch():
    rng = np.random.default_rng(2)
    g = Graph("rope")
    g.input("x", TensorSpec((2, 4, 8), "float32"))
    g.const("cos", np.cos(np.linspace(0, 1, 4, dtype="float32")).reshape(1, 1, 4))
    g.const("sin", np.sin(np.linspace(0, 1, 4, dtype="float32")).reshape(1, 1, 4))
    g.rope("y", "x", "cos", "sin")
    g.output("y")
    x = rng.standard_normal((2, 4, 8)).astype("float32")
    y_np = InferenceSession(g, backend="numpy", optimize=True).run({"x": x})["y"]
    y_torch = InferenceSession(g, backend="torch", device="auto", optimize=True).run({"x": x})["y"]
    assert_close(y_np, y_torch)



def test_cuda_graph_matches_eager():
    import torch
    if not torch.cuda.is_available():
        return
    rng = np.random.default_rng(3)
    g = Graph("cuda_graph")
    g.input("x", TensorSpec((2, 6), "float32"))
    g.const("w", rng.standard_normal((6, 9)).astype("float32"))
    g.const("b", rng.standard_normal((9,)).astype("float32"))
    g.matmul("m", "x", "w")
    g.add("a", "m", "b")
    g.relu("y", "a")
    g.output("y")
    x = rng.standard_normal((2, 6)).astype("float32")
    eager = InferenceSession(g, backend="torch", device="cuda", output_numpy=True, cuda_graph=False).run({"x": x})["y"]
    captured_session = InferenceSession(g, backend="torch", device="cuda", output_numpy=True, cuda_graph=True)
    captured = captured_session.run({"x": x})["y"]
    captured2 = captured_session.run({"x": x + 0.25})["y"]
    eager2 = InferenceSession(g, backend="torch", device="cuda", output_numpy=True, cuda_graph=False).run({"x": x + 0.25})["y"]
    assert_close(eager, captured)
    assert_close(eager2, captured2)

if __name__ == "__main__":
    test_mlp_numpy_vs_torch()
    test_execution_plan_and_capabilities()
    test_memory_plan_infers_temporaries_and_reuses_allocations()
    test_scheduler_builds_arena_fusion_and_tile_plan()
    test_aexrt_graph_abi_exports_node_list()
    test_native_cpp_loads_python_exported_aexrt_json_when_available()
    test_native_cpp_fuses_relu_gelu_when_available()
    test_native_cpp_dynamic_fuses_elementwise_dag_when_available()
    test_native_cpp_fuses_scalar_constants_from_abi_json_when_available()
    test_native_cpp_runs_three_input_fused_dag_when_available()
    test_torch_backend_runs_vision_ops()
    test_yolo_postprocess_nms()
    test_unsupported_op_rejected_by_backend()
    test_auto_backend_creates_execution_plan()
    test_host_device_upload_download_roundtrip()
    test_native_d3d12_probe_is_not_directml_bridge()
    test_native_d3d12_upload_download_when_available()
    test_native_d3d12_relu_dispatch_when_available()
    test_native_d3d12_relu_dispatch_into_reuses_output_when_available()
    test_native_d3d12_prepared_relu_dispatch_when_available()
    test_native_d3d12_backend_runs_single_relu_when_available()
    test_native_d3d12_backend_runs_relu_gelu_graph_when_available()
    test_native_d3d12_backend_runs_add_relu_graph_when_available()
    test_native_d3d12_backend_prepares_when_available()
    test_auto_d3d12_does_not_fallback_to_directml_or_cpu()
    test_attention_numpy_vs_torch()
    test_rope_numpy_vs_torch()
    test_cuda_graph_matches_eager()
    print("all tests passed")
