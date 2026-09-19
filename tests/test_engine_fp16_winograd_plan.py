import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import aexrt.engine as engine_module
from aexrt import Graph, TensorSpec, save_aexrt_engine


def _conv_params(
    *,
    in_channels=128,
    out_channels=128,
    in_height=20,
    in_width=20,
    out_height=20,
    out_width=20,
    kernel=3,
    stride=1,
    pad=1,
    dilation=1,
    groups=1,
):
    return [
        1,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel,
        kernel,
        stride,
        stride,
        pad,
        pad,
        dilation,
        dilation,
        groups,
    ]


@pytest.mark.parametrize(
    "params",
    (
        _conv_params(),
        _conv_params(in_channels=96, out_channels=64),
        _conv_params(in_height=20, in_width=18, out_height=20, out_width=18),
    ),
)
def test_fp16_winograd_gate_accepts_aligned_even_outputs(params):
    assert engine_module._uses_native_fp16_winograd(params)


@pytest.mark.parametrize(
    "params",
    (
        _conv_params(in_channels=80),
        _conv_params(in_height=19, out_height=19),
        _conv_params(in_width=19, out_width=19),
        _conv_params(kernel=1, pad=0),
        _conv_params(stride=2),
        _conv_params(pad=0),
        _conv_params(dilation=2),
        _conv_params(groups=2),
    ),
)
def test_fp16_winograd_gate_rejects_unaligned_or_non_winograd_convs(params):
    assert not engine_module._uses_native_fp16_winograd(params)


def test_fp16_winograd_pack_helper_is_idempotent_and_transforms_quantized_weights():
    in_channels = out_channels = 32
    weight_id = 1
    weights = (
        np.arange(out_channels * in_channels * 9, dtype=np.float32).reshape(
            out_channels, in_channels, 3, 3
        )
        / 4096.0
    )
    values = [
        {"raw": b""},
        {"raw": weights.tobytes(order="C")},
    ]
    packed = {}
    for _ in range(2):
        engine_module._add_packed_winograd_f2x2_fp16(
            packed,
            values,
            weight_id=weight_id,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    assert list(packed) == [weight_id]
    item = packed[weight_id]
    assert item["layout"] == engine_module.PACKED_LAYOUT_WINOGRAD_F2X2_FP16
    transform = np.asarray(
        ((1.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    quantized = weights.astype("<f2").astype(np.float32)
    expected = np.einsum(
        "ak,oikl,bl->oiab", transform, quantized, transform, optimize=True
    ).astype("<f2")
    actual = np.frombuffer(item["data"], dtype="<f2").reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)


def test_fp32_accum_winograd_pack_preserves_transformed_fp16_source():
    in_channels = out_channels = 32
    weight_id = 1
    weights = (
        np.arange(out_channels * in_channels * 9, dtype=np.float32).reshape(
            out_channels, in_channels, 3, 3
        )
        / 4096.0
    )
    values = [{"raw": b""}, {"raw": weights.tobytes(order="C")}]
    packed = {}
    engine_module._add_packed_winograd_f2x2_fp32(
        packed,
        values,
        weight_id=weight_id,
        in_channels=in_channels,
        out_channels=out_channels,
    )

    item = packed[weight_id]
    assert item["layout"] == engine_module.PACKED_LAYOUT_WINOGRAD_F2X2_FP32
    transform = np.asarray(
        ((1.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    quantized = weights.astype("<f2").astype(np.float32)
    expected = np.einsum(
        "ak,oikl,bl->oiab", transform, quantized, transform, optimize=True
    ).astype("<f4")
    actual = np.frombuffer(item["data"], dtype="<f4").reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)


def _c3_residual_graph(name, *, channels, dim, first_kernel):
    graph = Graph(name)
    graph.input("images", TensorSpec((1, channels, dim, dim), "float32"))
    for suffix, kernel in (("0", first_kernel), ("1", 3)):
        graph.const(
            f"w{suffix}",
            np.zeros((channels, channels, kernel, kernel), dtype="float32"),
        )
        graph.const(f"b{suffix}", np.zeros((channels,), dtype="float32"))
        graph.node(
            "Conv",
            f"c{suffix}",
            "images" if suffix == "0" else "a0",
            f"w{suffix}",
            f"b{suffix}",
            strides=[1, 1],
            pads=[kernel // 2] * 4,
            dilations=[1, 1],
            group=1,
        )
        graph.sigmoid(f"s{suffix}", f"c{suffix}")
        graph.mul(f"a{suffix}", f"c{suffix}", f"s{suffix}")
    graph.add("residual", "images", "a1")
    graph.reshape("output0", "residual", (1, channels, dim * dim))
    graph.output("output0")
    return graph


@pytest.mark.parametrize("first_kernel", (1, 3))
def test_fp16_c3_tail_accepts_1x1_or_3x3_front_and_embeds_winograd_weight(
    tmp_path, first_kernel
):
    graph = _c3_residual_graph(
        f"fp16_c3_front_{first_kernel}x{first_kernel}",
        channels=64,
        dim=20,
        first_kernel=first_kernel,
    )
    info = save_aexrt_engine(graph, tmp_path / f"c3_{first_kernel}.aexrt", classes=60)
    groups = [group for group in info["fusion_plan"] if group["kind"] == 2]
    # The residual path still reads the FP32 graph input.  Homogeneous physical
    # inputs therefore keep this synthetic tail at an FP32 boundary even when
    # the front convolution itself is Winograd-capable.
    tail_flags = 3
    assert groups == [
        {
            "kind": 2,
            "start": 1,
            "end": 2,
            "precision": 2,
            "kernel": 40,
            "flags": tail_flags,
            "aux0": 0,
            "aux1": 0,
        }
    ]
    assert info["kernel_plan"][1][2] == 40
    assert info["kernel_plan"][1][3] == 2
    assert info["kernel_plan"][1][4] != (1 << 32) - 1
    assert info["kernel_plan"][1][5] == 3
    assert info["packed_layout_counts"] == {3: 1 if first_kernel == 1 else 2}
    assert info["arena_storage_precision_counts"]["fp16"] == 0


def test_fp16_c3_tail_advertises_typed_winograd_when_shape_capable_even_if_not_planned_id40(tmp_path):
    """形状支持 native fp16 winograd 时，即使单独执行的 planner 不选 40，
    C2F tail 也产 typed 融合组（融合消 dispatch 的收益压过单 kernel 偏好；
    数值正确性由 fail-closed + 门禁保证，实测 12 融合 vs 10 融合快 ~0.5ms）。"""
    graph = _c3_residual_graph(
        "fp16_c3_unplanned_winograd", channels=32, dim=40, first_kernel=1
    )
    info = save_aexrt_engine(graph, tmp_path / "c3_unplanned.aexrt", classes=28)
    group = next(group for group in info["fusion_plan"] if group["kind"] == 2)
    assert (group["precision"], group["kernel"], group["flags"]) == (2, 40, 3)
    assert info["kernel_plan"][1][2] != 40
    assert info["packed_layout_counts"] == {3: 1}


@pytest.mark.parametrize(
    ("channels", "dim", "expected_kernel"),
    (
        (64, 40, 44),
        (256, 10, 46),
    ),
)
def test_fp16_c3_tail_skips_measured_slow_v5_fusions(
    tmp_path, channels, dim, expected_kernel
):
    graph = _c3_residual_graph(
        f"fp16_c3_v5_separate_{channels}_{dim}",
        channels=channels,
        dim=dim,
        first_kernel=1,
    )
    info = save_aexrt_engine(
        graph, tmp_path / f"c3_v5_separate_{channels}_{dim}.aexrt", classes=channels - 4
    )

    assert not [group for group in info["fusion_plan"] if group["kind"] == 2]
    assert info["kernel_plan"][1][2] == expected_kernel
    if expected_kernel == 46:
        assert info["kernel_plan"][1][3] == 2
        assert info["kernel_plan"][1][4] != (1 << 32) - 1
        assert info["kernel_plan"][1][5] == 3
        assert info["packed_layout_counts"] == {4: 1, 5: 1}


@pytest.mark.parametrize(("channels", "dim"), ((64, 20), (128, 10)))
def test_fp16_c3_tail_keeps_measured_id40_fusions(tmp_path, channels, dim):
    graph = _c3_residual_graph(
        f"fp16_c3_keep_id40_{channels}_{dim}",
        channels=channels,
        dim=dim,
        first_kernel=1,
    )
    info = save_aexrt_engine(
        graph, tmp_path / f"c3_keep_id40_{channels}_{dim}.aexrt", classes=channels - 4
    )

    group = next(group for group in info["fusion_plan"] if group["kind"] == 2)
    assert (group["precision"], group["kernel"], group["flags"]) == (2, 40, 3)
    assert info["kernel_plan"][1][2] == 40
    assert info["packed_layout_counts"] == {3: 1}


def test_fp16_c3_tail_drops_group_when_winograd_gate_rejects_shape(tmp_path):
    """形状不支持（ic=48 非 %32）：不产融合组（退化为逐命令执行）。
    旧契约（un-typed (2,0,1) 组）会产出运行时被 fail-closed 拒绝的物理记录。"""
    graph = _c3_residual_graph(
        "fp16_c3_unaligned_channels", channels=48, dim=20, first_kernel=1
    )
    info = save_aexrt_engine(graph, tmp_path / "c3_fallback.aexrt", classes=44)
    assert not [group for group in info["fusion_plan"] if group["kind"] == 2]
    assert info["packed_layout_counts"] == {}
