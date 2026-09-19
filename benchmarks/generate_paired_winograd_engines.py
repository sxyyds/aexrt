from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))

from aexrt import Graph, TensorSpec, inspect_aexrt_engine, save_aexrt_engine  # noqa: E402
from aexrt import engine as engine_module  # noqa: E402


FUSION_PAIRED_CONV3X3_SILU = 1
PRECISION_FLOAT16 = 2
KERNEL_WINOGRAD_F2X2_3X3 = 40
PLAN_FLAG_AUTHORITATIVE = 1
PLAN_FLAG_DXIL = 2


def build_graph() -> Graph:
    graph = Graph("paired_winograd_10x10_256x64_benchmark")
    graph.input("images", TensorSpec((1, 256, 10, 10), "float32"))
    for suffix in ("0", "1"):
        graph.const(f"w{suffix}", np.zeros((64, 256, 3, 3), dtype="float32"))
        graph.const(f"b{suffix}", np.zeros((64,), dtype="float32"))
        graph.node(
            "Conv",
            f"c{suffix}",
            "images",
            f"w{suffix}",
            f"b{suffix}",
            strides=[1, 1],
            pads=[1, 1, 1, 1],
            dilations=[1, 1],
            group=1,
        )
        graph.sigmoid(f"s{suffix}", f"c{suffix}")
        graph.mul(f"a{suffix}", f"c{suffix}", f"s{suffix}")
    graph.node("Concat", "cat", "a0", "a1", axis=1)
    graph.reshape("output0", "cat", (1, 128, 100))
    graph.output("output0")
    return graph


def resolve_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def paired_groups(info: dict[str, object]) -> list[dict[str, int]]:
    return [
        group
        for group in info["fusion_plan"]  # type: ignore[index]
        if int(group["kind"]) == FUSION_PAIRED_CONV3X3_SILU
    ]


def assert_engine_plans(candidate_path: Path, baseline_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    candidate = inspect_aexrt_engine(candidate_path)
    baseline = inspect_aexrt_engine(baseline_path)
    expected_group = {
        "kind": FUSION_PAIRED_CONV3X3_SILU,
        "start": 0,
        "end": 1,
        "precision": PRECISION_FLOAT16,
        "kernel": KERNEL_WINOGRAD_F2X2_3X3,
        "flags": PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL,
        "aux0": 0,
        "aux1": 0,
    }
    if paired_groups(candidate) != [expected_group]:
        raise RuntimeError(f"candidate does not contain the expected paired fusion plan: {candidate_path}")
    if paired_groups(baseline):
        raise RuntimeError(f"baseline unexpectedly contains paired fusion kind 1: {baseline_path}")
    if baseline["fusion_plan"]:
        raise RuntimeError(f"baseline unexpectedly contains another fusion group: {baseline_path}")
    if int(candidate["command_count"]) != int(baseline["command_count"]):
        raise RuntimeError("candidate and baseline command streams differ in length")
    if candidate["source_hash"] != baseline["source_hash"]:
        raise RuntimeError("candidate and baseline were not generated from the same graph")
    if candidate["kernel_plan"] != baseline["kernel_plan"]:
        raise RuntimeError("candidate and baseline kernel plans differ")

    for name, info in (("candidate", candidate), ("baseline", baseline)):
        plans = info["kernel_plan"]
        if len(plans) < 2:
            raise RuntimeError(f"{name} engine is missing the two convolution plans")
        for index, plan in enumerate(plans[:2]):
            if int(plan[2]) != KERNEL_WINOGRAD_F2X2_3X3 or int(plan[3]) != PRECISION_FLOAT16:
                raise RuntimeError(
                    f"{name} command {index} is not FP16 Winograd: algorithm={plan[2]} precision={plan[3]}"
                )
    if int(candidate["packed_layout_counts"].get(3, 0)) != 2:
        raise RuntimeError("candidate does not contain two paired-Winograd packed weights (layout 3)")
    return candidate, baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate authoritative paired-Winograd candidate and unfused baseline AEXRT engines."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build" / "native" / "paired_winograd_engines",
        help="output directory (default: build/native/paired_winograd_engines)",
    )
    parser.add_argument("--candidate-name", default="paired_winograd_candidate.aexrt")
    parser.add_argument("--baseline-name", default="paired_winograd_baseline.aexrt")
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / args.candidate_name
    baseline_path = output_dir / args.baseline_name
    if candidate_path.resolve() == baseline_path.resolve():
        raise ValueError("candidate and baseline output paths must differ")
    graph = build_graph()

    save_aexrt_engine(graph, candidate_path, classes=124, precision="fp16")

    build_fusion_plan = engine_module._build_fusion_plan

    def without_paired_fusion(*plan_args: object, **plan_kwargs: object) -> list[dict[str, int]]:
        return [
            group
            for group in build_fusion_plan(*plan_args, **plan_kwargs)
            if int(group["kind"]) != FUSION_PAIRED_CONV3X3_SILU
        ]

    engine_module._build_fusion_plan = without_paired_fusion
    try:
        save_aexrt_engine(graph, baseline_path, classes=124, precision="fp16")
    finally:
        engine_module._build_fusion_plan = build_fusion_plan

    candidate, baseline = assert_engine_plans(candidate_path, baseline_path)
    print(
        f"candidate: {candidate_path} "
        f"fusion_kind1={len(paired_groups(candidate))} "
        f"packed_layout3={candidate['packed_layout_counts'].get(3, 0)}"
    )
    print(
        f"baseline:  {baseline_path} "
        f"fusion_kind1={len(paired_groups(baseline))} "
        f"packed_layout3={baseline['packed_layout_counts'].get(3, 0)}"
    )


if __name__ == "__main__":
    main()
