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


def build_conv_engine(path: Path, dim: int, in_channels: int, out_channels: int, algorithm: int) -> None:
    graph = Graph(f"conv3x3_{dim}x{dim}_{in_channels}x{out_channels}_alg{algorithm}")
    graph.input("images", TensorSpec((1, in_channels, dim, dim), "float32"))
    graph.const("weight", np.zeros((out_channels, in_channels, 3, 3), dtype="float32"))
    graph.const("bias", np.zeros((out_channels,), dtype="float32"))
    graph.node(
        "Conv",
        "conv",
        "images",
        "weight",
        "bias",
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1,
    )
    graph.sigmoid("gate", "conv")
    graph.mul("act", "conv", "gate")
    graph.reshape("output0", "act", (1, out_channels, dim * dim))
    graph.output("output0")

    stable_conv_algorithm = engine_module._stable_conv_algorithm
    native_fp16_pos2 = engine_module._uses_native_fp16_pos2_40x40_64x64
    engine_module._stable_conv_algorithm = lambda params, *, silu: algorithm
    engine_module._uses_native_fp16_pos2_40x40_64x64 = lambda params: algorithm == 44
    try:
        save_aexrt_engine(graph, path, classes=out_channels - 4, precision="fp16")
    finally:
        engine_module._stable_conv_algorithm = stable_conv_algorithm
        engine_module._uses_native_fp16_pos2_40x40_64x64 = native_fp16_pos2
    actual_algorithm = int(inspect_aexrt_engine(path)["kernel_plan"][0][2])
    if actual_algorithm != algorithm:
        raise RuntimeError(f"requested kernel {algorithm}, engine contains kernel {actual_algorithm}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-kernel single-convolution AEXRT engines.")
    parser.add_argument(
        "--algorithms",
        type=int,
        nargs="+",
        help="only generate cases whose planned kernel ID is listed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build" / "native" / "hotspot_engines",
        help="output directory (default: build/native/hotspot_engines)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = (
        (20, 256, 64, 7),
        (20, 256, 64, 40),
        (20, 128, 64, 12),
        (20, 128, 64, 40),
        (10, 256, 64, 2),
        (10, 256, 64, 40),
        (40, 64, 64, 25),
        (40, 64, 64, 40),
        (40, 64, 64, 2),
        (40, 64, 64, 3),
        (40, 64, 64, 5),
        (40, 64, 64, 39),
        (40, 64, 64, 44),
        (20, 64, 64, 10),
        (20, 64, 64, 40),
        (10, 128, 128, 11),
        (10, 128, 128, 40),
        (10, 64, 64, 2),
        (10, 64, 64, 40),
    )
    selected_algorithms = set(args.algorithms) if args.algorithms else None
    for dim, in_channels, out_channels, algorithm in cases:
        if selected_algorithms is not None and algorithm not in selected_algorithms:
            continue
        name = f"conv{dim}_{in_channels}x{out_channels}_alg{algorithm}.aexrt"
        build_conv_engine(output_dir / name, dim, in_channels, out_channels, algorithm)
        print(output_dir / name)


if __name__ == "__main__":
    main()
