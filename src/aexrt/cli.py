from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .engine import inspect_aexrt_engine, install_aexrt_pipeline_cache, save_aexrt_engine_from_onnx


_OBJECTNESS_VALUES = {
    "auto": None,
    "yes": True,
    "no": False,
}


def _build(args: argparse.Namespace) -> int:
    model = Path(args.model)
    output = Path(args.output) if args.output else model.with_suffix(".aexrt")
    algo_overrides = None
    if getattr(args, "algo_map", None):
        import json

        with open(args.algo_map, "r", encoding="utf-8") as handle:
            algo_overrides = {int(k): int(v) for k, v in json.load(handle).items()}
    info = save_aexrt_engine_from_onnx(
        model,
        output,
        default_batch=args.batch,
        algo_overrides=algo_overrides,
        classes=args.classes,
        objectness=_OBJECTNESS_VALUES[args.objectness],
        max_candidates=args.max_candidates,
        max_detections=args.max_detections,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        precision=args.precision,
    )
    if not args.no_pipeline_cache:
        from .cpp_runtime import NativeCppYoloModel

        runtime = NativeCppYoloModel(output)
        try:
            runtime.run(np.zeros(runtime.input_element_count, dtype=np.float32), max_detections=1)
            info = install_aexrt_pipeline_cache(output, runtime.export_pipeline_cache())
        finally:
            runtime.close()
    print(f"wrote {output}")
    print(
        f"engine v{info['version']} precision={'fp16' if info['precision'] == 2 else 'fp32'} values={info['value_count']} "
        f"commands={info['command_count']} constants={info['constant_count']} "
        f"arena={info['arena_nbytes']} bytes file={info['file_size']} bytes "
        f"fusions={info['fusion_group_count']} shaders={info['shader_cache_count']} "
        f"dxil={info['dxil_cache_count']} dxbc={info['dxbc_cache_count']} pso={info['pso_cache_count']}"
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    info = inspect_aexrt_engine(args.engine)
    print(
        f"AEXRT engine v{info['version']} target=d3d12 precision={'fp16' if info['precision'] == 2 else 'fp32'} "
        f"input_elements={info['input_elements']} classes={info['classes']} anchors={info['anchors']}"
    )
    print(
        f"values={info['value_count']} commands={info['command_count']} "
        f"constants={info['constant_count']} arena={info['arena_nbytes']} bytes "
        f"file={info['file_size']} bytes fusions={info['fusion_group_count']} "
        f"shaders={info['shader_cache_count']} dxil={info['dxil_cache_count']} "
        f"dxbc={info['dxbc_cache_count']} pso={info['pso_cache_count']}"
    )
    print(f"source_sha256={info['source_hash']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aexrtc", description="Build and inspect AEXRT binary engines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="compile an ONNX model into a .aexrt engine")
    build.add_argument("model")
    build.add_argument("-o", "--output", default=None)
    build.add_argument("--classes", type=int, default=None)
    build.add_argument("--objectness", choices=tuple(_OBJECTNESS_VALUES), default="auto")
    build.add_argument("--max-candidates", type=int, default=512)
    build.add_argument("--max-detections", type=int, default=100)
    build.add_argument("--conf", type=float, default=0.25)
    build.add_argument("--iou", type=float, default=0.45)
    build.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    build.add_argument("--batch", type=int, default=1, help="dynamic batch dim (-1) resolution (default: 1)")
    build.add_argument("--algo-map", default=None,
                       help="JSON {command_index: kernel_id} from conv_autotune --emit-map; freezes per-device tuned algorithms into the engine")
    build.add_argument("--no-pipeline-cache", action="store_true")
    build.set_defaults(handler=_build)

    inspect = subparsers.add_parser("inspect", help="inspect a .aexrt engine")
    inspect.add_argument("engine")
    inspect.set_defaults(handler=_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
