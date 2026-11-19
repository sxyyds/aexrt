"""Activation calibration for AEXRT-M W8A8.

Runs ONNX Runtime (CPU) on a deterministic synthetic batch and records the
per-tensor absolute maximum. The exporter consumes build/tmp/act_amax.json
and converts it into per-value activation scales.

Real deployment should pass real frames; the synthetic set exercises the
network and keeps the pipeline deterministic for CI.

  py mobile/tools/calibrate.py --model models/cs2V8_320.onnx --runs 8
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/cs2V8_320.onnx"))
    ap.add_argument("--out", default=str(ROOT / "build/tmp/act_amax.json"))
    ap.add_argument("--runs", type=int, default=8)
    args = ap.parse_args()

    model = onnx.load(args.model)
    graph = model.graph
    names = []
    seen = set()
    for v in list(graph.value_info) + list(graph.output):
        if v.name in seen or v.name in {i.name for i in graph.initializer}:
            continue
        dims = v.type.tensor_type.shape.dim
        if not dims:
            continue
        seen.add(v.name)
        names.append(v.name)
    # Fetch only tensors that ORT can materialize (all value_info entries).
    keep = [v for v in names if v not in {o.name for o in graph.output}]
    graph.output.extend([onnx.ValueInfoProto(name=n) for n in keep])
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_shape = [1 if d is None or isinstance(d, str) else d
                for d in sess.get_inputs()[0].shape]
    out_names = [o.name for o in sess.get_outputs()]
    amax = {}
    rng = np.random.default_rng(0xC2A11)
    for run in range(args.runs):
        x = rng.random(in_shape, dtype=np.float32)
        outs = sess.run(out_names, {in_name: x})
        im = float(np.abs(x).max()) if x.size else 0.0
        if im > amax.get(in_name, 0.0):
            amax[in_name] = im
        for name, arr in zip(out_names, outs):
            m = float(np.abs(arr).max()) if arr.size else 0.0
            if m > amax.get(name, 0.0):
                amax[name] = m
        print(f"calib {run + 1}/{args.runs}: {len(out_names)} tensors")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(amax, open(args.out, "w"), indent=1, sort_keys=True)
    top = sorted(amax.items(), key=lambda kv: -kv[1])[:10]
    print(f"wrote {args.out}: {len(amax)} tensors")
    for k, v in top:
        print(f"  {v:8.3f}  {k}")


if __name__ == "__main__":
    main()
