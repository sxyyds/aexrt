"""Numeric gate: AEXRT-M GPU head tensors vs ONNX Runtime CPU.

  py mobile/tools/verify_host.py --model models/cs2V8_320.onnx \
      --input build/tmp/input.f32 --heads build/tmp/heads

The runtime must have been run with --dump-heads. Compares the six pre-decode
head tensors (3 levels x box/cls), position-major fp16 output vs NCHW fp32
ORT reference, and fails on large deviation.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
LEVEL_NAMES = [f"/model.22/cv2.{L}/cv2.{L}.2/Conv" for L in range(3)] + \
              [f"/model.22/cv3.{L}/cv3.{L}.2/Conv" for L in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/cs2V8_320.onnx"))
    ap.add_argument("--input", default=str(ROOT / "build/tmp/input.f32"))
    ap.add_argument("--heads", default=str(ROOT / "build/tmp/heads"))
    ap.add_argument("--int8", action="store_true",
                    help="W8A8 model: gate on quantized tolerance instead of fp16")
    args = ap.parse_args()
    np.set_printoptions(precision=5, suppress=True)

    m = onnx.load(args.model)
    name_to_out = {}
    for n in m.graph.node:
        if n.op_type == "Conv":
            name_to_out[n.name] = n.output[0]
    outs = [name_to_out[nm] for nm in LEVEL_NAMES]
    g = m.graph
    del g.output[:]
    for o in outs:
        g.output.extend([onnx.ValueInfoProto(name=o)])
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.fromfile(args.input, dtype=np.float32).reshape(1, 3, 320, 320)
    ref = sess.run(outs, {sess.get_inputs()[0].name: x})
    heads = Path(args.heads)

    worst = 0.0
    ok = True
    for L in range(3):
        for j, tag in enumerate(("box", "cls")):
            r = ref[j * 3 + L] if j == 0 else ref[3 + L]
            p = heads / f"L{L}_{tag}.f32"
            if not p.exists():
                print(f"missing {p}")
                ok = False
                continue
            got = np.fromfile(p, dtype=np.float32)
            # ORT: [1,C,H,W] -> [H*W,C] position-major
            want = np.transpose(r[0], (1, 2, 0)).reshape(-1, r.shape[1])
            got = got.reshape(want.shape)
            diff = got - want
            nrmse = float(np.linalg.norm(diff) / max(np.linalg.norm(want), 1e-9))
            cos = float(np.sum(got * want) /
                        max(np.linalg.norm(got) * np.linalg.norm(want), 1e-9))
            mx = float(np.abs(diff).max())
            worst = max(worst, mx)
            # fp16 storage + fp32 accumulation: the gate is on relative energy,
            # not near-zero elements (where relative error is meaningless).
            # W8A8: per-tensor activation scales put the heads at 2-5% NRMSE;
            # the gate catches structural breakage, not quantization noise.
            if args.int8:
                passed = nrmse < 0.08 and cos > 0.995 and mx < 2.0
            else:
                passed = nrmse < 5e-3 and cos > 0.9999 and mx < 0.1
            ok = ok and passed
            print(f"L{L} {tag}: shape {want.shape} maxabs {mx:.5f} "
                  f"nrmse {nrmse:.6f} cos {cos:.7f} {'PASS' if passed else 'FAIL'}")
    print("verify:", "PASS" if ok else "FAIL", f"worst abs {worst:.5f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
