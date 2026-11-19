"""Validate the C sort+NMS against a NumPy reference on the same candidate list.

  py mobile/tools/verify_nms.py --cands build/tmp/cands.bin --dets build/tmp/dets.bin
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:4], "little")
    return np.frombuffer(raw[4:4 + n * 24], dtype=np.float32).reshape(n, 6).copy()


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = ix2 - ix1; ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uni if uni > 0 else 0.0


def nms(cand, thr, topk=128):
    order = sorted(range(len(cand)),
                   key=lambda i: (-cand[i, 4], cand[i, 5], cand[i, 0],
                                  cand[i, 1], cand[i, 2], cand[i, 3]))
    # match runtime: top-K per class before the O(n^2) pass
    per = {}
    sel = []
    for i in order:
        c = int(round(float(cand[i, 5])))
        if c < 0 or c >= 4:
            continue
        if per.get(c, 0) >= topk:
            continue
        per[c] = per.get(c, 0) + 1
        sel.append(i)
    cand = cand[sel]
    order = list(range(len(cand)))
    dead = [False] * len(cand)
    keep = []
    for ii, i in enumerate(order):
        if dead[i]:
            continue
        keep.append(i)
        for j in order[ii + 1:]:
            if dead[j] or cand[j, 5] != cand[i, 5]:
                continue
            if iou(cand[i], cand[j]) > thr:
                dead[j] = True
    return np.array([cand[i] for i in keep], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=str(ROOT / "build/tmp/cands.bin"))
    ap.add_argument("--dets", default=str(ROOT / "build/tmp/dets.bin"))
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()
    cand = load(args.cands)
    want = nms(cand, args.iou)
    got = load(args.dets)
    print(f"candidates {len(cand)}, nms gpu {len(got)}, nms ref {len(want)}")
    if len(got) != len(want):
        print("FAIL: kept count differs")
        return 1
    # stable ordering-independent comparison
    key = lambda a: (int(round(a[5])), round(float(a[0]), 2), round(float(a[1]), 2),
                     round(float(a[4]), 6))
    ga = sorted(got.tolist(), key=key)
    wa = sorted(want.tolist(), key=key)
    diff = np.abs(np.array(ga, dtype=np.float64) - np.array(wa, dtype=np.float64))
    ok = diff.max() <= 1e-3
    print(f"max abs diff {diff.max():.6f} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
