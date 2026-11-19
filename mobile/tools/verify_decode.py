"""Validate GPU decode (DFL + sigmoid + box mapping) against a NumPy reference.

  py mobile/tools/verify_decode.py --oracle build/tmp/heads --cands build/tmp/cands.bin

Uses the ONNX Runtime head dumps from verify_host.py's oracle path. Run the
runtime with --conf 0.00001 so every anchor is emitted, then compare the two
candidate sets after sorting (atomic compaction order is not deterministic).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STRIDES = {0: 8, 1: 16, 2: 32}
NC = 4


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dfl(box):
    # box: [N, 64] -> [N, 4, 16] -> softmax over 16 -> expectation
    b = box.reshape(-1, 4, 16).astype(np.float64)
    b = b - b.max(axis=2, keepdims=True)
    e = np.exp(b)
    p = e / e.sum(axis=2, keepdims=True)
    return (p * np.arange(16)).sum(axis=2)


def oracle_candidates(heads: Path):
    out = []
    for L in range(3):
        box = np.fromfile(heads / f"L{L}_box.f32", dtype=np.float32).reshape(-1, 64)
        cls = np.fromfile(heads / f"L{L}_cls.f32", dtype=np.float32).reshape(-1, NC)
        sc = sigmoid(cls)
        bc = sc.argmax(axis=1)
        best = sc.max(axis=1)
        d = dfl(box)
        n = box.shape[0]
        gw = int(round(np.sqrt(n)))
        gy, gx = np.divmod(np.arange(n), gw)
        gx = gx + 0.5
        gy = gy + 0.5
        st = 320.0 / gw
        out.append(np.stack([
            (gx - d[:, 0]) * st, (gy - d[:, 1]) * st,
            (gx + d[:, 2]) * st, (gy + d[:, 3]) * st,
            best, bc.astype(np.float32)], axis=1).astype(np.float32))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=str(ROOT / "build/tmp/heads"))
    ap.add_argument("--cands", default=str(ROOT / "build/tmp/cands.bin"))
    ap.add_argument("--atol", type=float, default=0.05)
    args = ap.parse_args()
    raw = Path(args.cands).read_bytes()
    n = int.from_bytes(raw[:4], "little")
    got = np.frombuffer(raw[4:4 + n * 24], dtype=np.float32).reshape(n, 6)
    want = oracle_candidates(Path(args.oracle))
    print(f"candidates: gpu {n}, oracle {want.shape[0]}")
    if n != want.shape[0]:
        print("FAIL: candidate count differs")
        return 1
    # sort by score desc then position + class so both sides align
    ko = np.lexsort((want[:, 5], want[:, 4], want[:, 1], want[:, 0]))
    kg = np.lexsort((got[:, 5], got[:, 4], got[:, 1], got[:, 0]))
    a, b = got[kg], want[ko]
    diff = np.abs(a - b)
    worst = diff[:, :4].max()
    cls_ok = np.array_equal(a[:, 5].round().astype(int), b[:, 5].round().astype(int))
    score_err = diff[:, 4].max()
    ok = worst <= args.atol and score_err <= args.atol and cls_ok
    print(f"box maxabs {worst:.5f} score maxabs {score_err:.5f} "
          f"classes {'match' if cls_ok else 'MISMATCH'} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        bad = np.argmax(diff[:, :4].max(axis=1))
        print("worst sample gpu", a[bad], "oracle", b[bad])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
