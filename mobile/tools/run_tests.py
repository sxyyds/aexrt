#!/usr/bin/env python3
"""One-command host gate for AEXRT-M.

  py mobile/tools/run_tests.py           # use existing build artifacts
  py mobile/tools/run_tests.py --build   # shaders + model + host build first

Runs the host Vulkan binary, then checks:
  1. head tensors vs ONNX Runtime (NRMSE/cosine),
  2. GPU decode candidates vs a NumPy reference,
  3. C sort+NMS vs the same candidate list.
Exit code 0 only if all three pass.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "build" / "tmp"
HOST = ROOT / "build" / "mobile" / "yolo_flat_host.exe"
MODEL = TMP / "cs2v8_full.bin"
INPUT = TMP / "input.f32"
OUT = TMP / "gate"


def run(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], text=True, encoding="mbcs",
                       errors="replace", **kw)
    if p.returncode != 0:
        raise SystemExit(f"FAILED ({p.returncode}): {cmd[0]}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--int8", action="store_true",
                    help="gate the W8A8 (AXM3) model instead of fp16")
    args = ap.parse_args()
    model = (TMP / "i8" / "cs2v8_full.bin") if args.int8 else MODEL

    if args.build:
        run([sys.executable, ROOT / "mobile/tools/build.py", "shaders"])
        run([sys.executable, ROOT / "mobile/tools/make_subsets.py"])
        run([sys.executable, ROOT / "mobile/tools/build.py", "host"])
    for p in (HOST, model):
        if not p.exists():
            raise SystemExit(f"missing {p}; run with --build")
    if not INPUT.exists():
        sys.path.insert(0, str(ROOT))
        import numpy as np
        INPUT.parent.mkdir(parents=True, exist_ok=True)
        np.random.default_rng(1234).random((1, 3, 320, 320), dtype=np.float32).tofile(INPUT)

    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*"):
        f.unlink()
    run([HOST, model, INPUT, OUT / "dets.bin", "1",
         "--conv-variant", args.variant,
         "--conf", "0.00001",
         "--dump-heads", OUT, "--dump-cands", OUT / "cands.bin"])
    extra = ["--int8"] if args.int8 else []
    run([sys.executable, ROOT / "mobile/tools/verify_host.py", "--heads", OUT] + extra)
    run([sys.executable, ROOT / "mobile/tools/verify_decode.py",
         "--oracle", OUT, "--cands", OUT / "cands.bin"])
    run([sys.executable, ROOT / "mobile/tools/verify_nms.py",
         "--cands", OUT / "cands.bin", "--dets", OUT / "dets.bin"])
    # N-set pipeline uses independent value arenas; results must be identical.
    run([HOST, model, INPUT, OUT / "dets_p2.bin", "5",
         "--conv-variant", args.variant, "--conf", "0.00001", "--pipeline", "2"])
    if (OUT / "dets.bin").read_bytes() != (OUT / "dets_p2.bin").read_bytes():
        raise SystemExit("pipeline N=2 output differs from single-set output")
    print("pipeline: N=2 output identical")

    # Streaming: per-frame input staging must reproduce the single-frame run
    # for the last frame.
    import numpy as np
    base = np.fromfile(INPUT, dtype=np.float32)
    f2 = np.clip(np.roll(base, 64) * 0.9 + 0.02, 0, 1).astype(np.float32)
    np.concatenate([base, f2]).tofile(OUT / "frames2.f32")
    f2.tofile(OUT / "frame2.f32")
    run([HOST, model, INPUT, OUT / "dets_stream.bin", "2",
         "--conv-variant", args.variant, "--conf", "0.00001",
         "--stream", "--frames", OUT / "frames2.f32"],
        env=dict(os.environ, AEXRT_STREAM_DUMP=str(OUT / "sd")))
    run([HOST, model, OUT / "frame2.f32", OUT / "dets_f2.bin", "1",
         "--conv-variant", args.variant, "--conf", "0.00001"])
    if (OUT / "dets_stream.bin").read_bytes() != (OUT / "dets_f2.bin").read_bytes():
        raise SystemExit("stream last-frame output differs from single-frame output")
    if (OUT / "sd_0.bin").read_bytes() != (OUT / "dets.bin").read_bytes():
        raise SystemExit("stream per-frame NMS output differs from single-frame output")
    print("stream: last frame + per-frame NMS identical to single-frame runs")

    # Library API (yolo_flat.c built with -DAEXRT_M_NO_MAIN): the same frame
    # must produce byte-identical detections to the CLI's auto-variant run.
    api_exe = ROOT / "build" / "mobile" / "example_api.exe"
    if api_exe.exists():
        run([HOST, model, OUT / "frame2.f32", OUT / "dets_f2_auto.bin", "1",
             "--conf", "0.00001"])
        run([api_exe, model, OUT / "dets_api.bin", OUT / "frame2.f32"],
            env=dict(os.environ, AEXRT_API_CONF="0.00001"))
        if (OUT / "dets_api.bin").read_bytes() != (OUT / "dets_f2_auto.bin").read_bytes():
            raise SystemExit("library API output differs from CLI output")
        print("api: detections identical to CLI")
    print("AEXRT-M host gate: PASS")


if __name__ == "__main__":
    main()
