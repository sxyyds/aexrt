#!/usr/bin/env python3
"""Push AEXRT-M to an Android device over adb and run the on-device gate.

  py mobile/tools/deploy.py --serial <id> [--iters 50] [--profile]
  py mobile/tools/deploy.py --check          # adb devices only

Build first: py mobile/tools/build.py android
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "mobile" / "yolo_flat"
MODEL = ROOT / "build" / "tmp" / "cs2v8_full.bin"
INPUT = ROOT / "build" / "tmp" / "input.f32"
REMOTE = "/data/local/tmp/aexrt-m"


def adb(args, serial=None, **kw):
    cmd = ["adb"] + (["-s", serial] if serial else []) + args
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, text=True, encoding="mbcs", errors="replace", **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial")
    ap.add_argument("--iters", default="50")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--conf", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--pipeline", default=None)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--emit-policy", dest="emit_policy", default=None)
    args = ap.parse_args()
    model = Path(args.model)
    inp = Path(args.input)
    policy = Path(args.policy) if args.policy else None
    emit = Path(args.emit_policy) if args.emit_policy else None
    frames = Path(args.frames) if args.frames else None
    if args.check:
        adb(["devices", "-l"])
        return 0
    devices = adb(["devices"], stdout=subprocess.PIPE).stdout
    if len([l for l in devices.splitlines() if "\tdevice" in l]) == 0:
        print("未执行：没有连接的 adb 设备。")
        print("复现命令：")
        print(f"  adb push {BIN} {REMOTE}")
        print(f"  adb push {model} /data/local/tmp/")
        print(f"  adb push {inp} /data/local/tmp/")
        print(f'  adb shell "cd /data/local/tmp && chmod +x aexrt-m && '
              f'./aexrt-m {model.name} {inp.name} dets.bin {args.iters} --autotune'
              f'{" --profile" if args.profile else ""}"')
        return 2
    for local, remote in ((BIN, REMOTE),
                          (model, f"/data/local/tmp/{model.name}"),
                          (inp, f"/data/local/tmp/{inp.name}")):
        if not local.exists():
            raise SystemExit(f"missing {local}")
        if adb(["push", str(local), remote], serial=args.serial).returncode:
            raise SystemExit("adb push failed")
    if policy:
        if adb(["push", str(policy), f"/data/local/tmp/{policy.name}"],
               serial=args.serial).returncode:
            raise SystemExit("adb push failed")
    if frames:
        if adb(["push", str(frames), f"/data/local/tmp/{frames.name}"],
               serial=args.serial).returncode:
            raise SystemExit("adb push failed")
    cmd = f'cd /data/local/tmp && chmod +x aexrt-m && ./aexrt-m ' \
          f'{model.name} {inp.name} dets.bin {args.iters} --autotune'
    if args.profile:
        cmd += " --profile"
    if args.conf:
        cmd += f" --conf {args.conf}"
    if args.variant:
        cmd += f" --conv-variant {args.variant}"
    if args.pipeline:
        cmd += f" --pipeline {args.pipeline}"
    if args.stream:
        cmd += " --stream"
    if frames:
        cmd += f" --frames {frames.name}"
    if policy:
        cmd += f" --policy {policy.name}"
    if emit:
        cmd += f" --emit-policy {emit.name}"
    rc = adb(["shell", cmd], serial=args.serial).returncode
    if emit:
        adb(["pull", f"/data/local/tmp/{emit.name}", str(emit)],
            serial=args.serial)
    adb(["pull", "/data/local/tmp/dets.bin", str(ROOT / "build/tmp/dets_android.bin")],
        serial=args.serial)
    return rc


if __name__ == "__main__":
    sys.exit(main())
