"""Per-kernel roofline analysis from native_yolo_package --profile output.

Parses GPU dispatch profile lines like
    [ 38] 0.0411 ms  cmd#59 CONV_SILU 64x40x40 -> 64x40x40 k3s1 alg=...
computes minimum traffic (input read + output write, fp32) and MACs, and
reports achieved GB/s / TFLOPS per event plus per-algorithm aggregates.

Usage:
    py benchmarks\\roofline_report.py build\\tmp\\baseline_cs2v8_profile.txt
"""

from __future__ import annotations

import math
import re
import sys
from collections import defaultdict

EVENT_RE = re.compile(
    r"^\s*\[\s*(\d+)\]\s+([0-9.]+)\s+ms\s+(.*)$"
)
CONV_RE = re.compile(
    r"cmd#(\d+)\s+(\S+)\s+(\d+)x(\d+)x(\d+)\s+->\s+(\d+)x(\d+)x(\d+)\s+k(\d+)s(\d+)(?:\s+g(\d+))?.*?alg=(\S+)"
)
CONCAT_RE = re.compile(r"cmd#(\d+)\s+(\S+)\s+(\d+)\s*->\s*(\d+)\s*@\s*(\d+)x(\d+)\s+branches=(\d+)")
SUMMARY_RE = re.compile(r"summed_gpu_ms=([0-9.]+)")
E2E_RE = re.compile(r"avg_infer=([0-9.]+)\s+ms")
BREAKDOWN_RE = re.compile(r"fence=([0-9.]+)\s+ms\s+readback=([0-9.]+)")


def parse(path: str) -> dict:
    events = []
    summed_gpu = None
    e2e = None
    fence = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            hit = SUMMARY_RE.search(line)
            if hit:
                summed_gpu = float(hit.group(1))
            hit = E2E_RE.search(line)
            if hit:
                e2e = float(hit.group(1))
            hit = BREAKDOWN_RE.search(line)
            if hit:
                fence = float(hit.group(1))
            hit = EVENT_RE.match(line)
            if not hit:
                continue
            ms = float(hit.group(2))
            rest = hit.group(3)
            conv = CONV_RE.search(rest)
            if conv:
                (_cmd, kind, ic, ih, iw, oc, oh, ow, k, s, g, alg) = conv.groups()
                events.append(
                    {
                        "kind": kind,
                        "ms": ms,
                        "ic": int(ic), "ih": int(ih), "iw": int(iw),
                        "oc": int(oc), "oh": int(oh), "ow": int(ow),
                        "k": int(k), "s": int(s), "g": int(g or 1),
                        "alg": alg,
                    }
                )
                continue
            cat = CONCAT_RE.search(rest)
            if cat:
                (_cmd, kind, cin, cout, h, w, branches) = cat.groups()
                events.append(
                    {
                        "kind": kind,
                        "ms": ms,
                        "ic": int(cin), "ih": int(h), "iw": int(w),
                        "oc": int(cout), "oh": int(h), "ow": int(w),
                        "k": 1, "s": 1, "g": 1,
                        "alg": "concat_conv1x1",
                    }
                )
                continue
            events.append({"kind": rest.split()[1] if "cmd#" in rest else rest.strip(),
                           "ms": ms, "alg": "opaque"})
    return {"events": events, "summed_gpu": summed_gpu, "e2e": e2e, "fence": fence}


def conv_metrics(ev: dict) -> tuple[float, float]:
    """Return (min traffic bytes, macs) for a conv-shaped event (fp32 storage)."""
    ic, ih, iw = ev["ic"], ev["ih"], ev["iw"]
    oc, oh, ow = ev["oc"], ev["oh"], ev["ow"]
    k, g = ev["k"], ev["g"]
    in_elems = ic * ih * iw
    out_elems = oc * oh * ow
    weight_elems = oc * (ic // g) * k * k
    traffic = 4.0 * (in_elems + out_elems + weight_elems)
    macs = 2.0 * out_elems * (ic // g) * k * k
    return traffic, macs


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    data = parse(sys.argv[1])
    events = data["events"]
    if not events:
        print("no profile events parsed")
        return 2

    total_ms = sum(ev["ms"] for ev in events)
    print(f"events={len(events)} gpu_sum={total_ms:.3f} ms  e2e={data['e2e']} ms  fence={data['fence']} ms")
    if data["fence"] and data["summed_gpu"]:
        print(f"bubble (fence-gpu) = {data['fence'] - data['summed_gpu']:.3f} ms "
              f"over {len(events)} events = {(data['fence'] - data['summed_gpu']) * 1000.0 / len(events):.1f} us/event")

    rows = []
    for ev in events:
        if "ic" not in ev:
            continue
        traffic, macs = conv_metrics(ev)
        ms = ev["ms"]
        rows.append({
            **ev,
            "traffic": traffic,
            "macs": macs,
            "traffic_mb": traffic / 1e6,
            "gbs": traffic / (ms * 1e-3) / 1e9,
            "tflops": macs / (ms * 1e-3) / 1e12,
            "ai": macs / traffic if traffic else 0.0,
        })

    if not rows:
        print("no conv-shaped events")
        return 0

    best_gbs = max(r["gbs"] for r in rows)
    print(f"best achieved bandwidth among events: {best_gbs:.1f} GB/s")
    print()

    hdr = (f"{'ms':>7} {'GB/s':>7} {'TFLOPS':>7} {'AI':>6}  {'kind':<28} {'shape':<34} {'alg'}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["ms"], reverse=True)[:30]:
        shape = f"{r['ic']}x{r['ih']}x{r['iw']}->{r['oc']}x{r['oh']}x{r['ow']} k{r['k']}s{r['s']}"
        print(f"{r['ms']:7.3f} {r['gbs']:7.1f} {r['tflops']:7.2f} {r['ai']:6.1f}  "
              f"{r['kind']:<28} {shape:<34} {r['alg']}")

    print()
    per_alg = defaultdict(lambda: {"ms": 0.0, "traffic": 0.0, "macs": 0.0, "n": 0})
    for r in rows:
        slot = per_alg[r["alg"]]
        slot["ms"] += r["ms"]
        slot["traffic"] += r["traffic"]
        slot["macs"] += r["macs"]
        slot["n"] += 1
    print(f"{'total ms':>9} {'n':>3} {'GB/s':>7} {'TFLOPS':>7}  alg")
    print("-" * 60)
    for alg, slot in sorted(per_alg.items(), key=lambda kv: kv[1]["ms"], reverse=True):
        gbs = slot["traffic"] / (slot["ms"] * 1e-3) / 1e9
        tflops = slot["macs"] / (slot["ms"] * 1e-3) / 1e12
        print(f"{slot['ms']:9.3f} {slot['n']:3d} {gbs:7.1f} {tflops:7.2f}  {alg}")

    n_low = sum(1 for r in rows if r["gbs"] < best_gbs * 0.25 and r["ms"] > 0.02)
    low_ms = sum(r["ms"] for r in rows if r["gbs"] < best_gbs * 0.25 and r["ms"] > 0.02)
    print(f"\nkernels under 25% of best bandwidth (>20us): {n_low}, total {low_ms:.3f} ms "
          f"({100.0 * low_ms / total_ms:.0f}% of gpu time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
