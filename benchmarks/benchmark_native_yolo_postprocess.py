import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

from aexrt import NativeD3D12Device


def bench(fn, iters):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / iters


if __name__ == "__main__":
    info = NativeD3D12Device.probe()
    print(info)
    if not info.available:
        raise SystemExit("native D3D12 is unavailable")

    device = NativeD3D12Device()
    rng = np.random.default_rng(20260709)
    anchors = 8400
    classes = 80
    channels = 4 + classes
    max_candidates = 512
    max_detections = 100

    y = np.zeros((channels, anchors), dtype="float32")
    y[:4] = rng.uniform(0, 320, size=(4, anchors)).astype("float32")
    y[2:4] = rng.uniform(4, 80, size=(2, anchors)).astype("float32")
    hot = rng.choice(anchors, size=max_candidates, replace=False)
    cls = rng.integers(0, classes, size=max_candidates)
    scores = rng.uniform(0.25, 0.98, size=max_candidates).astype("float32")
    y[4 + cls, hot] = scores

    yb = device.upload(y, label="bench_yolo_head")
    candidates, candidate_counter, keep, detections, counter = device.allocate_yolo_nms_buffers(
        max_candidates,
        max_detections,
        label="bench_yolo_nms",
    )

    gpu_ms = bench(
        lambda: device.dispatch_yolo_decode_nms_float32(
            yb,
            candidates,
            candidate_counter,
            keep,
            detections,
            counter,
            anchors=anchors,
            channels=channels,
            classes=classes,
            max_candidates=max_candidates,
            max_detections=max_detections,
            conf_threshold=0.25,
            iou_threshold=0.45,
        ),
        40,
    )
    tiny_readback_ms = bench(lambda: device.download_yolo_topk(detections, counter), 40)
    print(f"gpu decode/filter/nms/topK: {gpu_ms:.4f} ms")
    print(f"tiny readback count/topK: {tiny_readback_ms:.4f} ms")
