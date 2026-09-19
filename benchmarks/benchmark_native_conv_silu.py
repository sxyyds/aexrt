import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

from aexrt import NativeD3D12Device


def bench(fn, iters):
    for _ in range(10):
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
    rng = np.random.default_rng(20260707)
    x = (rng.standard_normal((1, 3, 320, 320)) * 0.25).astype("float32")
    w = (rng.standard_normal((16, 3, 3, 3)) * 0.05).astype("float32")
    b = np.zeros((16,), dtype="float32")
    y_shape = (1, 16, 320, 320)
    desc = {
        "batch": 1,
        "in_channels": 3,
        "in_h": 320,
        "in_w": 320,
        "out_channels": 16,
        "out_h": 320,
        "out_w": 320,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_top": 1,
        "pad_left": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "groups": 1,
    }

    xb = device.allocate(x.nbytes, dtype="float32", shape=x.shape, label="bench_x")
    wb = device.upload(w, label="bench_w")
    bb = device.upload(b, label="bench_b")
    yb = device.allocate_uav(int(np.prod(y_shape)) * 4, dtype="float32", shape=y_shape, label="bench_y")
    device.upload_into(xb, x)
    dispatch = device.prepare_conv2d_silu_float32_dispatch(xb, wb, bb, yb, desc)
    upload_dispatch = device.prepare_conv2d_silu_upload_float32_dispatch(wb, bb, yb, desc, ring_size=2)

    execute_ms = bench(lambda: device.execute_conv2d_silu_float32_dispatch(dispatch), 80)
    upload_execute_ms = bench(lambda: (device.upload_into(xb, x), device.execute_conv2d_silu_float32_dispatch(dispatch)), 80)
    upload_ring_execute_ms = bench(lambda: device.execute_conv2d_silu_upload_float32_dispatch(upload_dispatch, x), 80)
    full_ms = bench(lambda: (device.upload_into(xb, x), device.execute_conv2d_silu_float32_dispatch(dispatch), device.download(yb)), 20)
    upload_ring_full_ms = bench(lambda: (device.execute_conv2d_silu_upload_float32_dispatch(upload_dispatch, x), device.download(yb)), 20)

    print(f"prepared execute only: {execute_ms:.4f} ms")
    print(f"upload_into + execute: {upload_execute_ms:.4f} ms")
    print(f"upload ring + execute: {upload_ring_execute_ms:.4f} ms")
    print(f"upload_into + execute + download: {full_ms:.4f} ms")
    print(f"upload ring + execute + download: {upload_ring_full_ms:.4f} ms")
