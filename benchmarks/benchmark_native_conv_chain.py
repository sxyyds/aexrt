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
    rng = np.random.default_rng(20260708)
    x = (rng.standard_normal((1, 3, 320, 320)) * 0.25).astype("float32")
    w0 = (rng.standard_normal((16, 3, 3, 3)) * 0.05).astype("float32")
    b0 = np.zeros((16,), dtype="float32")
    w1 = (rng.standard_normal((16, 16, 3, 3)) * 0.05).astype("float32")
    b1 = np.zeros((16,), dtype="float32")
    desc0 = {
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
    desc1 = dict(desc0)
    desc1["in_channels"] = 16

    w0b = device.upload(w0, label="chain_w0")
    b0b = device.upload(b0, label="chain_b0")
    w1b = device.upload(w1, label="chain_w1")
    b1b = device.upload(b1, label="chain_b1")
    h0 = device.allocate_uav(1 * 16 * 320 * 320 * 4, dtype="float32", shape=(1, 16, 320, 320), label="chain_h0")
    y = device.allocate_uav(1 * 16 * 320 * 320 * 4, dtype="float32", shape=(1, 16, 320, 320), label="chain_y")

    first = device.prepare_conv2d_silu_upload_float32_dispatch(w0b, b0b, h0, desc0, ring_size=2)
    second = device.prepare_conv2d_silu_float32_dispatch(h0, w1b, b1b, y, desc1)
    chain = device.prepare_conv2d_silu_chain_upload_float32_dispatch([w0b, w1b], [b0b, b1b], y, [desc0, desc1], ring_size=2)

    old_ms = bench(lambda: (device.execute_conv2d_silu_upload_float32_dispatch(first, x), device.execute_conv2d_silu_float32_dispatch(second)), 30)
    chain_ms = bench(lambda: device.execute_conv2d_silu_chain_upload_float32_dispatch(chain, x), 30)

    print(f"two dispatches/two fences: {old_ms:.4f} ms")
    print(f"chain command list/one fence: {chain_ms:.4f} ms")
    if chain_ms > 0:
        print(f"chain speedup: {old_ms / chain_ms:.2f}x")
