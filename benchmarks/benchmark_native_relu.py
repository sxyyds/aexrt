import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

from aexrt import Graph, InferenceSession, NativeD3D12Device, TensorSpec


def bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / iters


if __name__ == "__main__":
    info = NativeD3D12Device.probe()
    print("native_d3d12:", info)
    if not info.available:
        raise SystemExit("native D3D12 is unavailable")

    shape = (1 << 20,)
    x = np.random.default_rng(123).standard_normal(shape).astype("float32")

    g = Graph("native_relu_bench")
    g.input("x", TensorSpec(shape, "float32"))
    g.relu("y", "x")
    g.output("y")

    native = InferenceSession(g, backend="native_d3d12", optimize=False)
    print("note: native timing includes upload + dispatch + fence wait + readback")
    native_ms = bench(lambda: native.run({"x": x}), warmup=5, iters=50)

    device = NativeD3D12Device()
    x_buffer = device.upload(x, label="persistent_relu_x")
    y_buffer = device.allocate_uav(x.nbytes, dtype="float32", shape=x.shape, label="persistent_relu_y")
    persistent_ms = bench(
        lambda: device.dispatch_relu_float32_into(x_buffer, y_buffer, x.size),
        warmup=10,
        iters=200,
    )
    prepared = device.prepare_relu_float32_dispatch(x_buffer, y_buffer, x.size)
    prepared_ms = bench(
        lambda: device.execute_relu_float32_dispatch(prepared),
        warmup=20,
        iters=500,
    )
    numpy_ms = bench(lambda: np.maximum(x, 0), warmup=5, iters=50)

    y_native = native.run({"x": x})["y"]
    y_persistent = device.download(y_buffer)
    max_diff = float(np.max(np.abs(y_native - np.maximum(x, 0))))
    max_diff_prepared = float(np.max(np.abs(y_persistent - np.maximum(x, 0))))

    print(f"elements: {x.size}")
    print(f"native d3d12 relu session e2e avg: {native_ms:.4f} ms")
    print(f"native d3d12 relu persistent dispatch avg: {persistent_ms:.4f} ms")
    print(f"native d3d12 relu prepared dispatch avg: {prepared_ms:.4f} ms")
    print(f"numpy relu avg: {numpy_ms:.4f} ms")
    print(f"max diff: {max_diff}")
    print(f"max diff prepared/persistent: {max_diff_prepared}")
