import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
from aexrt import Graph, InferenceSession, TensorSpec


def build_mlp(batch=256, width=2048, hidden=4096):
    rng = np.random.default_rng(123)
    g = Graph("benchmark_mlp")
    g.input("x", TensorSpec((batch, width), "float32"))
    g.const("w1", (rng.standard_normal((width, hidden)) / np.sqrt(width)).astype("float32"))
    g.const("b1", np.zeros((hidden,), dtype="float32"))
    g.const("w2", (rng.standard_normal((hidden, width)) / np.sqrt(hidden)).astype("float32"))
    g.const("b2", np.zeros((width,), dtype="float32"))
    g.matmul("a", "x", "w1")
    g.add("b", "a", "b1")
    g.gelu("c", "b")
    g.matmul("d", "c", "w2")
    g.add("y", "d", "b2")
    g.output("y")
    return g


def bench(session, x, warmup=10, iters=50):
    for _ in range(warmup):
        session.run({"x": x})
    t0 = time.perf_counter()
    for _ in range(iters):
        session.run({"x": x})
    return (time.perf_counter() - t0) * 1000.0 / iters


if __name__ == "__main__":
    batch, width, hidden = 128, 1024, 2048
    x = np.random.default_rng(7).standard_normal((batch, width)).astype("float32")
    g = build_mlp(batch, width, hidden)

    cpu = InferenceSession(g, backend="numpy", optimize=True)
    gpu = InferenceSession(g, backend="torch", device="auto", optimize=True)

    print("CPU:", cpu.info())
    print("GPU:", gpu.info())
    print("nodes:", [n.op for n in gpu.graph.nodes])

    cpu_ms = bench(cpu, x, warmup=1, iters=3)
    gpu_ms = bench(gpu, x, warmup=10, iters=30)
    print(f"numpy cpu avg: {cpu_ms:.3f} ms")
    print(f"torch gpu/auto avg: {gpu_ms:.3f} ms")
    if gpu_ms > 0:
        print(f"speedup: {cpu_ms / gpu_ms:.2f}x")
