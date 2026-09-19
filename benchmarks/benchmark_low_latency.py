import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
from aexrt import Graph, InferenceSession, TensorSpec


def build_tiny(batch=1, width=512, hidden=1024, layers=4):
    rng = np.random.default_rng(2026)
    g = Graph("low_latency_mlp")
    g.input("x", TensorSpec((batch, width), "float32"))
    last = "x"
    for i in range(layers):
        in_dim = width if i == 0 else hidden
        out_dim = width if i == layers - 1 else hidden
        g.const(f"w{i}", (rng.standard_normal((in_dim, out_dim)) / np.sqrt(in_dim)).astype("float32"))
        g.const(f"b{i}", np.zeros((out_dim,), dtype="float32"))
        g.matmul(f"m{i}", last, f"w{i}")
        g.add(f"a{i}", f"m{i}", f"b{i}")
        if i != layers - 1:
            g.gelu(f"h{i}", f"a{i}")
            last = f"h{i}"
        else:
            last = f"a{i}"
    g.output(last)
    return g


def bench(session, x, iters=200):
    for _ in range(20):
        session.run({"x": x})
    t0 = time.perf_counter()
    for _ in range(iters):
        session.run({"x": x})
    return (time.perf_counter() - t0) * 1000.0 / iters


if __name__ == "__main__":
    x = np.random.default_rng(9).standard_normal((1, 512)).astype("float32")
    g = build_tiny()
    eager = InferenceSession(g, backend="torch", device="auto", output_numpy=False, cuda_graph=False)
    captured = InferenceSession(g, backend="torch", device="auto", output_numpy=False, cuda_graph=True)
    print("eager:", eager.info())
    print("captured:", captured.info())
    e = bench(eager, x)
    c = bench(captured, x)
    print(f"torch eager avg: {e:.4f} ms")
    print(f"cuda graph avg: {c:.4f} ms")
    if c > 0:
        print(f"latency reduction: {e / c:.2f}x")
