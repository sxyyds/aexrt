import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
from aexrt import Graph, InferenceSession, TensorSpec

rng = np.random.default_rng(42)
B, IN, H, OUT = 4, 16, 32, 8

g = Graph("two_layer_mlp")
g.input("x", TensorSpec((B, IN), "float32"))
g.const("w1", (rng.standard_normal((IN, H)) / np.sqrt(IN)).astype("float32"))
g.const("b1", np.zeros((H,), dtype="float32"))
g.const("w2", (rng.standard_normal((H, OUT)) / np.sqrt(H)).astype("float32"))
g.const("b2", np.zeros((OUT,), dtype="float32"))
g.matmul("h0", "x", "w1")
g.add("h1", "h0", "b1")
g.gelu("h2", "h1")
g.matmul("o0", "h2", "w2")
g.add("logits", "o0", "b2")
g.output("logits")

x = rng.standard_normal((B, IN)).astype("float32")
session = InferenceSession(g, backend="torch", device="auto", optimize=True)
y = session.run({"x": x})["logits"]

print("backend:", session.info())
print("optimized nodes:", [n.op for n in session.graph.nodes])
print("output shape:", y.shape)
print("output sample:", y[0, :4])
