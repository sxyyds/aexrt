import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aexrt import Graph, TensorSpec


g = Graph("add_relu_abi")
g.input("a", TensorSpec((2, 3), "float32"))
g.input("b", TensorSpec((2, 3), "float32"))
g.add("sum", "a", "b")
g.relu("y", "sum")
g.output("y")

out = os.path.abspath(os.path.join(os.path.dirname(__file__), "add_relu.aexrt.json"))
g.save_aexrt(out)
print(out)
