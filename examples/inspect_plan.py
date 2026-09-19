import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

from aexrt import Graph, InferenceSession, TensorSpec


g = Graph("inspect_plan")
g.input("x", TensorSpec((2, 4), "float32"))
g.const("w", np.ones((4, 4), dtype="float32"))
g.matmul("m", "x", "w")
g.gelu("h", "m")
g.relu("y", "h")
g.output("y")

session = InferenceSession(g, backend="auto", optimize=True)
plan = session.execution_plan()

print("backend:", session.info())
print("ops:", plan.node_ops)
print("buffers:")
for buffer in plan.memory.buffers:
    print(
        f"  {buffer.name:>8} kind={buffer.kind:<9} shape={buffer.shape} "
        f"dtype={buffer.dtype} nbytes={buffer.nbytes} alloc={buffer.allocation_id}"
    )

print("allocations:")
for allocation in plan.memory.allocations:
    print(
        f"  #{allocation.allocation_id} kind={allocation.kind:<9} "
        f"nbytes={allocation.nbytes} values={allocation.values}"
    )
