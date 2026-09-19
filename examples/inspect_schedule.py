import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

from aexrt import Graph, InferenceSession, TensorSpec, compile_graph_schedule


g = Graph("inspect_schedule")
g.input("x", TensorSpec((8, 16), "float32"))
g.const("w", np.ones((16, 32), dtype="float32"))
g.matmul("m", "x", "w")
g.relu("a", "m")
g.gelu("y", "a")
g.output("y")

session = InferenceSession(g, backend="numpy", optimize=False)
schedule = compile_graph_schedule(session.graph, session.execution_plan())

print("signature:", schedule.signature)
print("algorithm_tags:", schedule.algorithm_tags)

print("arena:")
print("  alignment:", schedule.arena.alignment)
print("  total_nbytes:", schedule.arena.total_nbytes)
for slot in schedule.arena.slots:
    print(f"  alloc={slot.allocation_id} offset={slot.offset} nbytes={slot.nbytes} values={slot.values}")

print("fusion_groups:")
for group in schedule.fusion_groups:
    print(f"  #{group.group_id} kind={group.kind} nodes={group.nodes} outputs={group.outputs}")

print("kernels:")
for kernel in schedule.kernels:
    tile = kernel.tile
    tile_text = None if tile is None else (
        f"{tile.strategy} tile=({tile.tile_m},{tile.tile_n},{tile.tile_k}) "
        f"waves={tile.waves_per_group} vec={tile.vector_width}"
    )
    print(
        f"  #{kernel.kernel_id} op={kernel.op} category={kernel.category} "
        f"work={kernel.estimated_work_items} fusion={kernel.fusion_group} tile={tile_text}"
    )
