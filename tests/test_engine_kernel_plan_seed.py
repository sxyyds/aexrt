import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aexrt import engine as engine_module  # noqa: E402


def _replay():
    return {
        "values": [
            {"id": 0, "flags": engine_module.VALUE_INPUT, "elements": 102400},
            {"id": 1, "flags": engine_module.VALUE_CONSTANT, "elements": 16384},
            {"id": 2, "flags": engine_module.VALUE_CONSTANT, "elements": 128},
            {"id": 3, "flags": 0, "elements": 51200},
        ],
        "commands": [
            {
                "kind": "CONV_SILU",
                "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
                "output": 3,
                "inputs": [0, 1, 2],
                "params": [
                    1, 128, 20, 20, 128, 20, 20,
                    1, 1, 1, 1, 0, 0, 1, 1, 1,
                ],
            }
        ],
        "input_value": 0,
        "output_value": 3,
    }


def test_kernel_plan_seed_freezes_promoted_choice_and_flags():
    replay = _replay()
    plan = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    assert plan[0]["planned_kernel"] == 45

    seed = [(0, 2, 1, 2, engine_module.NO_VALUE, 1)]
    engine_module._apply_kernel_plan_seed(replay, plan, seed)

    assert plan == [
        {
            "command_index": 0,
            "kind_id": 2,
            "planned_kernel": 1,
            "precision": 2,
            "packed_value": engine_module.NO_VALUE,
            "flags": engine_module.PLAN_FLAG_AUTHORITATIVE,
        }
    ]
    physical = engine_module._build_physical_dispatch_plan(replay, plan, [])
    assert physical[0]["kernel"] == 1
    assert physical[0]["flags"] == engine_module.PLAN_FLAG_AUTHORITATIVE


@pytest.mark.parametrize(
    "seed",
    [
        [],
        [(1, 2, 1, 2, engine_module.NO_VALUE, 1)],
        [(0, 99, 1, 2, engine_module.NO_VALUE, 1)],
        [(0, 2, 1, 2, 999, 1)],
        [(0, 2, 1, 2, engine_module.NO_VALUE, 0)],
    ],
)
def test_kernel_plan_seed_rejects_incompatible_records(seed):
    replay = _replay()
    plan = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    with pytest.raises(ValueError, match="kernel-plan seed"):
        engine_module._apply_kernel_plan_seed(replay, plan, seed)
