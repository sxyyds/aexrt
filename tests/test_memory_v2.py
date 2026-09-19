import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aexrt.memory_v2 import (  # noqa: E402
    LAYOUT_BLOCKED_NCHW8,
    PRECISION_FLOAT16,
    VALUE_ALIAS,
    VALUE_ARENA,
    build_physical_value_dag,
    build_page_colored_arena,
    encode_memory_plan_v2,
    inspect_memory_plan_v2,
)


def _value(value_id, elements, flags=0):
    return {"id": value_id, "flags": flags, "elements": elements}


def _command(kind, output, *inputs):
    return {"kind": kind, "output": output, "inputs": list(inputs)}


def test_page_coloring_separates_every_dispatch_input_from_output():
    replay = {
        "values": [
            _value(0, 64, flags=1),
            _value(1, 64),
            _value(2, 64),
            _value(3, 64),
        ],
        "commands": [
            _command("CONV_SILU", 1, 0),
            _command("CONV_SILU", 2, 1),
            _command("BINARY", 3, 0, 2),
        ],
        "output_value": 3,
    }
    plan = build_page_colored_arena(replay)
    pages = {value_id: record["page_id"] for value_id, record in plan["by_value"].items()}

    for command in replay["commands"]:
        assert pages[command["output"]] not in {pages[value_id] for value_id in command["inputs"]}
    assert len(plan["pages"]) == 2


def test_page_allocator_reuses_non_overlapping_lifetimes():
    replay = {
        "values": [
            _value(0, 64, flags=1),
            _value(1, 64),
            _value(2, 64),
            _value(3, 64),
        ],
        "commands": [
            _command("CONV_SILU", 1, 0),
            _command("CONV_SILU", 2, 1),
            _command("CONV_SILU", 3, 2),
        ],
        "output_value": 3,
    }
    plan = build_page_colored_arena(replay)
    first = plan["by_value"][0]
    last = plan["by_value"][2]

    assert first["page_id"] == last["page_id"]
    assert first["offset"] == last["offset"]
    assert plan["total_nbytes"] == 2 * 256


def test_exact_size_reuse_does_not_place_smaller_values_in_larger_slots():
    replay = {
        "values": [
            _value(0, 128, flags=1),
            _value(1, 128),
            _value(2, 64),
            _value(3, 64),
        ],
        "commands": [
            _command("CONV_SILU", 1, 0),
            _command("CONV_SILU", 2, 1),
            _command("CONV_SILU", 3, 2),
        ],
        "output_value": 3,
    }

    best_fit = build_page_colored_arena(replay)
    exact = build_page_colored_arena(replay, reuse_exact_size=True)

    assert best_fit["by_value"][0]["offset"] == best_fit["by_value"][2]["offset"]
    assert exact["by_value"][0]["offset"] != exact["by_value"][2]["offset"]
    assert exact["total_nbytes"] > best_fit["total_nbytes"]


def test_exact_size_reuse_still_reuses_equal_sized_values():
    replay = {
        "values": [
            _value(0, 64, flags=1),
            _value(1, 64),
            _value(2, 64),
            _value(3, 64),
        ],
        "commands": [
            _command("CONV_SILU", 1, 0),
            _command("CONV_SILU", 2, 1),
            _command("CONV_SILU", 3, 2),
        ],
        "output_value": 3,
    }

    exact = build_page_colored_arena(replay, reuse_exact_size=True)

    assert exact["by_value"][0]["offset"] == exact["by_value"][2]["offset"]


def test_alias_values_inherit_storage_and_extend_source_lifetime():
    replay = {
        "values": [
            _value(0, 128, flags=1),
            _value(1, 128),
            _value(2, 64),
            _value(3, 128),
            _value(4, 64),
        ],
        "commands": [
            _command("CONV_SILU", 1, 0),
            _command("VIEW", 2, 1),
            _command("CONV_SILU", 3, 0),
            _command("CONV_SILU", 4, 2),
        ],
        "output_value": 4,
    }

    plan = build_page_colored_arena(replay)

    assert 2 in plan["by_value"]
    assert replay["values"][2]["flags"] & VALUE_ALIAS
    assert replay["values"][2]["flags"] & VALUE_ARENA == 0
    source = plan["by_value"][1]
    alias = plan["by_value"][2]
    assert alias["flags"] & VALUE_ALIAS
    assert alias["flags"] & VALUE_ARENA
    assert alias["canonical_value_id"] == 1
    assert alias["is_alias"] == 1
    assert (alias["page_id"], alias["offset"], alias["storage_dtype"]) == (
        source["page_id"],
        source["offset"],
        source["storage_dtype"],
    )
    assert alias["nbytes"] == 64 * 4
    later_output = plan["by_value"][3]
    assert source["end"] == 4
    assert (source["page_id"], source["offset"]) != (
        later_output["page_id"],
        later_output["offset"],
    )


def test_nested_alias_bindings_accumulate_typed_view_offsets():
    replay = {
        "values": [
            _value(0, 512, flags=1),
            _value(1, 512),
            _value(2, 256),
            _value(3, 128),
            _value(4, 128),
        ],
        "commands": [
            _command("CONV_SILU", 1, 0),
            {
                **_command("VIEW", 2, 1),
                "params": [128, 256],
            },
            {
                **_command("ALIAS", 3, 2),
                "params": [128, 128],
            },
            _command("CONV_SILU", 4, 3),
        ],
        "output_value": 4,
    }
    plan = build_page_colored_arena(
        replay,
        storage={1: (PRECISION_FLOAT16, 1)},
    )

    source = plan["bindings_by_value"][1]
    first = plan["bindings_by_value"][2]
    second = plan["bindings_by_value"][3]
    assert first["canonical_value_id"] == second["canonical_value_id"] == 1
    assert first["view_offset_elements"] == 128
    assert second["view_offset_elements"] == 256
    assert first["offset"] == source["offset"] + 128 * 2
    assert second["offset"] == source["offset"] + 256 * 2
    assert first["storage_dtype"] == second["storage_dtype"] == PRECISION_FLOAT16
    inspected = inspect_memory_plan_v2(
        encode_memory_plan_v2(plan),
        value_elements={
            value["id"]: value["elements"] for value in replay["values"]
        },
    )
    assert inspected["by_value"][2]["flags"] & VALUE_ALIAS
    assert inspected["by_value"][3]["flags"] & VALUE_ALIAS
    assert inspected["by_value"][3]["offset"] == second["offset"]


def test_physical_dag_lowers_alias_only_records_to_zero_dispatch():
    replay = {
        "values": [
            _value(0, 512, flags=1),
            _value(1, 512),
            _value(2, 256),
            _value(3, 256),
            _value(4, 256),
        ],
        "commands": [
            _command("CONV_SILU", 1, 0),
            {**_command("VIEW", 2, 1), "params": [128, 256]},
            {**_command("ALIAS", 3, 2), "params": [0, 256]},
            _command("CONV_SILU", 4, 3),
        ],
        "output_value": 4,
    }
    physical = [
        {"logical_indices": (index,), "execution_index": index}
        for index in range(4)
    ]

    dag = build_physical_value_dag(replay, physical)

    assert dag["material_dispatch_count"] == 2
    assert dag["zero_dispatch_indices"] == (1, 2)
    assert [dispatch["stage_index"] for dispatch in dag["dispatches"]] == [0, -1, -1, 1]
    assert dag["dispatches"][1]["external_inputs"] == frozenset()
    assert dag["dispatches"][2]["external_inputs"] == frozenset()
    assert dag["consumer_dispatches"][1] == (3,)
    assert dag["producer_stage_by_value"][1] == 0
    assert dag["consumer_stages"][1] == (1,)
    assert dag["edges"] == ((0, 3),)

    arena = build_page_colored_arena(
        replay, physical_dispatch_plan=physical
    )
    assert arena["by_value"][1]["start"] == 1
    assert arena["by_value"][1]["end"] == 2


def test_dead_values_do_not_allocate_arena_storage():
    replay = {
        "values": [_value(0, 64, flags=1), _value(1, 64), _value(2, 1024)],
        "commands": [_command("CONV_SILU", 1, 0)],
        "output_value": 1,
    }

    plan = build_page_colored_arena(replay)

    assert 2 not in plan["by_value"]


def test_fp16_blocked_storage_halves_value_payload():
    replay = {
        "values": [_value(0, 128, flags=1), _value(1, 128)],
        "commands": [_command("CONV_SILU", 1, 0)],
        "output_value": 1,
    }
    storage = {
        0: (PRECISION_FLOAT16, LAYOUT_BLOCKED_NCHW8),
        1: (PRECISION_FLOAT16, LAYOUT_BLOCKED_NCHW8),
    }
    plan = build_page_colored_arena(replay, storage=storage)

    assert all(record["nbytes"] == 256 for record in plan["records"])
    assert all(page["storage_dtype"] == PRECISION_FLOAT16 for page in plan["pages"])
    assert all(page["storage_layout"] == LAYOUT_BLOCKED_NCHW8 for page in plan["pages"])


def test_memory_v2_binary_roundtrip_preserves_pages_and_values():
    replay = {
        "values": [_value(0, 64, flags=1), _value(1, 64), _value(2, 64)],
        "commands": [
            _command("CONV_SILU", 1, 0),
            _command("CONV_SILU", 2, 1),
        ],
        "output_value": 2,
    }
    plan = build_page_colored_arena(replay)
    payload = encode_memory_plan_v2(plan)
    inspected = inspect_memory_plan_v2(
        payload, value_elements={value["id"]: value["elements"] for value in replay["values"]}
    )

    assert inspected["version"] == 2
    assert inspected["total_nbytes"] == plan["total_nbytes"]
    assert inspected["pages"] == plan["pages"]
    assert [record["page_id"] for record in inspected["records"]] == [
        plan["by_value"][value_id]["page_id"] for value_id in sorted(plan["by_value"])
    ]


def test_memory_v2_binary_rejects_trailing_bytes():
    replay = {
        "values": [_value(0, 64, flags=1), _value(1, 64)],
        "commands": [_command("CONV_SILU", 1, 0)],
        "output_value": 1,
    }
    payload = encode_memory_plan_v2(build_page_colored_arena(replay))

    try:
        inspect_memory_plan_v2(payload + b"\0")
    except ValueError as error:
        assert "header" in str(error)
    else:
        raise AssertionError("trailing V2 memory-plan bytes must be rejected")


def test_page_coloring_accounts_for_inputs_hidden_by_physical_fusion():
    replay = {
        "values": [
            _value(0, 64, flags=1),
            _value(1, 64),
            _value(2, 64),
            _value(3, 64),
        ],
        "commands": [
            _command("CONV_SILU", 1, 0),
            _command("CONV_SILU", 2, 1),
            _command("BINARY", 3, 0, 2),
        ],
        "output_value": 3,
    }
    physical = [
        {"logical_indices": (0,)},
        {"logical_indices": (1, 2)},
    ]
    plan = build_page_colored_arena(replay, physical_dispatch_plan=physical)
    pages = {value_id: record["page_id"] for value_id, record in plan["by_value"].items()}

    assert pages[3] != pages[1]


def test_physical_timeline_keeps_delayed_inputs_alive():
    replay = {
        "values": [
            _value(0, 64, flags=1),
            _value(1, 64),
            _value(2, 64),
            _value(3, 64),
        ],
        "commands": [
            _command("CONV_SILU", 1, 0),
            _command("CONV_SILU", 2, 1),
            _command("CONV_SILU", 3, 0),
        ],
        "output_value": 2,
    }
    physical = [
        {"logical_indices": (0,)},
        {"logical_indices": (2,)},
        {"logical_indices": (1,)},
    ]

    plan = build_page_colored_arena(replay, physical_dispatch_plan=physical)
    delayed_input = plan["by_value"][1]
    intervening_output = plan["by_value"][3]

    assert delayed_input["page_id"] == intervening_output["page_id"]
    assert delayed_input["end"] > intervening_output["start"]
    assert delayed_input["offset"] != intervening_output["offset"]


def test_physical_timeline_rejects_forward_dependencies():
    replay = {
        "values": [_value(0, 64, flags=1), _value(1, 64), _value(2, 64)],
        "commands": [
            _command("CONV_SILU", 1, 0),
            _command("CONV_SILU", 2, 1),
        ],
        "output_value": 2,
    }
    physical = [
        {"logical_indices": (1,)},
        {"logical_indices": (0,)},
    ]

    try:
        build_page_colored_arena(replay, physical_dispatch_plan=physical)
    except ValueError as error:
        assert "topologically" in str(error)
    else:
        raise AssertionError("forward physical dependencies must be rejected")
