from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from typing import Any


PRECISION_FLOAT32 = 1
PRECISION_FLOAT16 = 2
LAYOUT_LINEAR_NCHW = 1
LAYOUT_BLOCKED_NCHW8 = 2

VALUE_INPUT = 1 << 0
VALUE_CONSTANT = 1 << 1
VALUE_ARENA = 1 << 2
VALUE_ALIAS = 1 << 3

MEMORY_PLAN_VERSION = 2
MEMORY_PLAN_FLAG_AUTHORITATIVE = 1 << 0
MEMORY_PLAN_HEADER_SIZE = 32
MEMORY_PAGE_RECORD_SIZE = 24
MEMORY_VALUE_RECORD_SIZE = 40

PHYSICAL_FUSION_LATTICE_CHAIN_C3 = 9


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def build_physical_value_dag(
    replay: dict[str, Any],
    physical_dispatch_plan: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate and describe value flow on the serialized dispatch timeline.

    Alias commands do not materialize a value, so their inputs and all later
    users are attributed to the canonical arena value.  The command ownership
    table is retained as well: compiler passes that change physical storage
    must distinguish a singleton dispatch from a logical command owned by a
    fusion.
    """
    values = replay["values"]
    commands = replay["commands"]
    value_count = len(values)

    alias_source: dict[int, int] = {}
    for command in commands:
        output = int(command["output"])
        inputs = tuple(int(value_id) for value_id in command["inputs"])
        if output < 0 or output >= value_count or any(
            value_id < 0 or value_id >= value_count for value_id in inputs
        ):
            raise ValueError("replay command references an invalid value")
        if command["kind"] in {"VIEW", "ALIAS"}:
            if not inputs:
                raise ValueError("replay alias has no source")
            alias_source[output] = inputs[0]
            values[output]["flags"] = int(values[output]["flags"]) | VALUE_ALIAS

    def resolve(value_id: int) -> int:
        seen: set[int] = set()
        while value_id in alias_source:
            if value_id in seen:
                raise ValueError("cyclic replay alias")
            seen.add(value_id)
            value_id = alias_source[value_id]
        return value_id

    canonical_values = tuple(resolve(value_id) for value_id in range(value_count))
    command_owner = [-1] * len(commands)
    dispatches: list[dict[str, Any]] = []
    materialized_values: set[int] = set()
    material_stage = 0
    for dispatch_index, dispatch in enumerate(physical_dispatch_plan):
        logical_indices = tuple(int(index) for index in dispatch["logical_indices"])
        if not logical_indices:
            raise ValueError("physical dispatch owns no logical commands")
        execution_index = int(dispatch.get("execution_index", logical_indices[0]))
        if execution_index not in logical_indices:
            raise ValueError("physical dispatch execution command is not owned")

        produced: set[int] = set()
        consumed: set[int] = set()
        material_commands: list[int] = []
        for command_index in logical_indices:
            if (
                command_index < 0
                or command_index >= len(commands)
                or command_owner[command_index] != -1
            ):
                raise ValueError("invalid physical dispatch command ownership")
            command_owner[command_index] = dispatch_index
            command = commands[command_index]
            if command["kind"] in {"VIEW", "ALIAS"}:
                continue
            material_commands.append(command_index)
            produced.add(canonical_values[int(command["output"])])
            consumed.update(
                canonical_values[int(value_id)] for value_id in command["inputs"]
            )
        zero_dispatch = not material_commands
        fusion_kind = int(dispatch.get("fusion_kind", 0))
        if zero_dispatch:
            materialized_outputs: set[int] = set()
        elif fusion_kind in {0, 1}:
            materialized_outputs = set(produced)
        elif fusion_kind == 4:
            materialized_outputs = set()
        elif fusion_kind in {2, 6, 8, PHYSICAL_FUSION_LATTICE_CHAIN_C3}:
            final_output = canonical_values[
                int(commands[logical_indices[-1]]["output"])
            ]
            materialized_outputs = {final_output}
        elif fusion_kind in {3, 7}:
            execution_output = canonical_values[
                int(commands[execution_index]["output"])
            ]
            materialized_outputs = {execution_output}
        else:
            raise ValueError("unsupported physical fusion materialization")
        if not materialized_outputs.issubset(produced):
            raise ValueError("physical fusion output is not produced by its owner")
        materialized_values.update(materialized_outputs)
        dispatches.append(
            {
                "dispatch_index": dispatch_index,
                "stage_index": -1 if zero_dispatch else material_stage,
                "zero_dispatch": zero_dispatch,
                "execution_index": execution_index,
                "logical_indices": logical_indices,
                "material_logical_indices": tuple(material_commands),
                "produced_values": frozenset(produced),
                "materialized_outputs": frozenset(materialized_outputs),
                "internal_outputs": frozenset(produced - materialized_outputs),
                "external_inputs": frozenset(consumed - produced),
            }
        )
        if not zero_dispatch:
            material_stage += 1

    if any(owner == -1 for owner in command_owner):
        raise ValueError("physical dispatch plan does not cover every command")

    producer_by_value = [-1] * value_count
    producer_stage_by_value = [-1] * value_count
    consumer_dispatches: list[set[int]] = [set() for _ in values]
    consumer_stages: list[set[int]] = [set() for _ in values]
    for dispatch in dispatches:
        dispatch_index = int(dispatch["dispatch_index"])
        stage_index = int(dispatch["stage_index"])
        for value_id in dispatch["produced_values"]:
            if producer_by_value[value_id] not in {-1, dispatch_index}:
                raise ValueError("physical dispatch plan has multiple value producers")
            producer_by_value[value_id] = dispatch_index
            producer_stage_by_value[value_id] = stage_index
        for value_id in dispatch["external_inputs"]:
            consumer_dispatches[value_id].add(dispatch_index)
            if stage_index < 0:
                raise ValueError("zero dispatch cannot own a material input")
            consumer_stages[value_id].add(stage_index)

    edges: set[tuple[int, int]] = set()
    for dispatch in dispatches:
        dispatch_index = int(dispatch["dispatch_index"])
        for value_id in dispatch["external_inputs"]:
            flags = int(values[value_id]["flags"])
            producer = producer_by_value[value_id]
            if flags & (VALUE_INPUT | VALUE_CONSTANT):
                continue
            if producer < 0 or producer >= dispatch_index:
                raise ValueError(
                    "physical dispatch plan is not topologically ordered: "
                    f"dispatch={dispatch_index} value={value_id} producer={producer}"
                )
            edges.add((producer, dispatch_index))

    return {
        "alias_source": alias_source,
        "canonical_values": canonical_values,
        "command_owner": tuple(command_owner),
        "dispatches": tuple(dispatches),
        "material_dispatch_count": material_stage,
        "zero_dispatch_indices": tuple(
            int(dispatch["dispatch_index"])
            for dispatch in dispatches
            if dispatch["zero_dispatch"]
        ),
        "producer_by_value": tuple(producer_by_value),
        "producer_stage_by_value": tuple(producer_stage_by_value),
        "materialized_values": frozenset(materialized_values),
        "consumer_dispatches": tuple(
            tuple(sorted(consumers)) for consumers in consumer_dispatches
        ),
        "consumer_stages": tuple(
            tuple(sorted(consumers)) for consumers in consumer_stages
        ),
        "edges": tuple(sorted(edges)),
    }


def build_page_colored_arena(
    replay: dict[str, Any],
    *,
    physical_dispatch_plan: list[dict[str, Any]] | None = None,
    storage: Mapping[int, tuple[int, int]] | None = None,
    alignment: int = 256,
    reuse_slots: bool = True,
    reuse_gap: int = 0,
    reuse_exact_size: bool = False,
    parallel_side_values: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Build a deterministic multi-resource arena from replay dependencies.

    parallel_side_values maps arena value ids to a dual-queue side (0/1) for
    values produced inside a planned parallel window; opposite-side values get
    extra conflict edges so the page coloring never shares a resource page
    between the two queues (page-level barrier plans must stay single-sided)."""
    if alignment < 4 or alignment & (alignment - 1):
        raise ValueError("arena alignment must be a power of two")
    if reuse_gap < 0:
        raise ValueError("arena reuse gap must be non-negative")

    values = replay["values"]
    commands = replay["commands"]
    value_count = len(values)
    producer = [-1] * value_count
    last_use = [-1] * value_count
    alias_source: dict[int, int] = {}
    alias_command: dict[int, dict[str, Any]] = {}

    for command in commands:
        output = int(command["output"])
        if command["kind"] in {"VIEW", "ALIAS"}:
            inputs = tuple(int(value_id) for value_id in command["inputs"])
            if (
                output < 0
                or output >= value_count
                or not inputs
                or inputs[0] < 0
                or inputs[0] >= value_count
                or output in alias_source
            ):
                raise ValueError("invalid replay alias binding")
            alias_source[output] = inputs[0]
            alias_command[output] = command
            values[output]["flags"] = int(values[output]["flags"]) | VALUE_ALIAS

    def resolve(value_id: int) -> int:
        seen: set[int] = set()
        while value_id in alias_source:
            if value_id in seen:
                raise ValueError("cyclic replay alias")
            seen.add(value_id)
            value_id = alias_source[value_id]
        return value_id

    def resolve_alias_view(value_id: int) -> tuple[int, int]:
        """Return canonical value and cumulative element offset for a view."""
        seen: set[int] = set()
        element_offset = 0
        original_value = value_id
        while value_id in alias_source:
            if value_id in seen:
                raise ValueError("cyclic replay alias")
            seen.add(value_id)
            command = alias_command[value_id]
            source = alias_source[value_id]
            params = tuple(int(param) for param in command.get("params", ()))
            view_offset = params[0] if len(params) >= 1 else 0
            view_count = (
                params[1]
                if len(params) >= 2
                else int(values[value_id]["elements"])
            )
            if (
                view_offset < 0
                or view_count <= 0
                or view_count != int(values[value_id]["elements"])
                or view_offset > int(values[source]["elements"])
                or view_count > int(values[source]["elements"]) - view_offset
            ):
                raise ValueError("invalid replay alias view range")
            element_offset += view_offset
            value_id = source
        if (
            element_offset > int(values[value_id]["elements"])
            or int(values[original_value]["elements"])
            > int(values[value_id]["elements"]) - element_offset
        ):
            raise ValueError("replay alias escapes canonical storage")
        return value_id, element_offset

    physical_dag: dict[str, Any] | None = None
    physical_suppressed_values: frozenset[int] = frozenset()
    if physical_dispatch_plan:
        physical_dag = build_physical_value_dag(replay, physical_dispatch_plan)
        alias_source = dict(physical_dag["alias_source"])
        canonical_values = physical_dag["canonical_values"]

        def resolve(value_id: int) -> int:
            return int(canonical_values[value_id])

        producer[:] = physical_dag["producer_stage_by_value"]
        # Kinds 0..8 retain their established value-binding ABI even when a
        # fused shader does not write every logical output.  Kind 9 is the
        # first plan kind whose contract explicitly makes its five internal
        # values register/TGSM-only, so only those bindings are omitted.
        physical_suppressed_values = frozenset(
            value_id
            for dispatch, record in zip(
                physical_dag["dispatches"], physical_dispatch_plan
            )
            if int(record.get("fusion_kind", 0)) == PHYSICAL_FUSION_LATTICE_CHAIN_C3
            for value_id in dispatch["internal_outputs"]
        )
        for value_id, consumers in enumerate(physical_dag["consumer_stages"]):
            if consumers:
                last_use[value_id] = max(int(index) for index in consumers)
        timeline_end = int(physical_dag["material_dispatch_count"])
    else:
        for index, command in enumerate(commands):
            if command["kind"] not in {"VIEW", "ALIAS"}:
                producer[resolve(int(command["output"]))] = index
            for value_id in command["inputs"]:
                source = resolve(int(value_id))
                last_use[source] = max(last_use[source], index)
        timeline_end = len(commands)

    arena_values: set[int] = set()
    for value in values:
        value_id = int(value["id"])
        flags = int(value["flags"])
        if value_id in alias_source or flags & (VALUE_CONSTANT | VALUE_ALIAS):
            continue
        if (
            value_id in physical_suppressed_values
            and not (flags & VALUE_INPUT)
            and producer[value_id] >= 0
        ):
            if last_use[value_id] >= 0:
                raise ValueError(
                    f"unmaterialized physical value {value_id} has an external consumer"
                )
            continue
        if not flags & VALUE_INPUT and producer[value_id] < 0:
            if last_use[value_id] >= 0:
                raise ValueError(f"arena value {value_id} is consumed without a producer")
            continue
        arena_values.add(value_id)
    conflicts = {value_id: set() for value_id in arena_values}
    for command in commands:
        output = resolve(int(command["output"]))
        if output not in arena_values:
            continue
        for raw_input in command["inputs"]:
            input_id = resolve(int(raw_input))
            if input_id in arena_values and input_id != output:
                conflicts[output].add(input_id)
                conflicts[input_id].add(output)

    for dispatch in physical_dag["dispatches"] if physical_dag else ():
        produced = set(dispatch["produced_values"]) & arena_values
        external_inputs = set(dispatch["external_inputs"]) & arena_values
        for output in produced:
            for input_id in external_inputs:
                if output != input_id:
                    conflicts[output].add(input_id)
                    conflicts[input_id].add(output)

    # Dual-queue windows ("hub allocation"): secondary-side values (side=1,
    # inputs and outputs) get conflict edges against EVERY other arena value,
    # pinning the whole secondary value domain onto pages of its own. Slot
    # reuse between a secondary output and any shared input is exactly the
    # race that breaks segmented recording - "thirty spokes share one hub;
    # it is the empty hub that lets the wheel turn" (Dao De Jing 11).
    if parallel_side_values:
        side_resolved: dict[int, int] = {}
        for raw_value_id, side in parallel_side_values.items():
            value_id = int(raw_value_id)
            if not 0 <= value_id < value_count:
                continue
            canonical = resolve(value_id)
            if canonical in conflicts:
                side_resolved[canonical] = int(side)
        hub_values = [v for v, s in side_resolved.items() if s == 1]
        if hub_values:
            hub_set = set(hub_values)
            for value_id in conflicts:
                if value_id in hub_set:
                    continue
                for hub in hub_values:
                    conflicts[value_id].add(hub)
                    conflicts[hub].add(value_id)

    page_by_value: dict[int, int] = {}
    for value_id in sorted(arena_values):
        if int(values[value_id]["flags"]) & VALUE_INPUT:
            page_by_value[value_id] = 0

    uncolored = arena_values - page_by_value.keys()
    while uncolored:
        value_id = max(
            uncolored,
            key=lambda item: (
                len({page_by_value[n] for n in conflicts[item] if n in page_by_value}),
                len(conflicts[item]),
                -item,
            ),
        )
        unavailable = {page_by_value[n] for n in conflicts[value_id] if n in page_by_value}
        page = 0
        while page in unavailable:
            page += 1
        page_by_value[value_id] = page
        uncolored.remove(value_id)

    storage_by_value: dict[int, tuple[int, int]] = {}
    for raw_value_id in storage or ():
        value_id = int(raw_value_id)
        if value_id < 0 or value_id >= value_count or value_id not in arena_values:
            raise ValueError("arena storage override does not name a materialized value")
    for value_id in arena_values:
        dtype, layout = (storage or {}).get(
            value_id, (PRECISION_FLOAT32, LAYOUT_LINEAR_NCHW)
        )
        if dtype not in {PRECISION_FLOAT32, PRECISION_FLOAT16}:
            raise ValueError("unsupported arena storage precision")
        if layout not in {LAYOUT_LINEAR_NCHW, LAYOUT_BLOCKED_NCHW8}:
            raise ValueError("unsupported arena storage layout")
        if layout == LAYOUT_BLOCKED_NCHW8 and dtype != PRECISION_FLOAT16:
            raise ValueError("blocked NCHW8 storage requires FP16")
        storage_by_value[value_id] = (dtype, layout)

    # A resource page has one element type/layout so typed views stay valid.
    remap: dict[tuple[int, int, int], int] = {}
    for value_id in sorted(arena_values):
        color = page_by_value[value_id]
        dtype, layout = storage_by_value[value_id]
        key = (color, dtype, layout)
        if key not in remap:
            remap[key] = len(remap)
        page_by_value[value_id] = remap[key]

    output_value = resolve(int(replay["output_value"]))
    last_use[output_value] = max(last_use[output_value], timeline_end + 1)
    candidates_by_page: dict[int, list[tuple[int, int, int, int]]] = {}
    for value_id in sorted(arena_values):
        flags = int(values[value_id]["flags"])
        start = 0 if flags & VALUE_INPUT else producer[value_id] + 1
        end = max(start, last_use[value_id] + 1)
        dtype, _ = storage_by_value[value_id]
        element_nbytes = 2 if dtype == PRECISION_FLOAT16 else 4
        nbytes = int(values[value_id]["elements"]) * element_nbytes
        candidates_by_page.setdefault(page_by_value[value_id], []).append(
            (start, end, nbytes, value_id)
        )

    records: list[dict[str, int]] = []
    pages: list[dict[str, int]] = []
    for page_id in sorted(candidates_by_page):
        slots: list[dict[str, int]] = []
        cursor = 0
        candidates = sorted(candidates_by_page[page_id], key=lambda item: (item[0], item[3]))
        for start, end, nbytes, value_id in candidates:
            slot_size = _align(nbytes, alignment)
            reusable = (
                [
                    slot
                    for slot in slots
                    if slot["end"] + reuse_gap <= start and slot["size"] >= nbytes
                    and (not reuse_exact_size or slot["nbytes"] == nbytes)
                ]
                if reuse_slots
                else []
            )
            if reusable:
                slot = min(reusable, key=lambda item: (item["size"], item["offset"]))
                offset = slot["offset"]
                slot["end"] = end
            else:
                offset = _align(cursor, alignment)
                cursor = offset + slot_size
                slots.append(
                    {"offset": offset, "size": slot_size, "nbytes": nbytes, "end": end}
                )

            dtype, layout = storage_by_value[value_id]
            records.append(
                {
                    "value_id": value_id,
                    "flags": int(values[value_id]["flags"]) | VALUE_ARENA,
                    "page_id": page_id,
                    "storage_dtype": dtype,
                    "storage_layout": layout,
                    "offset": offset,
                    "nbytes": nbytes,
                    "start": start,
                    "end": end,
                }
            )

        first_value = candidates[0][3]
        dtype, layout = storage_by_value[first_value]
        pages.append(
            {
                "page_id": page_id,
                "flags": 1,
                "storage_dtype": dtype,
                "storage_layout": layout,
                "nbytes": _align(cursor, alignment),
            }
        )

    materialized_records = list(records)
    by_value = {record["value_id"]: record for record in materialized_records}
    bindings_by_value = {
        value_id: {
            **record,
            "canonical_value_id": value_id,
            "view_offset_elements": 0,
            "is_alias": 0,
        }
        for value_id, record in by_value.items()
    }
    for alias_id in sorted(alias_source):
        canonical_id, view_offset_elements = resolve_alias_view(alias_id)
        source = by_value.get(canonical_id)
        if source is None:
            continue
        element_nbytes = 2 if int(source["storage_dtype"]) == PRECISION_FLOAT16 else 4
        offset = int(source["offset"]) + view_offset_elements * element_nbytes
        nbytes = int(values[alias_id]["elements"]) * element_nbytes
        page = pages[int(source["page_id"])]
        if (
            offset % alignment != 0
            or offset > int(page["nbytes"])
            or nbytes > int(page["nbytes"]) - offset
        ):
            raise ValueError("replay alias view is not representable in the arena ABI")
        binding = {
            "value_id": alias_id,
            "flags": int(values[alias_id]["flags"]) | VALUE_ARENA | VALUE_ALIAS,
            "page_id": int(source["page_id"]),
            "storage_dtype": int(source["storage_dtype"]),
            "storage_layout": int(source["storage_layout"]),
            "offset": offset,
            "nbytes": nbytes,
            "start": int(source["start"]),
            "end": int(source["end"]),
            "canonical_value_id": canonical_id,
            "view_offset_elements": view_offset_elements,
            "is_alias": 1,
        }
        records.append(binding)
        by_value[alias_id] = binding
        bindings_by_value[alias_id] = binding
    return {
        "version": 2,
        "flags": 1,
        "alignment": alignment,
        "pages": pages,
        "records": records,
        "by_value": by_value,
        "bindings_by_value": bindings_by_value,
        "materialized_records": materialized_records,
        "canonical_values": tuple(resolve(value_id) for value_id in range(value_count)),
        "total_nbytes": sum(page["nbytes"] for page in pages),
    }


def encode_memory_plan_v2(plan: dict[str, Any]) -> bytes:
    pages = sorted(plan["pages"], key=lambda item: int(item["page_id"]))
    records = sorted(plan["records"], key=lambda item: int(item["value_id"]))
    out = bytearray(
        struct.pack(
            "<4IQII",
            MEMORY_PLAN_VERSION,
            int(plan["alignment"]),
            len(pages),
            len(records),
            int(plan["total_nbytes"]),
            MEMORY_PLAN_FLAG_AUTHORITATIVE,
            0,
        )
    )
    for page in pages:
        out += struct.pack(
            "<4IQ",
            int(page["page_id"]),
            int(page["storage_dtype"]),
            int(page["storage_layout"]),
            int(page["flags"]),
            int(page["nbytes"]),
        )
    for record in records:
        out += struct.pack(
            "<6IQQ",
            int(record["value_id"]),
            int(record["flags"]),
            int(record["page_id"]),
            int(record["storage_dtype"]),
            int(record["storage_layout"]),
            0,
            int(record["offset"]),
            int(record["nbytes"]),
        )
    return bytes(out)


def inspect_memory_plan_v2(
    payload: bytes | bytearray | memoryview,
    *,
    value_elements: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    data = bytes(payload)
    if len(data) < MEMORY_PLAN_HEADER_SIZE:
        raise ValueError("truncated AEXRT V2 memory plan")
    version, alignment, page_count, record_count, total_nbytes, flags, reserved = struct.unpack_from(
        "<4IQII", data, 0
    )
    expected_size = (
        MEMORY_PLAN_HEADER_SIZE
        + page_count * MEMORY_PAGE_RECORD_SIZE
        + record_count * MEMORY_VALUE_RECORD_SIZE
    )
    if (
        version != MEMORY_PLAN_VERSION
        or alignment < 4
        or alignment & (alignment - 1)
        or page_count == 0
        or flags != MEMORY_PLAN_FLAG_AUTHORITATIVE
        or reserved != 0
        or expected_size != len(data)
    ):
        raise ValueError("invalid AEXRT V2 memory-plan header")

    pages: list[dict[str, int]] = []
    page_by_id: dict[int, dict[str, int]] = {}
    cursor = MEMORY_PLAN_HEADER_SIZE
    for expected_page_id in range(page_count):
        page_id, dtype, layout, page_flags, nbytes = struct.unpack_from("<4IQ", data, cursor)
        cursor += MEMORY_PAGE_RECORD_SIZE
        if (
            page_id != expected_page_id
            or dtype not in {PRECISION_FLOAT32, PRECISION_FLOAT16}
            or layout not in {LAYOUT_LINEAR_NCHW, LAYOUT_BLOCKED_NCHW8}
            or (layout == LAYOUT_BLOCKED_NCHW8 and dtype != PRECISION_FLOAT16)
            or page_flags != 1
            or nbytes == 0
            or nbytes % alignment != 0
        ):
            raise ValueError("invalid AEXRT V2 memory page")
        page = {
            "page_id": page_id,
            "storage_dtype": dtype,
            "storage_layout": layout,
            "flags": page_flags,
            "nbytes": nbytes,
        }
        pages.append(page)
        page_by_id[page_id] = page
    if sum(page["nbytes"] for page in pages) != total_nbytes:
        raise ValueError("invalid AEXRT V2 memory-page total")

    records: list[dict[str, int]] = []
    seen_values: set[int] = set()
    for _ in range(record_count):
        value_id, value_flags, page_id, dtype, layout, item_reserved, offset, nbytes = struct.unpack_from(
            "<6IQQ", data, cursor
        )
        cursor += MEMORY_VALUE_RECORD_SIZE
        page = page_by_id.get(page_id)
        expected_elements = (value_elements or {}).get(value_id)
        element_nbytes = 2 if dtype == PRECISION_FLOAT16 else 4
        if (
            value_id in seen_values
            or value_flags & VALUE_ARENA == 0
            or page is None
            or dtype != page["storage_dtype"]
            or layout != page["storage_layout"]
            or item_reserved != 0
            or offset % alignment != 0
            or nbytes == 0
            or offset > page["nbytes"]
            or nbytes > page["nbytes"] - offset
            or (expected_elements is not None and nbytes != expected_elements * element_nbytes)
        ):
            raise ValueError("invalid AEXRT V2 memory value")
        seen_values.add(value_id)
        records.append(
            {
                "value_id": value_id,
                "flags": value_flags,
                "page_id": page_id,
                "storage_dtype": dtype,
                "storage_layout": layout,
                "offset": offset,
                "nbytes": nbytes,
            }
        )
    return {
        "version": version,
        "flags": flags,
        "alignment": alignment,
        "total_nbytes": total_nbytes,
        "pages": pages,
        "records": records,
        "by_value": {record["value_id"]: record for record in records},
    }


__all__ = [
    "LAYOUT_BLOCKED_NCHW8",
    "LAYOUT_LINEAR_NCHW",
    "MEMORY_PLAN_VERSION",
    "PRECISION_FLOAT16",
    "PRECISION_FLOAT32",
    "build_physical_value_dag",
    "build_page_colored_arena",
    "encode_memory_plan_v2",
    "inspect_memory_plan_v2",
]
