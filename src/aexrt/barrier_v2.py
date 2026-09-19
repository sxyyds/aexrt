from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from typing import Any

from .memory_v2 import build_physical_value_dag


ARENA_BARRIER_PLAN_VERSION = 1
ARENA_BARRIER_PLAN_FLAG_AUTHORITATIVE = 1 << 0
ARENA_BARRIER_KIND_TRANSITION_SRV = 1
ARENA_BARRIER_KIND_TRANSITION_UAV = 2
ARENA_BARRIER_KIND_UAV_REUSE = 3
ARENA_BARRIER_PLAN_HEADER_SIZE = 32
ARENA_BARRIER_RECORD_SIZE = 40

_PAGE_STATE_COMMON = 0
_PAGE_STATE_SRV = 1
_PAGE_STATE_UAV = 2


def build_arena_barrier_plan(
    replay: dict[str, Any],
    physical_dispatch_plan: Sequence[dict[str, Any]],
    memory_plan: Mapping[str, Any],
    parallel_window: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compile arena resource-state and reuse hazards onto the dispatch timeline.

    A buffer transition already orders prior UAV work.  Consequently a UAV
    barrier is needed only when a page remains UAV and a later physical
    dispatch overwrites a byte range that is still represented in the pending
    UAV-write set.  This is the same information the runtime used to infer
    from individual buffer views, but it is deterministic compiler state and
    belongs in the engine.

    parallel_window (begin, end, side-per-dispatch): for a dual-queue window,
    pages read by the secondary side are transitioned to SRV at the fork
    point (the dispatch just before the window) instead of at their first
    consumer, which may record on the main side and execute concurrently.
    """
    dag = build_physical_value_dag(replay, physical_dispatch_plan)
    dispatches = dag["dispatches"]
    pages = sorted(memory_plan["pages"], key=lambda item: int(item["page_id"]))
    by_value = memory_plan["by_value"]
    page_count = len(pages)
    if page_count == 0:
        raise ValueError("arena barrier plan requires at least one memory page")
    if [int(page["page_id"]) for page in pages] != list(range(page_count)):
        raise ValueError("arena barrier plan requires dense memory page ids")

    canonical_values = dag["canonical_values"]
    page_states = [_PAGE_STATE_COMMON] * page_count
    pending: list[list[tuple[int, int, int, int]]] = [
        [] for _ in range(page_count)
    ]
    records: list[dict[str, int]] = []
    access_inputs: list[set[int]] = []
    for dispatch in dispatches:
        dispatch_index = int(dispatch["dispatch_index"])
        material_inputs: set[int] = set()
        produced = set(dispatch["produced_values"])
        for logical_index in dispatch["logical_indices"]:
            command = replay["commands"][int(logical_index)]
            if command["kind"] in {"VIEW", "ALIAS"}:
                continue
            material_inputs.update(
                int(canonical_values[int(value_id)])
                for value_id in command["inputs"]
            )
        material_inputs.difference_update(produced)
        access_inputs.append(material_inputs)

    # Dual-queue window: hoist secondary-side input pages to SRV at the fork
    # dispatch (window_begin - 1). Without this the transition lands at the
    # first consumer, which may sit on the main side and execute only after
    # the secondary queue has already started reading the page.
    hoist_at_dispatch = -1
    hoist_pages: dict[int, int] = {}
    if parallel_window is not None:
        begin, end, side = (
            int(parallel_window[0]),
            int(parallel_window[1]),
            parallel_window[2],
        )
        if begin >= 1:
            hoist_at_dispatch = begin - 1
            for dispatch_index in range(begin, end):
                if not side[dispatch_index]:
                    continue
                dispatch = dispatches[dispatch_index]
                produced_here = set(dispatch["produced_values"])
                for logical_index in dispatch["logical_indices"]:
                    command = replay["commands"][int(logical_index)]
                    if command["kind"] in {"VIEW", "ALIAS"}:
                        continue
                    for raw_value in command["inputs"]:
                        value_id = int(canonical_values[int(raw_value)])
                        if value_id in produced_here or value_id not in by_value:
                            continue
                        page_id = int(by_value[value_id]["page_id"])
                        hoist_pages.setdefault(page_id, value_id)

    for dispatch in dispatches:
        dispatch_index = int(dispatch["dispatch_index"])
        material_inputs = access_inputs[dispatch_index]
        reads_by_page: dict[int, list[int]] = {}
        for value_id in sorted(material_inputs):
            if value_id not in by_value:
                continue
            page_id = int(by_value[value_id]["page_id"])
            reads_by_page.setdefault(page_id, []).append(value_id)
        read_pages = set(reads_by_page)

        # Outputs consumed only by another logical command inside this same
        # physical dispatch are registers/TGSM, not arena writes.  The value
        # DAG owns this fusion-specific materialization contract so liveness
        # and barriers cannot disagree (notably for the six-op kind-9 chain).
        physical_outputs = set(dispatch["materialized_outputs"])
        write_values = sorted(
            value_id for value_id in physical_outputs if value_id in by_value
        )
        writes_by_page: dict[int, list[tuple[int, int, int, int]]] = {}
        for value_id in write_values:
            item = by_value[value_id]
            page_id = int(item["page_id"])
            offset = int(item["offset"])
            end = offset + int(item["nbytes"])
            if page_id < 0 or page_id >= page_count or offset < 0 or end <= offset:
                raise ValueError("invalid arena write footprint")
            writes_by_page.setdefault(page_id, []).append(
                (offset, end, value_id, dispatch_index)
            )

        write_pages = set(writes_by_page)
        if read_pages & write_pages:
            raise ValueError(
                "physical dispatch requires simultaneous SRV/UAV state on one arena page"
            )

        # Record functions transition inputs before outputs.  A transition
        # clears all pending UAV hazards for that resource.
        for page_id in sorted(read_pages):
            if page_states[page_id] != _PAGE_STATE_SRV:
                value_id = reads_by_page[page_id][0]
                item = by_value[value_id]
                records.append(
                    {
                        "dispatch_index": dispatch_index,
                        "page_id": page_id,
                        "before_value_id": 0xFFFFFFFF,
                        "after_value_id": value_id,
                        "kind": ARENA_BARRIER_KIND_TRANSITION_SRV,
                        "offset": int(item["offset"]),
                        "nbytes": int(item["nbytes"]),
                    }
                )
                page_states[page_id] = _PAGE_STATE_SRV
                pending[page_id].clear()

        for page_id in sorted(write_pages):
            writes = sorted(writes_by_page[page_id])
            for index, current in enumerate(writes):
                if any(
                    current[0] < other[1] and other[0] < current[1]
                    for other in writes[:index]
                ):
                    raise ValueError("physical dispatch has overlapping arena outputs")

            if page_states[page_id] != _PAGE_STATE_UAV:
                value_id = writes[0][2]
                item = by_value[value_id]
                records.append(
                    {
                        "dispatch_index": dispatch_index,
                        "page_id": page_id,
                        "before_value_id": 0xFFFFFFFF,
                        "after_value_id": value_id,
                        "kind": ARENA_BARRIER_KIND_TRANSITION_UAV,
                        "offset": int(item["offset"]),
                        "nbytes": int(item["nbytes"]),
                    }
                )
                page_states[page_id] = _PAGE_STATE_UAV
                pending[page_id].clear()
            else:
                hazards = [
                    (prior, current)
                    for current in writes
                    for prior in pending[page_id]
                    if current[0] < prior[1] and prior[0] < current[1]
                ]
                if hazards:
                    prior, current = min(
                        hazards,
                        key=lambda item: (
                            item[1][2],
                            item[0][3],
                            item[0][2],
                            item[1][0],
                        ),
                    )
                    overlap_offset = max(prior[0], current[0])
                    overlap_end = min(prior[1], current[1])
                    records.append(
                        {
                            "dispatch_index": dispatch_index,
                            "page_id": page_id,
                            "before_value_id": prior[2],
                            "after_value_id": current[2],
                            "kind": ARENA_BARRIER_KIND_UAV_REUSE,
                            "offset": overlap_offset,
                            "nbytes": overlap_end - overlap_offset,
                        }
                    )
                    pending[page_id].clear()
            pending[page_id].extend(writes)

        # Dual-queue fork hoist: AFTER the fork dispatch's own reads/writes, so
        # that a B-input page freshly written here (v138 case: d49 writes the
        # shared concat output, leaving the page UAV) is transitioned back to
        # SRV before the window opens. Running last also keeps records sorted
        # (page may legitimately appear twice at one dispatch: UAV write then
        # SRV hoist).
        if dispatch_index == hoist_at_dispatch and hoist_pages:
            for page_id in sorted(hoist_pages):
                if page_states[page_id] != _PAGE_STATE_SRV:
                    value_id = hoist_pages[page_id]
                    item = by_value[value_id]
                    records.append(
                        {
                            "dispatch_index": dispatch_index,
                            "page_id": page_id,
                            "before_value_id": 0xFFFFFFFF,
                            "after_value_id": value_id,
                            "kind": ARENA_BARRIER_KIND_TRANSITION_SRV,
                            "offset": int(item["offset"]),
                            "nbytes": int(item["nbytes"]),
                        }
                    )
                    page_states[page_id] = _PAGE_STATE_SRV
                    pending[page_id].clear()

    # Stable sort by (dispatch, page) only: generation order already sequences
    # same-page records correctly (write UAV before the dual-queue hoist SRV).
    records.sort(
        key=lambda record: (
            int(record["dispatch_index"]),
            int(record["page_id"]),
        )
    )
    return {
        "version": ARENA_BARRIER_PLAN_VERSION,
        "flags": ARENA_BARRIER_PLAN_FLAG_AUTHORITATIVE,
        "dispatch_count": len(dispatches),
        "page_count": page_count,
        "records": records,
    }


def encode_arena_barrier_plan(plan: Mapping[str, Any]) -> bytes:
    records = list(plan["records"])
    out = bytearray(
        struct.pack(
            "<8I",
            ARENA_BARRIER_PLAN_VERSION,
            ARENA_BARRIER_PLAN_FLAG_AUTHORITATIVE,
            int(plan["dispatch_count"]),
            int(plan["page_count"]),
            len(records),
            ARENA_BARRIER_RECORD_SIZE,
            0,
            0,
        )
    )
    for record in records:
        out += struct.pack(
            "<6IQQ",
            int(record["dispatch_index"]),
            int(record["page_id"]),
            int(record.get("kind", ARENA_BARRIER_KIND_UAV_REUSE)),
            int(record["before_value_id"]),
            int(record["after_value_id"]),
            0,
            int(record["offset"]),
            int(record["nbytes"]),
        )
    return bytes(out)


def inspect_arena_barrier_plan(
    payload: bytes | bytearray | memoryview,
    *,
    dispatch_count: int | None = None,
    page_nbytes: Sequence[int] | None = None,
    value_count: int | None = None,
) -> dict[str, Any]:
    data = bytes(payload)
    if len(data) < ARENA_BARRIER_PLAN_HEADER_SIZE:
        raise ValueError("truncated AEXRT arena barrier plan")
    (
        version,
        flags,
        encoded_dispatch_count,
        page_count,
        record_count,
        record_size,
        reserved0,
        reserved1,
    ) = struct.unpack_from("<8I", data, 0)
    expected_size = ARENA_BARRIER_PLAN_HEADER_SIZE + record_count * record_size
    if (
        version != ARENA_BARRIER_PLAN_VERSION
        or flags != ARENA_BARRIER_PLAN_FLAG_AUTHORITATIVE
        or encoded_dispatch_count == 0
        or page_count == 0
        or record_size != ARENA_BARRIER_RECORD_SIZE
        or reserved0 != 0
        or reserved1 != 0
        or expected_size != len(data)
        or (dispatch_count is not None and encoded_dispatch_count != dispatch_count)
        or (page_nbytes is not None and page_count != len(page_nbytes))
    ):
        raise ValueError("invalid AEXRT arena barrier-plan header")

    records: list[dict[str, int]] = []
    previous_key: tuple[int, int] | None = None
    cursor = ARENA_BARRIER_PLAN_HEADER_SIZE
    for _ in range(record_count):
        (
            barrier_dispatch,
            page_id,
            kind,
            before_value_id,
            after_value_id,
            reserved,
            offset,
            nbytes,
        ) = struct.unpack_from("<6IQQ", data, cursor)
        cursor += ARENA_BARRIER_RECORD_SIZE
        key = (barrier_dispatch, page_id)
        page_size = None if page_nbytes is None else int(page_nbytes[page_id]) if page_id < page_count else None
        if (
            barrier_dispatch >= encoded_dispatch_count
            or page_id >= page_count
            or kind not in {
                ARENA_BARRIER_KIND_TRANSITION_SRV,
                ARENA_BARRIER_KIND_TRANSITION_UAV,
                ARENA_BARRIER_KIND_UAV_REUSE,
            }
            or (
                kind == ARENA_BARRIER_KIND_UAV_REUSE
                and (before_value_id == 0xFFFFFFFF or before_value_id == after_value_id)
            )
            or (
                kind != ARENA_BARRIER_KIND_UAV_REUSE
                and before_value_id != 0xFFFFFFFF
            )
            or reserved != 0
            or nbytes == 0
            or (page_size is not None and (offset > page_size or nbytes > page_size - offset))
            or (value_count is not None and (
                after_value_id >= value_count
                or (before_value_id != 0xFFFFFFFF and before_value_id >= value_count)
            ))
            or (previous_key is not None and key < previous_key)
        ):
            # Equal (dispatch, page) keys are allowed: dual-queue fork hoists
            # may append an SRV transition after a same-dispatch UAV write.
            raise ValueError("invalid AEXRT arena barrier record")
        previous_key = key
        records.append(
            {
                "dispatch_index": barrier_dispatch,
                "page_id": page_id,
                "before_value_id": before_value_id,
                "after_value_id": after_value_id,
                "kind": kind,
                "offset": offset,
                "nbytes": nbytes,
            }
        )
    return {
        "version": version,
        "flags": flags,
        "dispatch_count": encoded_dispatch_count,
        "page_count": page_count,
        "records": records,
    }


__all__ = [
    "ARENA_BARRIER_KIND_TRANSITION_SRV",
    "ARENA_BARRIER_KIND_TRANSITION_UAV",
    "ARENA_BARRIER_KIND_UAV_REUSE",
    "ARENA_BARRIER_PLAN_FLAG_AUTHORITATIVE",
    "ARENA_BARRIER_PLAN_HEADER_SIZE",
    "ARENA_BARRIER_PLAN_VERSION",
    "ARENA_BARRIER_RECORD_SIZE",
    "build_arena_barrier_plan",
    "encode_arena_barrier_plan",
    "inspect_arena_barrier_plan",
]
