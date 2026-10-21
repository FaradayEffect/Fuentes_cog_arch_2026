"""Chunks: small facts. Buffers and memory hold nothing else.

A chunk has a kind (its type) and named slots, like a row of a table.
It is immutable: to change a slot you make a new chunk with with_slots().
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class Chunk:
    kind: str
    slots: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # keep a read-only copy so nobody can change the chunk by accident
        object.__setattr__(self, "slots", MappingProxyType(dict(self.slots)))

    def get(self, name: str, default: Any = None) -> Any:
        return self.slots.get(name, default)

    def with_slots(self, **changes: Any) -> "Chunk":
        new_slots = dict(self.slots)
        new_slots.update(changes)
        return Chunk(self.kind, new_slots)

    def __hash__(self):  # slots may hold lists, so a chunk is not hashable
        raise TypeError("Chunk is not hashable")


def to_jsonable(obj: Any) -> Any:
    """Turn chunks, dataclasses, tuples and so on into plain JSON data (for logs)."""
    if isinstance(obj, Chunk):
        return {"kind": obj.kind, **{k: to_jsonable(v) for k, v in obj.slots.items()}}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return str(obj)
        return round(obj, 4)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)
