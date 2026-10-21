"""Buffers with access control.

In ACT-R a buffer holds one chunk and a module can only talk to the rest of
the system through its own buffers. Here every read and write names the module
that does it, and a policy table says what each module may read and write.
Every write is also saved in an audit list.

This is a software guard inside one process. It stops mistakes (for example
the LLM module writing an action) but it is not an operating-system sandbox.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Set

from .chunks import Chunk, to_jsonable

BUFFER_NAMES = (
    "goal",        # what the mission is and in which phase we are
    "plan",        # the accepted plan and the current step (imaginal buffer)
    "percept",     # what the robot sees now: pose, cubes, localization
    "retrieval",   # the last chunk found in declarative memory
    "proposal",    # a plan from the LLM, not trusted yet
    "action",      # the action we want to do now
    "motor",       # state of the "hand": free, busy or done
    "evaluation",  # reward of the last action
)

ALL = set(BUFFER_NAMES)

# module -> buffers it may read / write
DEFAULT_POLICY: Dict[str, Dict[str, Set[str]]] = {  # SAFETY LAYER 3: who may read / write
    "perception": {"read": set(), "write": {"percept"}},
    "memory": {"read": {"goal", "percept", "action"}, "write": {"retrieval"}},
    "llm": {"read": {"goal", "percept", "retrieval", "evaluation"}, "write": {"proposal"}},
    "validator": {"read": {"proposal", "percept", "goal"}, "write": {"proposal"}},
    "gate": {"read": {"action", "percept", "goal", "plan"}, "write": {"action"}},
    "executor": {"read": {"action", "motor"}, "write": {"motor"}},
    "evaluator": {"read": {"motor", "action", "goal", "plan", "percept"}, "write": {"evaluation"}},
    "procedural": {"read": set(ALL), "write": {"goal", "plan", "action", "proposal", "retrieval", "motor"}},
}


class AccessDenied(Exception):
    """A module tried to read or write a buffer it has no right to use."""


class BufferSystem:
    def __init__(self, policy: Optional[Dict[str, Dict[str, Set[str]]]] = None,
                 clock: Optional[Callable[[], float]] = None, max_audit: int = 20000):
        self.policy = policy if policy is not None else DEFAULT_POLICY
        self.clock = clock or time.monotonic
        self._data: Dict[str, Optional[Chunk]] = {name: None for name in BUFFER_NAMES}
        self.audit: List[dict] = []
        self.max_audit = max_audit
        self.denied = 0

    # -- helpers ------------------------------------------------------------
    def _check(self, op: str, name: str, by: str):
        if name not in self._data:
            raise KeyError(f"unknown buffer: {name}")
        rights = self.policy.get(by)
        allowed = rights is not None and name in rights[op]
        if not allowed:
            self.denied += 1
            self._log(by, op, name, None, False)
            raise AccessDenied(f"module '{by}' may not {op} buffer '{name}'")

    def _log(self, by, op, name, chunk, ok):
        if op == "read" and ok:
            return  # reads are many; only writes and denied reads are kept
        self.audit.append({
            "t": round(self.clock(), 3), "by": by, "op": op, "buffer": name,
            "kind": None if chunk is None else chunk.kind, "ok": ok,
        })
        if len(self.audit) > self.max_audit:
            del self.audit[: self.max_audit // 10]

    # -- API ------------------------------------------------------------------
    def read(self, name: str, by: str) -> Optional[Chunk]:
        self._check("read", name, by)
        return self._data[name]

    def write(self, name: str, chunk: Chunk, by: str) -> None:
        if not isinstance(chunk, Chunk):
            raise TypeError("only chunks can be written to a buffer")
        self._check("write", name, by)
        self._data[name] = chunk
        self._log(by, "write", name, chunk, True)

    def clear(self, name: str, by: str) -> None:
        self._check("write", name, by)
        self._data[name] = None
        self._log(by, "clear", name, None, True)

    def snapshot(self, by: str = "procedural") -> Dict[str, Optional[dict]]:
        """Copy of all buffers that module `by` may read (for logs and prompts)."""
        out = {}
        for name in BUFFER_NAMES:
            rights = self.policy.get(by)
            if rights is not None and name in rights["read"]:
                out[name] = to_jsonable(self._data[name]) if self._data[name] is not None else None
        return out

    # -- small shortcuts for the production system (module 'procedural') -------
    def get(self, name: str) -> Optional[Chunk]:
        return self.read(name, "procedural")

    def slot(self, name: str, slot: str, default=None):
        chunk = self.read(name, "procedural")
        return default if chunk is None else chunk.get(slot, default)
