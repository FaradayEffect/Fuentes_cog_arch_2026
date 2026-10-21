"""Event log: one JSON line per event in a file, and short readable lines on the screen."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from .chunks import to_jsonable


class EventLog:
    def __init__(self, directory: Optional[str] = None, console: bool = True,
                 clock: Optional[Callable[[], float]] = None, name: str = "events.jsonl",
                 keep: bool = False, console_kinds: Optional[set] = None,
                 on_event: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.clock = clock or time.time
        self.console = console
        self.keep = keep
        self.on_event = on_event                # called with every record (the ROS node publishes it)
        self.events: List[Dict[str, Any]] = []
        self.console_kinds = console_kinds      # None = print every event that has a message
        self._file = None
        self.path = None
        if directory:
            os.makedirs(directory, exist_ok=True)
            self.path = os.path.join(directory, name)
            self._file = open(self.path, "w", encoding="utf-8")

    def event(self, kind: str, msg: str = "", **data: Any) -> None:
        t = self.clock()
        record = {"t": round(t, 3), "kind": kind}
        if msg:
            record["msg"] = msg
        record.update({k: to_jsonable(v) for k, v in data.items()})
        if self.keep:
            self.events.append(record)
        if self._file:
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()
        if self.on_event is not None:
            try:
                self.on_event(record)
            except Exception:                   # a broken listener must not stop the robot
                pass
        if self.console and msg and (self.console_kinds is None or kind in self.console_kinds):
            print(f"[{t:7.1f}s] {kind.upper():8s} {msg}", flush=True)

    def of_kind(self, kind: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["kind"] == kind]

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
