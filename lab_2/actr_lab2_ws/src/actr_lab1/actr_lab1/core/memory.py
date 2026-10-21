"""Declarative memory with ACT-R base-level activation.

Every time a chunk is stored or retrieved we remember the time. The activation is

    B = ln( sum_j (now - t_j) ^ -d )          (+ optional noise)

so a chunk that was used often and recently is easy to retrieve, and an old
one is slowly forgotten. A chunk is returned only if its activation is above
the retrieval threshold. Facts marked permanent (the map of places) never fade.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .chunks import Chunk

Pattern = Union[Dict[str, Any], Callable[[Chunk], bool]]


@dataclass
class MemoryEntry:
    chunk: Chunk
    times: List[float] = field(default_factory=list)
    permanent: bool = False


class DeclarativeMemory:
    def __init__(self, clock: Callable[[], float], decay: float = 0.5, threshold: float = -3.0,
                 noise: float = 0.0, rng: Optional[random.Random] = None, min_age: float = 0.5):
        self.clock = clock
        self.decay = decay
        self.threshold = threshold
        self.noise = noise          # scale of the logistic noise (0 = no noise)
        self.rng = rng or random.Random(0)
        self.min_age = min_age      # avoids division by zero for brand new chunks
        self.entries: List[MemoryEntry] = []
        self.retrievals = 0
        self.failures = 0

    # -- storing ----------------------------------------------------------------
    def add(self, chunk: Chunk, permanent: bool = False) -> MemoryEntry:
        entry = MemoryEntry(chunk, [self.clock()], permanent)
        self.entries.append(entry)
        return entry

    def __len__(self) -> int:
        return len(self.entries)

    # -- activation --------------------------------------------------------------
    def activation(self, entry: MemoryEntry, now: Optional[float] = None) -> float:
        if entry.permanent:
            return 10.0
        now = self.clock() if now is None else now
        total = sum(max(now - t, self.min_age) ** (-self.decay) for t in entry.times)
        value = math.log(total)
        if self.noise > 0:
            u = min(max(self.rng.random(), 1e-9), 1 - 1e-9)
            value += self.noise * math.log(u / (1 - u))   # logistic noise
        return value

    # -- retrieval ---------------------------------------------------------------
    @staticmethod
    def _matches(chunk: Chunk, pattern: Pattern) -> bool:
        if callable(pattern):
            return bool(pattern(chunk))
        for key, value in pattern.items():
            if key == "kind":
                if chunk.kind != value:
                    return False
            elif chunk.get(key, None) != value:
                return False
        return True

    def retrieve_all(self, pattern: Pattern) -> List[Chunk]:
        """All matching chunks above the threshold, most active first (no side effects)."""
        now = self.clock()
        scored = []
        for entry in self.entries:
            if self._matches(entry.chunk, pattern):
                a = self.activation(entry, now)
                if a >= self.threshold:
                    scored.append((a, entry))
        scored.sort(key=lambda pair: -pair[0])
        return [e.chunk for _, e in scored]

    def retrieve(self, pattern: Pattern) -> Optional[Chunk]:
        """The most active matching chunk, or None (retrieval failure)."""
        now = self.clock()
        best, best_a = None, None
        for entry in self.entries:
            if self._matches(entry.chunk, pattern):
                a = self.activation(entry, now)
                if a >= self.threshold and (best_a is None or a > best_a):
                    best, best_a = entry, a
        if best is None:
            self.failures += 1
            return None
        best.times.append(now)      # a retrieval is also a presentation: practice makes it stronger
        self.retrievals += 1
        return best.chunk
