"""Very small language tools: find cubes and places in a mission text (Russian or English).

Used in two places:
  * MockLLMPlanner - to build a plan without a real LLM,
  * the agent - to know what the mission wants (cube X must end in room Y),
    so the result can be checked against the real world, not against the LLM's words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .world import Layout

CYR_TO_LAT = {"а": "a", "б": "b", "в": "c", "г": "d"}

CUBE_RE = re.compile(r"(?:cubes?|куб\w*)\s*(?:номер\w*|no|#)?\s*_?\s*(\d+(?:\s+(?:и\s+|and\s+)?\d+)*)")
ROOM_RE = re.compile(r"(?:room|комнат\w*)\s*_?\s*([a-zа-я])(?![a-zа-я0-9])")
ALL_RE = re.compile(r"\b(?:оба|обе|все|всех|all|both|every)\b")
CUBE_WORD_RE = re.compile(r"\b(?:cubes?|куб\w*)\b")
CONNECTOR_RE = re.compile(r"\b(?:потом|затем|после этого|then|after that|next)\b")


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9_ ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Mention:
    kind: str       # "cube" or "place"
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class Task:
    kind: str                  # "deliver" (cube -> place) or "visit" (just go there)
    place: str
    cube: Optional[str] = None


def find_mentions(text: str, layout: Layout) -> List[Mention]:
    norm = normalize(text)
    found: List[Mention] = []
    for m in CUBE_RE.finditer(norm):
        for num in re.findall(r"\d+", m.group(1)):
            found.append(Mention("cube", f"cube_{int(num)}", m.start(), m.end()))
    for m in ROOM_RE.finditer(norm):
        letter = CYR_TO_LAT.get(m.group(1), m.group(1))
        found.append(Mention("place", f"room_{letter}", m.start(), m.end()))
    taken = [(m.start, m.end) for m in found]
    for name, place in layout.places.items():
        for alias in place.aliases:
            a = normalize(alias)
            if not a:
                continue
            tail = r"\w*" if re.search(r"[а-я]", a) else ""
            for m in re.finditer(r"(?<![a-zа-я0-9_])" + re.escape(a) + tail + r"(?![a-zа-я0-9_])", norm):
                if any(s < m.end() and m.start() < e for s, e in taken):
                    continue
                found.append(Mention("place", name, m.start(), m.end()))
                taken.append((m.start(), m.end()))
    found.sort(key=lambda m: m.start)
    return found


def resolve_name(text: str, layout: Layout) -> Optional[str]:
    """Turn what the LLM wrote ('Room A', 'комната Б', 'cube0') into an exact name, or None."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t in layout.places or t in layout.cubes:
        return t
    norm = normalize(t)
    for name in list(layout.places) + list(layout.cubes):
        if norm == normalize(name):
            return name
    mentions = find_mentions(t, layout)
    if len(mentions) == 1 and len(norm.split()) <= 4:     # e.g. "room A", "the Room A", "кубик 0"
        return mentions[0].name
    return None


def _pieces(text: str, layout: Layout) -> List[str]:
    """Split a mission into clauses: sentences, commas (only if both sides say something), 'then'."""
    clauses: List[str] = []
    for sentence in re.split(r"[.;!?\n]+", text):
        if not sentence.strip():
            continue
        parts = [p for p in sentence.split(",") if p.strip()]
        merged: List[str] = []
        for p in parts:
            if merged and not find_mentions(p, layout):
                merged[-1] += " " + p       # a part with no cube or place belongs to the previous one
            else:
                merged.append(p)
        for p in merged:
            clauses.extend(c for c in CONNECTOR_RE.split(normalize(p)) if c and c.strip())
    return clauses


def _segments(ms: List[Mention]) -> List[List[Mention]]:
    """Cut one clause into pieces 'cube(s) ... place'. 'cube 0 to room B and cube 1 to room A' has two:
    a new piece starts when a cube is named after a place that came after a cube."""
    out: List[List[Mention]] = [[]]
    has_cube = has_dest = False
    for m in ms:
        if m.kind == "cube" and has_cube and has_dest:
            out.append([])
            has_cube = has_dest = False
        out[-1].append(m)
        if m.kind == "cube":
            has_cube = True
        elif has_cube:
            has_dest = True
    return [seg for seg in out if seg]


def parse_mission(text: str, layout: Layout) -> List[Task]:
    """List of tasks in the order of the text. Never raises; an unknown text gives []."""
    tasks: List[Task] = []
    pending: List[str] = []             # cubes that wait for a destination

    def handle(ms: List[Mention], all_cubes: bool = False) -> None:
        nonlocal pending
        cubes = [m.name for m in ms if m.kind == "cube"]
        places = [m.name for m in ms if m.kind == "place"]
        if all_cubes:
            cubes = list(layout.cubes)  # "both cubes", "all cubes"
        if cubes:
            if places:
                dest = places[-1]           # the last place of the piece is where the cube goes
                first_cube = min((m.start for m in ms if m.kind == "cube"), default=0)
                for m in ms:                # a place named before the cube is a stop on the way
                    if m.kind == "place" and m.start < first_cube and m.name != dest:
                        tasks.append(Task("visit", m.name))
                for c in pending + cubes:
                    tasks.append(Task("deliver", dest, c))
                pending = []
            else:
                pending.extend(cubes)
        elif places:
            if pending:
                for c in pending:
                    tasks.append(Task("deliver", places[-1], c))
                pending = []
            else:
                for p in places:
                    tasks.append(Task("visit", p))

    for clause in _pieces(text, layout):
        ms = find_mentions(clause, layout)
        if not any(m.kind == "cube" for m in ms) and CUBE_WORD_RE.search(clause) and ALL_RE.search(clause):
            handle(ms, all_cubes=True)
        else:
            for seg in _segments(ms):
                handle(seg)
    for c in pending:                       # no destination in the text: use the default room
        tasks.append(Task("deliver", layout.goal_room, c))
    # one task per cube (the last wish wins), keep the order of first appearance
    seen = {}
    for t in tasks:
        if t.kind == "deliver":
            seen[t.cube] = t
    out: List[Task] = []
    done = set()
    for t in tasks:
        if t.kind == "deliver":
            if t.cube in done:
                continue
            done.add(t.cube)
            out.append(seen[t.cube])
        else:
            out.append(t)
    return out


def expected_deliveries(tasks: List[Task]) -> Tuple[Tuple[str, str], ...]:
    return tuple((t.cube, t.place) for t in tasks if t.kind == "deliver")
