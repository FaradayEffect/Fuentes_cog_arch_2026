"""Plan steps and the plan validator.

The LLM writes a plan as JSON. Before the robot may use it, the validator checks:
  * the JSON has the right form,
  * every action is in the allowed list,
  * every name (place, cube) exists,
  * the steps make sense when we "play" the plan on paper (the robot holds one cube at a time),
  * the plan really delivers what the mission asks (if we know it).
The error texts are short, because we also send them back to the LLM when it must try again.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .nlp import resolve_name
from .world import Layout

ACTIONS = ("navigate", "pick", "place")

SYNONYMS = {
    "go": "navigate", "goto": "navigate", "go_to": "navigate", "move": "navigate",
    "move_to": "navigate", "drive": "navigate", "drive_to": "navigate",
    "take": "pick", "grab": "pick", "pickup": "pick", "pick_up": "pick", "get": "pick",
    "drop": "place", "put": "place", "deliver": "place", "release": "place", "put_down": "place",
}

TARGET_KEYS = ("target", "object", "place", "room", "cube", "destination", "to")


@dataclass(frozen=True)
class PlanStep:
    action: str
    target: str
    variant: Optional[int] = None   # which waypoint / approach pose to use (set by recovery)

    def label(self) -> str:
        return f"{self.action}({self.target})"


@dataclass
class Validation:
    valid: bool
    steps: Tuple[PlanStep, ...]
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]


STEP_TEXT_RE = re.compile(r"^\s*([A-Za-z_]+)\s*\(\s*([^)]*?)\s*\)\s*$")


class PlanValidator:
    def __init__(self, layout: Layout, max_steps: int = 24):
        self.layout = layout
        self.max_steps = max_steps

    def _known_names(self) -> str:
        return ", ".join(list(self.layout.places) + list(self.layout.cubes))

    def validate(self, raw: Any, state: Optional[Dict[str, Any]] = None,
                 expected: Sequence[Tuple[str, str]] = ()) -> Validation:
        state = state or {}
        errors: List[str] = []
        warnings: List[str] = []

        items = raw.get("steps") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return Validation(False, (), ("the answer must be a JSON object with a list 'steps'",), ())
        if len(items) > self.max_steps:
            errors.append(f"too many steps ({len(items)}), the maximum is {self.max_steps}")
            items = items[: self.max_steps]

        steps: List[PlanStep] = []
        for i, item in enumerate(items, start=1):
            step = self._read_step(i, item, errors)
            if step is not None:
                steps.append(step)

        # --- play the plan on paper -------------------------------------------------
        holding = state.get("holding")
        at = state.get("robot_place", "")
        near: Optional[str] = None
        cube_places: Dict[str, Optional[str]] = dict(state.get("cube_places", {}))
        for c in self.layout.cubes:
            cube_places.setdefault(c, None)
        for i, s in enumerate(steps, start=1):
            if s.action == "navigate":
                if s.target in self.layout.cubes:
                    near = s.target
                else:
                    at, near = s.target, None
            elif s.action == "pick":
                if s.target not in self.layout.cubes:
                    errors.append(f"step {i}: pick needs a cube, but '{s.target}' is not a cube")
                    continue
                if holding:
                    errors.append(f"step {i}: pick({s.target}) but the robot already holds {holding}; place it first")
                    continue
                if near != s.target:
                    warnings.append(f"step {i}: pick({s.target}) without navigate({s.target}) before it")
                holding = s.target
                cube_places[s.target] = None
            elif s.action == "place":
                if not self.layout.is_room(s.target) and not self.layout.is_forbidden_place(s.target):
                    errors.append(f"step {i}: place needs a room, but '{s.target}' is not a room")
                    continue
                if not holding:
                    errors.append(f"step {i}: place({s.target}) but the robot holds no cube; pick a cube first")
                    continue
                if at != s.target:
                    warnings.append(f"step {i}: place({s.target}) without navigate({s.target}) before it")
                cube_places[holding] = s.target
                holding = None
        if holding and not errors:
            warnings.append(f"the plan ends while the robot still holds {holding}")

        for cube, room in expected:
            if cube_places.get(cube) != room:
                errors.append(f"the plan does not deliver {cube} to {room}")

        if not steps and not errors and not expected:
            errors.append("the plan has no steps")      # (an empty plan is fine if the goal is already true)
        return Validation(not errors, tuple(steps), tuple(errors), tuple(warnings))

    # -- one step ----------------------------------------------------------------
    def _read_step(self, i: int, item: Any, errors: List[str]) -> Optional[PlanStep]:
        action, target = None, None
        if isinstance(item, str):
            m = STEP_TEXT_RE.match(item)
            if not m:
                errors.append(f"step {i}: cannot read '{item}'")
                return None
            action, target = m.group(1), m.group(2)
        elif isinstance(item, dict):
            action = item.get("action")
            for key in TARGET_KEYS:
                if item.get(key):
                    target = item.get(key)
                    break
        else:
            errors.append(f"step {i}: a step must be an object with 'action' and 'target'")
            return None

        if not isinstance(action, str):
            errors.append(f"step {i}: missing 'action'")
            return None
        act = action.strip().lower()
        act = SYNONYMS.get(act, act)
        if act not in ACTIONS:
            errors.append(f"step {i}: action '{action}' is not allowed (allowed: {', '.join(ACTIONS)})")
            return None
        if not isinstance(target, str) or not target.strip():
            errors.append(f"step {i}: missing 'target'")
            return None
        name = resolve_name(target, self.layout)
        if name is None or (name not in self.layout.places and name not in self.layout.cubes):
            errors.append(f"step {i}: unknown target '{target}' (known: {self._known_names()})")
            return None
        return PlanStep(act, name)
