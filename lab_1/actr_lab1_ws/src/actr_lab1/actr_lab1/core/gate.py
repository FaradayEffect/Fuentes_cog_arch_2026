"""The safety gate: the last check before an action may reach the robot.

The LLM only proposes. The production rules decide what to try. This gate is the
one place that says "yes, the robot may do this now" (it writes `approved=True`
into the action chunk), and the executor refuses any action that was not approved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .chunks import Chunk
from .world import Layout, OccupancyGrid, Pose2D, dist

ALLOWED_ACTIONS = ("navigate", "pick", "place", "stop")


@dataclass
class GateDecision:
    approved: bool
    rule: str = "ok"
    reason: str = ""


class ActionGate:
    def __init__(self, layout: Layout, grid: Optional[OccupancyGrid], pick_radius: float = 0.95,
                 max_actions: int = 200, margin: float = 0.08):
        self.layout = layout
        self.grid = grid
        self.pick_radius = pick_radius
        self.max_actions = max_actions
        self.margin = margin
        self.checked = 0
        self.denials = 0

    def check(self, action: Chunk, percept: Optional[Chunk]) -> GateDecision:
        d = self._check(action, percept)
        if not d.approved:
            self.denials += 1
        return d

    def _check(self, action: Chunk, percept: Optional[Chunk]) -> GateDecision:
        cmd = action.get("cmd")
        if cmd not in ALLOWED_ACTIONS:
            return GateDecision(False, "allowed_actions", f"'{cmd}' is not an allowed action")
        if cmd == "stop":
            return GateDecision(True, "stop_always_allowed")          # stopping is never refused
        self.checked += 1
        if self.checked > self.max_actions:
            return GateDecision(False, "action_budget", f"more than {self.max_actions} actions in one mission")

        target = action.get("target")
        p = percept
        holding = p.get("holding") if p else None
        rx, ry = (p.get("x", 0.0), p.get("y", 0.0)) if p else (0.0, 0.0)

        if cmd == "navigate":
            if target not in self.layout.places and target not in self.layout.cubes:
                return GateDecision(False, "unknown_target", f"'{target}' is not a known place or cube")
            pose = action.get("pose")
            if pose is None:
                return GateDecision(False, "no_pose", f"no pose found for '{target}'")
            if self.layout.is_forbidden_place(target) or self.layout.in_forbidden(pose[0], pose[1]):
                return GateDecision(False, "forbidden_zone", f"'{target}' is in a zone the robot must not enter")
            if self.grid is not None and not self.grid.is_free(pose[0], pose[1], self.layout.robot_radius + self.margin):
                return GateDecision(False, "target_not_free", "the goal pose is not on free floor")
            if p is not None and not p.get("localized", True):
                return GateDecision(False, "localization_lost", "the robot does not know where it is")
            return GateDecision(True)

        if cmd == "pick":
            if target not in self.layout.cubes:
                return GateDecision(False, "unknown_target", f"'{target}' is not a cube")
            if holding:
                return GateDecision(False, "already_holding", f"the robot already holds {holding}")
            cube_xy = {n: (x, y) for n, x, y, _r in p.get("cubes", ())}.get(target) if p else None
            if cube_xy is None:
                return GateDecision(False, "cube_not_seen", f"{target} is not visible")
            if self.layout.in_forbidden(*cube_xy):
                return GateDecision(False, "forbidden_zone", f"{target} lies in a restricted zone")
            if dist((rx, ry), cube_xy) > self.pick_radius:
                return GateDecision(False, "too_far", f"{target} is {dist((rx, ry), cube_xy):.2f} m away")
            return GateDecision(True)

        if cmd == "place":
            if not holding:
                return GateDecision(False, "not_holding", "the robot holds no cube")
            if target not in self.layout.places:
                return GateDecision(False, "unknown_target", f"'{target}' is not a place")
            if self.layout.is_forbidden_place(target):
                return GateDecision(False, "forbidden_zone", f"'{target}' is a restricted place")
            if p.get("room", "") != target:
                return GateDecision(False, "wrong_room", f"the robot is in '{p.get('room', '')}', not in '{target}'")
            if action.get("drop") is None:
                return GateDecision(False, "no_drop_point", "no free floor to put the cube")
            return GateDecision(True)
        return GateDecision(False, "allowed_actions", "unhandled action")
