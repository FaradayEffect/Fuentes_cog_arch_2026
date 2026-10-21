"""The only door between the agent and a robot.

The agent (ACT-R part) never imports ROS. It talks to anything that has these methods:
  * SimRobot (core/sim.py)  - pure Python, for tests and for the demo without Gazebo,
  * RosRobot (ros/ros_robot.py) - Nav2 + Gazebo.
That is why the same agent code runs in both worlds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .world import Pose2D


@dataclass
class Observation:
    t: float
    pose: Pose2D
    holding: Optional[str]
    cubes: Dict[str, Tuple[float, float]]         # real cube positions (x, y)
    localized: bool = True                        # is the robot sure where it is?
    cov: float = 0.0                              # trace of the position covariance (AMCL)
    nav: Dict[str, Any] = field(default_factory=dict)   # feedback from the navigation goal


@dataclass
class MotionStatus:
    state: str                  # "idle", "running", "succeeded", "failed" or "canceled"
    reason: str = ""
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    ok: bool
    reason: str = ""


class RobotAPI:
    """Interface description (a base class that documents the methods)."""

    def now(self) -> float:
        """Time in seconds (simulation time)."""
        raise NotImplementedError

    def spin(self, dt: float) -> None:
        """Let the world run for about dt seconds (or process messages for dt seconds)."""
        raise NotImplementedError

    def observe(self) -> Observation:
        raise NotImplementedError

    def start_navigate(self, pose: Pose2D) -> None:
        """Start driving to the pose. Does not wait."""
        raise NotImplementedError

    def poll_motion(self) -> MotionStatus:
        raise NotImplementedError

    def cancel_motion(self) -> None:
        raise NotImplementedError

    def pick(self, cube: str) -> ActionResult:
        raise NotImplementedError

    def place(self, cube: str, xy: Tuple[float, float]) -> ActionResult:
        raise NotImplementedError

    def stop(self) -> None:
        """Emergency stop: cancel the goal and command zero speed."""
        raise NotImplementedError
