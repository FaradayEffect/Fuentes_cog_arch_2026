"""LqgSimRobot: the pure-Python robot of Lab 1 (SimRobot), but its navigate steps are driven by the LQG tracker
on the simulated unicycle, with a noisy pose. The ACT-R agent of Lab 1 runs on it unchanged, so whole missions
can be checked with the LQG controller without Gazebo (a test does exactly that).

It is NOT Gazebo: no lidar, no forklift, the cubes are carried by the virtual gripper of SimRobot.
"""
from __future__ import annotations

from typing import Optional

from actr_lab1.core.sim import SimRobot
from actr_lab1.core.world import Pose2D

from .tracker import LqgTracker, TrackerConfig
from .unicycle import Unicycle


class LqgSimRobot(SimRobot):
    def __init__(self, layout, grid, cfg: Optional[TrackerConfig] = None, pose_noise: float = 0.01,
                 yaw_noise: float = 0.02, seed: int = 0, **kw):
        super().__init__(layout, grid, seed=seed, **kw)
        self.cfg = cfg or TrackerConfig()
        self.body = Unicycle(self.pose.as_tuple(), seed=seed, pose_noise=pose_noise, yaw_noise=yaw_noise)
        self.tracker: Optional[LqgTracker] = None
        self._clock = 0.0                        # time of the next control step

    def start_navigate(self, pose: Pose2D) -> None:
        self.nav_goals += 1
        cells = self.grid.astar(self.pose.xy(), pose.xy(), self.layout.robot_radius + 0.08)
        if cells is None:
            self._state, self._reason, self._motion = "failed", "no path to the goal", None
            self.nav_failures += 1
            return
        self.tracker = LqgTracker(self.cfg)
        self.tracker.set_path([self.pose.xy()] + list(cells[1:-1]) + [pose.xy()], pose.as_tuple())
        self._motion = {"goal": pose, "remaining": self.tracker.path.length}
        self._state, self._reason = "running", ""
        self._clock = self.t

    def _advance(self, dt: float) -> None:
        """Called by SimRobot.spin() while a navigate step runs: control steps of the LQG up to the time t."""
        while self._motion is not None and self._clock <= self.t + 1e-9:
            cmd = self.tracker.step(self.body.measure(), self._clock)
            self.body.step(cmd.v, cmd.w, self.cfg.dt)
            self._clock += self.cfg.dt
            x, y, yaw = self.body.pose
            self.distance += abs(cmd.v) * self.cfg.dt
            self.pose = Pose2D(x, y, yaw)
            self._motion["remaining"] = cmd.remaining
            if cmd.state == "succeeded":
                self._state, self._reason, self._motion = "succeeded", "", None
            elif cmd.state == "failed":
                self._fail(f"aborted: {cmd.reason}")
