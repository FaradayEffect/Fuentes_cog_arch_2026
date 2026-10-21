"""A small simulated robot (unicycle) for the experiments and the tests: no ROS, no Gazebo.

True motion:   x' = v cos(yaw),   y' = v sin(yaw),   yaw' = omega
The wheels do not obey at once (first-order lag) and slip a little (random noise on v and omega).
The measured pose has Gaussian noise, like the pose from AMCL, so the Kalman filter has work to do.
"""
from __future__ import annotations

import math
import random
from typing import Tuple

from .lqg import wrap

Pose = Tuple[float, float, float]


class Unicycle:
    def __init__(self, pose: Pose, seed: int = 0, pose_noise: float = 0.0, yaw_noise: float = 0.0,
                 slip: float = 0.03, slip_w: float = 0.03, lag_v: float = 0.15, lag_w: float = 0.10,
                 v_hw: float = 0.22, w_hw: float = 2.84, physics_dt: float = 0.01):
        self.x, self.y, self.yaw = (float(v) for v in pose)
        self.v = 0.0                   # what the wheels really do now
        self.w = 0.0
        self.rng = random.Random(seed)
        self.pose_noise, self.yaw_noise = pose_noise, yaw_noise
        self.slip, self.slip_w = slip, slip_w
        self.lag_v, self.lag_w = lag_v, lag_w
        self.v_hw, self.w_hw = v_hw, w_hw          # limits of the motors (TurtleBot3 Burger)
        self.physics_dt = physics_dt

    @property
    def pose(self) -> Pose:
        return self.x, self.y, self.yaw

    def step(self, v_cmd: float, w_cmd: float, dt: float) -> None:
        """Drive with the command (v_cmd, w_cmd) for dt seconds."""
        v_cmd = max(-self.v_hw, min(self.v_hw, v_cmd))
        w_cmd = max(-self.w_hw, min(self.w_hw, w_cmd))
        k_v = 1.0 + self.rng.gauss(0.0, self.slip)                 # slip, the same during one command
        d_w = self.rng.gauss(0.0, self.slip_w)
        n = max(1, int(round(dt / self.physics_dt)))
        h = dt / n
        for _ in range(n):
            self.v += (v_cmd - self.v) * min(1.0, h / self.lag_v)
            self.w += (w_cmd - self.w) * min(1.0, h / self.lag_w)
            v, w = self.v * k_v, self.w + (d_w if abs(self.w) > 1e-6 or abs(self.v) > 1e-6 else 0.0)
            self.x += v * math.cos(self.yaw) * h
            self.y += v * math.sin(self.yaw) * h
            self.yaw = wrap(self.yaw + w * h)

    def measure(self) -> Pose:
        """The pose that the controller sees (with noise)."""
        return (self.x + self.rng.gauss(0.0, self.pose_noise) if self.pose_noise else self.x,
                self.y + self.rng.gauss(0.0, self.pose_noise) if self.pose_noise else self.y,
                wrap(self.yaw + self.rng.gauss(0.0, self.yaw_noise)) if self.yaw_noise else self.yaw)
