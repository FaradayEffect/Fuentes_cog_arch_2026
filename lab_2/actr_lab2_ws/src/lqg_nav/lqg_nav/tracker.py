"""LqgTracker: the controller that drives the robot along the path of the Nav2 planner.

step() is called once every dt (10 Hz). In: the measured pose (and the lidar points). Out: (v, omega) and the
state: running, succeeded or failed.

Phases:
    align   turn in place until the robot looks along the path (the linear model needs small errors)
    track   steering = LQG (LQR + Kalman filter); speed = a simple profile (slower in curves and near the goal)
    turn    at the goal: turn in place to the yaw of the goal
SAFETY LAYER 4, inside the controller (the lowest level of the safety layer):
    speed and acceleration limits; stop for an obstacle in front (lidar); no turning in place when something is
    too close for the forks; failed after a long block or without progress; a new path when far from the path.

steering = "lqr" (no Kalman filter) and "pure_pursuit" (a classic geometric controller) exist only to compare:
everything else (speed, phases, safety) is the same, so only the steering law differs.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple

from .lqg import KalmanFilter, LqrSchedule, model, wrap
from .path import RefPath

Pose = Tuple[float, float, float]


@dataclass
class TrackerConfig:
    steering: str = "lqg"         # lqg | lqr | pure_pursuit
    dt: float = 0.1               # control period, s (10 Hz, as the Nav2 controller of TurtleBot3)
    # limits
    v_max: float = 0.20           # m/s (TurtleBot3 Burger: 0.22)
    v_min: float = 0.04           # m/s while the goal is not reached
    w_max: float = 1.0            # rad/s (as max_vel_theta of the Nav2 controller of TurtleBot3)
    acc_max: float = 0.5          # m/s^2
    alpha_max: float = 3.0        # rad/s^2
    lat_acc_max: float = 0.3      # m/s^2: v^2 * |curvature| stays below this
    goal_slow: float = 0.8        # 1/s: v <= goal_slow * (path left)
    # LQR weights: errors cost x'Qx, extra turning costs u'Ru
    q_y: float = 100.0            # 1 / (10 cm)^2
    q_theta: float = 6.0          # about 1 / (0.4 rad)^2
    r: float = 1.0                # 1 / (1 rad/s)^2
    # Kalman filter (standard deviations): the model in one step, and the measured pose
    w_y: float = 0.003            # m
    w_theta: float = 0.01         # rad
    v_y: float = 0.02             # m
    v_theta: float = 0.03         # rad
    # pure pursuit (comparison only)
    lookahead: float = 0.30       # m
    # phases
    align_enter: float = 0.8      # rad: turn in place first when the heading error is bigger
    align_exit: float = 0.2       # rad
    k_turn: float = 2.0           # turning in place: omega = -k_turn * error
    w_turn_min: float = 0.25      # rad/s, so that the turn does not stop just before the end
    goal_precision: float = 0.03  # m: this close to the goal, stop even before the end of the path
    xy_tolerance: float = 0.25    # m: at the end of the path the goal is reached within this (as Nav2)
    yaw_tolerance: float = 0.10   # rad (Nav2 of TurtleBot3: 0.25)
    lost_distance: float = 0.5    # m from the path: ask for a new path
    # safety
    stop_distance: float = 0.45   # m in front of the centre: stop for an obstacle closer than this
    slow_distance: float = 0.75   # m: slow down
    slow_speed: float = 0.08      # m/s
    half_width: float = 0.15      # m: half width of the corridor that is checked in front
    turn_clearance: float = 0.0   # m: no turning in place with an obstacle closer (the forks sweep a circle)
    blocked_timeout: float = 5.0  # s blocked by an obstacle: failed (ACT-R decides what to do next)
    stuck_time: float = 12.0      # s with less than stuck_progress m of progress: failed
    stuck_progress: float = 0.05  # m


@dataclass
class Command:
    v: float
    w: float
    state: str = "running"        # running | succeeded | failed
    reason: str = ""
    phase: str = ""
    e_y: float = 0.0              # measured errors (for logs and feedback)
    e_theta: float = 0.0
    remaining: float = 0.0        # m of path left
    new_path: bool = False        # please ask the planner for a new path


class LqgTracker:
    def __init__(self, cfg: Optional[TrackerConfig] = None):
        self.cfg = cfg = cfg or TrackerConfig()
        if cfg.steering not in ("lqg", "lqr", "pure_pursuit"):
            raise ValueError(f"unknown steering '{cfg.steering}' (lqg, lqr or pure_pursuit)")
        self.gains = LqrSchedule(cfg.q_y, cfg.q_theta, cfg.r, cfg.dt)
        self.kf = KalmanFilter(cfg.w_y, cfg.w_theta, cfg.v_y, cfg.v_theta)
        self.path: Optional[RefPath] = None
        self.goal: Pose = (0.0, 0.0, 0.0)
        self.phase = "start"
        self.index = 0
        self.v = 0.0                  # last command (for the acceleration limits)
        self.w = 0.0
        self._u = 0.0                 # last extra turn u that was really sent (the Kalman filter needs it)
        self._blocked_since: Optional[float] = None
        self._history: Deque[Tuple[float, float, float]] = deque()

    # ------------------------------------------------------------------------------------------ the path
    def set_path(self, points: Sequence[Sequence[float]], goal: Pose, replan: bool = False) -> None:
        """A new path. replan=True: the same goal, a fresher path: keep tracking (and the Kalman filter)."""
        self.path = RefPath(points)
        self.index = 0
        keep = replan and self.phase == "track" and tuple(goal) == tuple(self.goal)
        self.goal = tuple(goal)
        if not keep:
            self.phase = "start"
            self.kf.ready = False
            self._history.clear()

    def pause(self) -> None:
        """The node sent zero speed without a step (no pose or no lidar): start again from rest, and the Kalman
        filter starts again from the next measurement."""
        self.v = self.w = self._u = 0.0
        self.kf.ready = False
        self._history.clear()

    # ------------------------------------------------------------------------------------------ one step
    def step(self, pose: Pose, t: float, obstacles: Sequence[Tuple[float, float]] = ()) -> Command:
        cfg = self.cfg
        if self.path is None:
            return self._stop("failed", "no path")
        x, y, yaw = pose
        gx, gy, gyaw = self.goal
        to_goal = math.hypot(gx - x, gy - y)

        if self.phase != "turn" and to_goal <= cfg.goal_precision:
            self.phase = "turn"
        if self.phase == "turn":                                       # at the goal: only the yaw is left
            err = wrap(yaw - gyaw)
            if abs(err) <= cfg.yaw_tolerance:
                return self._stop("succeeded", phase="turn")
            if self._no_room_to_turn(obstacles):                       # the place is reached: the yaw is not
                return self._stop("succeeded", "the last turn was skipped: no room to turn", phase="turn")
            return self._turn_in_place(err, obstacles, t)

        i = self.index = self.path.nearest(x, y, self.index)
        e_y, e_th = self.path.errors(i, x, y, yaw)
        left = self.path.remaining(i)
        if left <= 0.02:                                               # the end of the path
            if to_goal <= cfg.xy_tolerance:
                self.phase = "turn"
                return self.step(pose, t, obstacles)
            end_gap = math.hypot(gx - self.path.end[0], gy - self.path.end[1])
            if end_gap > cfg.xy_tolerance:                             # the planner could not reach the goal
                return self._stop("failed", f"the path ends {end_gap:.2f} m from the goal", e_y=e_y, e_th=e_th)
            return self._stop("running", "passed the end of the path", new_path=True, e_y=e_y, e_th=e_th)
        if abs(e_y) > cfg.lost_distance:
            return self._stop("running", "too far from the path", new_path=True, e_y=e_y, e_th=e_th)

        if self.phase in ("start", "align"):                           # look along the path first
            limit = cfg.align_exit if self.phase == "align" else cfg.align_enter
            if abs(e_th) > limit:
                self.phase = "align"
                return self._turn_in_place(e_th, obstacles, t, e_y=e_y, left=left)
            self.phase = "track"
            self.kf.ready = False
            self._history.clear()
        elif abs(e_th) > cfg.align_enter:
            self.phase = "align"
            return self._turn_in_place(e_th, obstacles, t, e_y=e_y, left=left)

        # ---- speed profile
        k_ahead = self.path.max_curvature(i, 0.4)
        v_ref = min(cfg.v_max,
                    math.sqrt(cfg.lat_acc_max / k_ahead) if k_ahead > 1e-6 else cfg.v_max,   # curves
                    cfg.goal_slow * left,                                                     # the goal
                    cfg.v_max * max(0.0, math.cos(e_th)))                                     # looking away
        v_ref = max(v_ref, cfg.v_min)

        # ---- SAFETY LAYER 4: an obstacle in the corridor in front of the robot (lidar)
        front = min((px for px, py in obstacles if 0.0 < px <= cfg.slow_distance and abs(py) <= cfg.half_width),
                    default=None)
        if front is not None and front <= cfg.stop_distance:
            return self._blocked(t, f"obstacle {front:.2f} m in front", e_y, e_th, left)
        self._blocked_since = None
        if front is not None:
            v_ref = min(v_ref, cfg.slow_speed)

        v = self._limit(self.v, v_ref, cfg.acc_max * cfg.dt)  # SAFETY LAYER 4: speed and acceleration limits
        kappa = self.path.curvatures[i]
        w_ref = self._steer(i, x, y, yaw, e_y, e_th, v, kappa)
        w = self._limit(self.w, max(-cfg.w_max, min(cfg.w_max, w_ref)), cfg.alpha_max * cfg.dt)
        self.v, self.w = v, w
        self._u = w - v * kappa

        # ---- progress: the robot must move along
        self._history.append((t, x, y))
        while self._history and t - self._history[0][0] > cfg.stuck_time:
            t0, x0, y0 = self._history.popleft()
            if math.hypot(x - x0, y - y0) < cfg.stuck_progress:
                return self._stop("failed", f"no progress in {cfg.stuck_time:.0f} s", e_y=e_y, e_th=e_th)
        return Command(v, w, "running", phase="track", e_y=e_y, e_theta=e_th, remaining=left)

    # ------------------------------------------------------------------------------------ steering laws
    def _steer(self, i: int, x: float, y: float, yaw: float, e_y: float, e_th: float, v: float,
               kappa: float) -> float:
        cfg = self.cfg
        if cfg.steering == "pure_pursuit":
            tx, ty = self.path.ahead(i, cfg.lookahead)
            dx, dy = tx - x, ty - y
            ly = -math.sin(yaw) * dx + math.cos(yaw) * dy               # the target point in the robot frame
            d2 = dx * dx + dy * dy
            return v * 2.0 * ly / d2 if d2 > 1e-6 else 0.0
        z = (e_y, e_th)
        if cfg.steering == "lqg":
            if not self.kf.ready:
                self.kf.reset(z)
            else:
                A, B = model(self.v, cfg.dt)                             # the step that has just happened
                self.kf.predict(A, B, self._u)
                self.kf.update(z)
            z = self.kf.estimate()
        k_y, k_th = self.gains.gain(v)
        return v * kappa - (k_y * z[0] + k_th * z[1])                    # feedforward + LQR (u = -K x)

    # ------------------------------------------------------------------------------------------ helpers
    def _turn_in_place(self, err: float, obstacles, t: float, e_y: float = 0.0, left: float = 0.0) -> Command:
        cfg = self.cfg
        if self._no_room_to_turn(obstacles):
            return self._blocked(t, "no room to turn in place", e_y, err, left)
        self._blocked_since = None
        w_ref = -cfg.k_turn * err
        if abs(w_ref) < cfg.w_turn_min:
            w_ref = math.copysign(cfg.w_turn_min, w_ref)
        self.v = self._limit(self.v, 0.0, cfg.acc_max * cfg.dt)
        self.w = self._limit(self.w, max(-cfg.w_max, min(cfg.w_max, w_ref)), cfg.alpha_max * cfg.dt)
        self._u = 0.0
        return Command(self.v, self.w, "running", phase=self.phase, e_y=e_y, e_theta=err, remaining=left)

    def _no_room_to_turn(self, obstacles) -> bool:  # SAFETY LAYER 4: the forks sweep a circle
        r = self.cfg.turn_clearance
        return r > 0 and any(math.hypot(px, py) <= r for px, py in obstacles)

    def _blocked(self, t: float, why: str, e_y: float, e_th: float, left: float) -> Command:
        """Safety first: stop at once (no smooth braking) and wait. Too long: failed."""
        self.v = self.w = self._u = 0.0
        self._history.clear()
        if self._blocked_since is None:
            self._blocked_since = t
        if t - self._blocked_since >= self.cfg.blocked_timeout:
            return Command(0.0, 0.0, "failed", f"blocked: {why}", self.phase, e_y, e_th, left)
        return Command(0.0, 0.0, "running", f"waiting: {why}", self.phase, e_y, e_th, left, new_path=True)

    def _stop(self, state: str, reason: str = "", phase: str = "", new_path: bool = False,
              e_y: float = 0.0, e_th: float = 0.0) -> Command:
        self.v = self.w = self._u = 0.0
        return Command(0.0, 0.0, state, reason, phase or self.phase, e_y, e_th, 0.0, new_path)

    @staticmethod
    def _limit(old: float, new: float, step: float) -> float:
        return old + max(-step, min(step, new - old))
