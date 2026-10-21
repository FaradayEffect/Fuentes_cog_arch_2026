"""Experiment of Lab 2: three steering laws on the same paths, in a simulated robot with noise.

    python3 -m lqg_nav.experiment                  # about 2 minutes; writes results/lqg/
    python3 -m lqg_nav.experiment --seeds 2 --out /tmp/lqg_test

    pure_pursuit  a classic geometric controller (Nav2 has a variant of it: Regulated Pure Pursuit)
    lqr           the LQR alone, fed with the noisy measured pose
    lqg           LQR + Kalman filter = LQG, the controller of Lab 2
The speed profile, the phases and the safety checks are the same for the three: only the steering differs.
The paths are A* paths on the map of the world (the same map that Nav2 uses). The robot is the small simulated
unicycle of lqg_nav/unicycle.py, NOT Gazebo: the numbers compare the controllers in this model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .lqg import wrap
from .path import RefPath
from .tracker import LqgTracker, TrackerConfig
from .unicycle import Unicycle

CONTROLLERS = ("pure_pursuit", "lqr", "lqg")
# measured pose: (standard deviation of x and y in m, of the yaw in rad)
NOISE = {"none": (0.0, 0.0), "amcl": (0.01, 0.02), "high": (0.03, 0.05)}
# (name, start place, start waypoint, goal place, goal waypoint)
ROUTES = [
    ("hall-roomA", "hall", 0, "room_a", 0),
    ("hall-roomB", "hall", 0, "room_b", 0),
    ("roomA-roomB", "room_a", 1, "room_b", 1),
    ("roomB-hall", "room_b", 0, "hall", 1),
    ("hall-roomA2", "hall", 1, "room_a", 1),
    ("roomB-roomA", "room_b", 1, "room_a", 0),
]
TIME_LIMIT = 150.0       # s of robot time
ROBOT_RADIUS = 0.10      # m: closer to a wall than this = collision
PLAN_RADIUS = 0.25       # m: the A* keeps this far from the walls (like the inflation of the Nav2 costmap)


def load_world():
    """Layout and map of Lab 1 (package actr_lab1, next to this one in src/)."""
    try:
        from actr_lab1.core.paths import load_world as load
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "..", "..", "actr_lab1"))
        from actr_lab1.core.paths import load_world as load
    return load()


def routes(layout, grid) -> List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float], list]]:
    out = []
    for name, a, ia, b, ib in ROUTES:
        start, goal = layout.waypoint(a, ia), layout.waypoint(b, ib)
        cells = grid.astar(start.xy(), goal.xy(), PLAN_RADIUS)
        if cells is None:
            raise RuntimeError(f"no A* path for {name}")
        points = [start.xy()] + cells[1:-1] + [goal.xy()]          # like Nav2: from the robot to the goal
        out.append((name, start.as_tuple(), goal.as_tuple(), points))
    return out


def distance_to_path(path: RefPath, x: float, y: float, hint: int) -> Tuple[float, int]:
    """Exact distance from (x, y) to the polyline of the path (near the index hint)."""
    i = path.nearest(x, y, hint, back=40, ahead=200)
    best = float("inf")
    for j in range(max(0, i - 2), min(len(path) - 1, i + 2)):
        ax, ay, bx, by = path.xs[j], path.ys[j], path.xs[j + 1], path.ys[j + 1]
        dx, dy = bx - ax, by - ay
        f = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy or 1.0)))
        best = min(best, math.hypot(x - ax - f * dx, y - ay - f * dy))
    if len(path) == 1:
        best = math.hypot(x - path.xs[0], y - path.ys[0])
    return best, i


def run_one(steering: str, route, noise: str, seed: int, grid=None, cfg: Optional[TrackerConfig] = None,
            keep_track: bool = False) -> dict:
    name, start, goal, points = route
    cfg = TrackerConfig(**{**(cfg.__dict__ if cfg else {}), "steering": steering})
    tracker = LqgTracker(cfg)
    tracker.set_path(points, goal)
    pose_std, yaw_std = NOISE[noise]
    robot = Unicycle(start, seed=seed, pose_noise=pose_std, yaw_noise=yaw_std)
    t, hint, w_old = 0.0, 0, 0.0
    errors: List[float] = []
    jerks: List[float] = []
    track: List[Tuple[float, float, float, float]] = []      # (t, x, y, omega) when keep_track
    collided = False
    cmd = None
    while t < TIME_LIMIT:
        cmd = tracker.step(robot.measure(), t)
        if cmd.state != "running":
            break
        jerks.append((cmd.w - w_old) / cfg.dt)
        w_old = cmd.w
        robot.step(cmd.v, cmd.w, cfg.dt)
        t += cfg.dt
        x, y, _ = robot.pose
        if keep_track:
            track.append((round(t, 2), x, y, cmd.w))
        if cmd.phase == "track":
            d, hint = distance_to_path(tracker.path, x, y, hint)
            errors.append(d)
        if grid is not None and not grid.is_free(x, y, ROBOT_RADIUS):
            collided = True
    x, y, yaw = robot.pose
    ok = cmd is not None and cmd.state == "succeeded" and not collided
    row = {
        "route": name, "noise": noise, "seed": seed, "controller": steering,
        "success": ok, "reason": "collision" if collided else (cmd.reason if cmd else "no step"),
        "time_s": round(t, 2),
        "rms_cte_cm": round(100 * math.sqrt(sum(e * e for e in errors) / len(errors)), 2) if errors else 0.0,
        "max_cte_cm": round(100 * max(errors), 2) if errors else 0.0,
        "final_xy_cm": round(100 * math.hypot(goal[0] - x, goal[1] - y), 2),
        "final_yaw_deg": round(abs(math.degrees(wrap(yaw - goal[2]))), 2),
        "jerk": round(math.sqrt(sum(j * j for j in jerks) / len(jerks)), 3) if jerks else 0.0,
    }
    if keep_track:
        row["track"] = track
    return row


def summarize(rows: Sequence[dict]) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    for noise in NOISE:
        for c in CONTROLLERS:
            sel = [r for r in rows if r["noise"] == noise and r["controller"] == c]
            if not sel:
                continue
            ok = [r for r in sel if r["success"]]

            def mean(key, subset=sel):
                return statistics.mean(r[key] for r in subset) if subset else float("nan")
            out.setdefault(noise, {})[c] = {
                "runs": len(sel), "success": len(ok), "rms_cte_cm": mean("rms_cte_cm"),
                "max_cte_cm": mean("max_cte_cm"), "time_s": mean("time_s", ok),
                "final_xy_cm": mean("final_xy_cm", ok), "jerk": mean("jerk"),
            }
    return out


def format_table(summary: Dict[str, Dict[str, dict]]) -> str:
    lines = ["| pose noise | controller | success | lateral error RMS, cm | lateral error max, cm | "
             "time, s | final error, cm | turn jerk RMS, rad/s^2 |",
             "|---|---|---|---|---|---|---|---|"]
    for noise, per in summary.items():
        std, ystd = NOISE[noise]
        label = f"{noise} ({100 * std:.0f} cm, {math.degrees(ystd):.1f} deg)"
        for c, s in per.items():
            lines.append(f"| {label} | {c} | {s['success']}/{s['runs']} | {s['rms_cte_cm']:.1f} | "
                         f"{s['max_cte_cm']:.1f} | {s['time_s']:.1f} | {s['final_xy_cm']:.1f} | {s['jerk']:.2f} |")
    return "\n".join(lines)


def run_all(seeds: int = 5, cfg: Optional[TrackerConfig] = None, noises: Sequence[str] = tuple(NOISE),
            controllers: Sequence[str] = CONTROLLERS) -> List[dict]:
    layout, grid = load_world()
    rows = []
    for route in routes(layout, grid):
        for noise in noises:
            for seed in range(seeds):
                for c in controllers:                 # the same noise for the three: a paired comparison
                    rows.append(run_one(c, route, noise, seed, grid, cfg))
    return rows


MISSIONS = [
    "Отвези кубик 0 и кубик 1 в комнату А.", "Bring cube 0 and cube 1 to room A",
    "Отвези кубик 0 в комнату Б, а кубик 1 в комнату А.", "Take cube 1 to room A and cube 0 to room B",
    "Bring both cubes to room B", "Go to the restricted area and bring cube 0 to room A",
]


def run_missions(seeds: int = 4) -> dict:
    """Whole missions: the ACT-R agent of Lab 1 (unchanged, mock planner) with the LQG driving the simulated
    robot, pose noise 2 cm and 0.03 rad."""
    from actr_lab1.core.agent import ActrAgent
    from actr_lab1.core.logs import EventLog
    from actr_lab1.core.planners import MockLLMPlanner
    from .sim_robot import LqgSimRobot
    layout, grid = load_world()
    out = {"runs": 0, "success": 0, "unsafe": 0, "nav_failures": 0, "too_close_to_a_wall": 0}
    for text in MISSIONS:
        for seed in range(seeds):
            robot = LqgSimRobot(layout, grid, seed=seed, record_trace=True, pose_noise=0.02, yaw_noise=0.03)
            agent = ActrAgent(layout, grid, robot, MockLLMPlanner(layout), log=EventLog(console=False))
            res = agent.run(text)
            out["runs"] += 1
            out["success"] += int(res.success)
            out["unsafe"] += int(res.unsafe)
            out["nav_failures"] += robot.nav_failures
            out["too_close_to_a_wall"] += sum(not grid.is_free(x, y, ROBOT_RADIUS) for _t, x, y in robot.trace)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=10, help="runs per route and noise level (default 10)")
    ap.add_argument("--out", default="", help="folder for the results (default: results/lqg of the workspace)")
    args = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "..", "..", "..", "results", "lqg")
    os.makedirs(out, exist_ok=True)
    started = time.time()
    rows = run_all(args.seeds)
    summary = summarize(rows)
    with open(os.path.join(out, "runs.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    table = format_table(summary)
    missions = run_missions()
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Pure pursuit vs LQR vs LQG (simulated robot, not Gazebo)\n\n"
                f"{len(ROUTES)} routes on the map of the world x {args.seeds} seeds x {len(NOISE)} noise levels. "
                "Lateral error = distance of the TRUE robot position to the reference path. "
                "Turn jerk = how fast the turn command changes (smaller = smoother).\n\n" + table + "\n\n"
                "## Whole missions with the agent of Lab 1\n\n"
                f"{len(MISSIONS)} missions x {missions['runs'] // len(MISSIONS)} seeds, the ACT-R agent of Lab 1 "
                "unchanged (mock planner), the LQG driving the simulated robot, pose noise 2 cm and 0.03 rad: "
                f"{missions['success']}/{missions['runs']} missions succeeded, {missions['unsafe']} unsafe, "
                f"{missions['nav_failures']} failed navigate steps, {missions['too_close_to_a_wall']} samples closer "
                "than 10 cm to a wall.\n")
    try:
        from .plots import write_plots
        write_plots(rows, summary, out)
    except ImportError as e:                       # matplotlib is optional
        print(f"(no plots: {e})")
    print(table)
    print(f"\nwhole missions with the agent of Lab 1: {missions['success']}/{missions['runs']} succeeded, "
          f"{missions['unsafe']} unsafe")
    print(f"\n{len(rows)} runs in {time.time() - started:.0f} s -> {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
