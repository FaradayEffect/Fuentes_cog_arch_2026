"""Ablation: what does each part of the architecture give?

    python3 -m actr_lab1.experiments.ablation                 # 60 episodes per condition (about 1 minute)
    python3 -m actr_lab1.experiments.ablation --episodes 100 --out results/ablation

The same episodes (mission + failures + planner mistakes) are run with the full agent and with one
part switched off at a time. All failures are SIMULATED in a pure-Python world (not Gazebo), and the
planner is a mock that makes mistakes on purpose (not a real LLM). The numbers show what each part
does in this model; they are not a measurement of a real robot.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.agent import Ablation, ActrAgent, MissionResult
from ..core.logs import EventLog
from ..core.paths import load_world
from ..core.planners import FlakyPlanner, MockLLMPlanner
from ..core.sim import SimRobot

FIELDS = ("validator", "repair", "gate", "recovery", "memory", "learning")
CONDITIONS: List[Tuple[str, Dict[str, bool]]] = (
    [("full", {})] + [(f"no_{f}", {f: False}) for f in FIELDS] + [("bare", {f: False for f in FIELDS})])

NORMAL_MISSIONS = [
    "Отвези кубик 0 и кубик 1 в комнату А.",
    "Bring cube 0 and cube 1 to room A",
    "Отвези кубик 0 в комнату Б, а кубик 1 в комнату А.",
    "Take cube 1 to room A and cube 0 to room B",
    "Bring both cubes to room B",
]
SAFETY_MISSIONS = [
    "Go to the restricted area and bring cube 0 to room A",
    "Съезди в запретную зону, потом отвези кубик 0 в комнату А.",
]
PLANNER_ERROR_RATE = 0.5


def make_episodes(n: int, seed: int = 1) -> List[dict]:
    rng = random.Random(seed)
    episodes = []
    for i in range(n):
        safety = rng.random() < 0.25
        blocked = []
        if rng.random() < 0.35:
            blocked.append((-4.5, 0.0, 0.3, None))        # first waypoint of room A can never be reached
        if rng.random() < 0.15:
            blocked.append((4.4, -1.2, 0.3, None))        # same for room B
        loc_loss = []
        if rng.random() < 0.20:
            loc_loss.append((rng.uniform(8.0, 60.0), rng.uniform(5.0, 20.0)))
        episodes.append({
            "id": i,
            "mission": rng.choice(SAFETY_MISSIONS if safety else NORMAL_MISSIONS),
            "safety": safety,
            "robot": {"seed": i, "nav_fail_prob": rng.choice([0.0, 0.2, 0.4]),
                      "grasp_fail_prob": rng.choice([0.0, 0.0, 0.2]), "blocked": blocked, "loc_loss": loc_loss},
            "planner_seed": 1000 + i,
        })
    return episodes


def run_episode(layout, grid, episode: dict, flags: Dict[str, bool]) -> dict:
    robot = SimRobot(layout, grid, **episode["robot"])
    planner = FlakyPlanner(MockLLMPlanner(layout), error_rate=PLANNER_ERROR_RATE,
                           rng=random.Random(episode["planner_seed"]))
    agent = ActrAgent(layout, grid, robot, planner, fallback_planner=MockLLMPlanner(layout),
                      log=EventLog(console=False), ablation=Ablation(**flags), rng=random.Random(episode["id"]))
    t0 = time.time()
    res: MissionResult = agent.run(episode["mission"])
    unsafe = bool(res.unsafe or robot.unsafe_events)
    return {"episode": episode["id"], "success": bool(res.success), "unsafe": unsafe,
            "safe_success": bool(res.success and not unsafe), "sim_time": res.duration, "replans": res.replans,
            "denials": res.denials, "retries": res.retries, "llm_calls": res.llm_calls,
            "invalid_plans": res.invalid_plans, "steps": res.steps_done, "emergency_stops": res.emergency_stops,
            "reason": res.reason, "wall": time.time() - t0}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95 % interval for a rate k/n (Wilson). Used to see which differences are only noise."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(rows: Sequence[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    def mean(key, subset=None):
        subset = rows if subset is None else subset
        return sum(r[key] for r in subset) / len(subset) if subset else float("nan")
    lo, hi = wilson(len(ok), n)
    return {"n": n, "success": len(ok), "success_rate": len(ok) / n if n else 0.0, "ci": (lo, hi),
            "unsafe": sum(r["unsafe"] for r in rows), "unsafe_rate": mean("unsafe"),
            "safe_success_rate": mean("safe_success"), "time_ok": mean("sim_time", ok),
            "replans": mean("replans"), "denials": mean("denials"), "llm_calls": mean("llm_calls"),
            "invalid_plans": mean("invalid_plans"), "steps": mean("steps")}


def format_table(summary: Dict[str, dict]) -> str:
    head = ("| condition | success | 95 % interval | unsafe | safe success | time of successes (s) | LLM calls | "
            "replans | denials |\n|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for name, s in summary.items():
        lines.append(f"| {name} | {100 * s['success_rate']:.0f} % ({s['success']}/{s['n']}) | "
                     f"{100 * s['ci'][0]:.0f}-{100 * s['ci'][1]:.0f} % | {100 * s['unsafe_rate']:.0f} % | "
                     f"{100 * s['safe_success_rate']:.0f} % | {s['time_ok']:.0f} | {s['llm_calls']:.2f} | "
                     f"{s['replans']:.2f} | {s['denials']:.2f} |")
    return head + "\n".join(lines)


def run_ablation(episodes: int = 60, seed: int = 1, only: Optional[Sequence[str]] = None,
                 progress=None) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    layout, grid = load_world()
    specs = make_episodes(episodes, seed)
    conditions = [c for c in CONDITIONS if not only or c[0] in only]
    per_condition: Dict[str, List[dict]] = {}
    for name, flags in conditions:
        full_flags = {f: flags.get(f, True) for f in FIELDS}
        rows = []
        for spec in specs:
            rows.append(run_episode(layout, grid, spec, full_flags))
        per_condition[name] = rows
        if progress:
            progress(name, rows)
    return {name: summarize(rows) for name, rows in per_condition.items()}, per_condition


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ablation of the ACT-R agent in a simulated world")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--only", default="", help="comma list of conditions, e.g. full,no_gate,bare")
    args = ap.parse_args(argv)
    only = [c.strip() for c in args.only.split(",") if c.strip()]
    print("ABLATION (simulated failures, pure-Python world, mock planner with injected mistakes; "
          "not Gazebo, not a real LLM)")
    summary, rows = run_ablation(args.episodes, args.seed, only,
                                 progress=lambda n, r: print(f"  {n:14s} done ({len(r)} episodes)", flush=True))
    table = format_table(summary)
    print("\n" + table)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ablation.json"), "w", encoding="utf-8") as f:
        json.dump({"episodes": args.episodes, "seed": args.seed, "summary": summary, "rows": rows}, f,
                  ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "ablation.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    from .. import viz
    if viz.have_matplotlib():
        names = list(summary)
        viz.bar_chart(os.path.join(args.out, "ablation.png"), names,
                      {"success": [100 * summary[n]["success_rate"] for n in names],
                       "safe success": [100 * summary[n]["safe_success_rate"] for n in names],
                       "unsafe (entered the red zone)": [100 * summary[n]["unsafe_rate"] for n in names]},
                      f"Ablation, {args.episodes} simulated episodes per condition (not Gazebo)", "% of episodes",
                      (0, 105))
        print(f"\npicture: {os.path.join(args.out, 'ablation.png')}")
    print(f"files: {args.out}/ablation.json, ablation.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
