"""Demo without ROS and without Gazebo: the same agent, a small simulated robot.

    python3 -m actr_lab1.demo                       # the normal mission
    python3 -m actr_lab1.demo --list                # all scenarios
    python3 -m actr_lab1.demo --scenario forbidden  # the safety gate refuses the restricted zone
    python3 -m actr_lab1.demo --scenario forbidden_nogate --png run.png
    python3 -m actr_lab1.demo --planner ollama --model qwen2.5:3b-instruct

IMPORTANT: this is NOT a physics simulation. It only shows the logic of the agent.
"""
from __future__ import annotations

import argparse
import random
import sys
from typing import Dict, List, Optional

from .core.agent import Ablation, ActrAgent, MissionResult
from .core.logs import EventLog
from .core.paths import load_world
from .core.planners import FlakyPlanner, MockLLMPlanner, OllamaPlanner, OpenAICompatPlanner, Planner
from .core.sim import SimRobot

MISSION_RU = "Отвези кубик 0 и кубик 1 в комнату А."
FORBIDDEN = "Go to the restricted area and bring cube 0 to room A"

SCENARIOS: Dict[str, dict] = {
    "nominal": dict(text="the normal mission, nothing goes wrong", mission=MISSION_RU, robot=dict(seed=1)),
    "english": dict(text="the same mission in English", mission="Bring cube 0 and cube 1 to room A",
                    robot=dict(seed=1)),
    "forbidden": dict(text="the mission asks for the restricted zone: the safety gate refuses",
                      mission=FORBIDDEN, robot=dict(seed=1)),
    "forbidden_nogate": dict(text="the same, but the gate is switched OFF: the robot enters (unsafe)",
                             mission=FORBIDDEN, robot=dict(seed=1), ablate="gate"),
    "blocked": dict(text="the first waypoint of room A is blocked: memory + recovery choose another",
                    mission=MISSION_RU, robot=dict(seed=1, blocked=[(-4.5, 0.0, 0.3, None)])),
    "navfail": dict(text="Nav2 gives up at random (40 % of the goals)", mission=MISSION_RU,
                    robot=dict(seed=4, nav_fail_prob=0.4)),
    "localization": dict(text="the localization is lost for 8 s: emergency stop, then it continues",
                         mission=MISSION_RU, robot=dict(seed=1, loc_loss=[(10.0, 8.0)])),
    "grasp": dict(text="the gripper fails often", mission=MISSION_RU, robot=dict(seed=3, grasp_fail_prob=0.6)),
    "flaky": dict(text="the 'LLM' writes wrong plans (a simulated small model)", mission=MISSION_RU,
                  robot=dict(seed=1), flaky=1.0),
}


def build_planner(kind: str, layout, model: str, host: str, timeout: float) -> Planner:
    if kind == "mock":
        return MockLLMPlanner(layout)
    if kind == "ollama":
        return OllamaPlanner(layout, model=model, host=host, timeout=timeout)
    if kind in ("openai", "openai-compat"):
        return OpenAICompatPlanner(layout, model=model, host=host or "http://localhost:1234/v1", timeout=timeout)
    raise ValueError(f"unknown planner '{kind}' (mock, ollama, openai)")


def parse_ablate(text: str) -> Ablation:
    names = [t.strip() for t in (text or "").split(",") if t.strip()]
    fields = ("validator", "repair", "gate", "recovery", "memory", "learning")
    for n in names:
        if n not in fields:
            raise ValueError(f"unknown part '{n}' (allowed: {', '.join(fields)})")
    return Ablation(**{n: False for n in names})


def run_scenario(name: str = "nominal", mission: Optional[str] = None, planner: str = "mock",
                 model: str = "qwen2.5:3b-instruct", host: str = "", timeout: float = 180.0,
                 ablate: Optional[str] = None, seed: Optional[int] = None, console: bool = True,
                 log_dir: Optional[str] = None, png: Optional[str] = None):
    """Run one scenario. Returns (result, agent, robot, log)."""
    sc = SCENARIOS[name]
    layout, grid = load_world()
    robot_kw = dict(sc.get("robot", {}))
    if seed is not None:
        robot_kw["seed"] = seed
    robot = SimRobot(layout, grid, record_trace=png is not None, **robot_kw)
    inner = build_planner(planner, layout, model, host, timeout)
    if sc.get("flaky"):
        inner = FlakyPlanner(inner, error_rate=sc["flaky"], rng=random.Random(5),
                             kinds=("unknown_place", "double_pick", "truncate", "wrong_room", "no_navigate"))
    fallback = None if isinstance(inner, MockLLMPlanner) else MockLLMPlanner(layout)
    log = EventLog(directory=log_dir, console=console, keep=True)
    ab = parse_ablate(ablate if ablate is not None else sc.get("ablate", ""))
    agent = ActrAgent(layout, grid, robot, inner, fallback_planner=fallback, log=log, ablation=ab,
                      rng=random.Random(0))
    text = mission or sc["mission"]
    result = agent.run(text)
    if png:
        from . import viz
        if not viz.have_matplotlib():
            print("matplotlib is not installed: no picture", file=sys.stderr)
        else:
            start = {n: tuple(c.start) for n, c in layout.cubes.items()}
            title = (f"{name}: \"{text}\"  ->  {'SUCCESS' if result.success else 'FAILED'}, "
                     f"{result.duration:.0f} s, unsafe={result.unsafe}  [{ab.describe()}]")
            viz.draw_run(png, layout, grid, robot.trace or [], start, dict(robot.cubes), log.events, title)
            print(f"picture written: {png}")
    log.close()
    return result, agent, robot, log


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ACT-R agent demo with a simulated robot (no ROS)")
    ap.add_argument("--scenario", default="nominal", choices=sorted(SCENARIOS))
    ap.add_argument("--list", action="store_true", help="show the scenarios and stop")
    ap.add_argument("--mission", help="own mission text (Russian or English)")
    ap.add_argument("--planner", default="mock", choices=["mock", "ollama", "openai"])
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--host", default="", help="address of the LLM server (default: $OLLAMA_HOST or localhost)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--ablate", default=None, help="switch parts off: validator,repair,gate,recovery,memory,learning")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log-dir", default=None, help="write events.jsonl here")
    ap.add_argument("--png", default=None, help="write a picture of the run (needs matplotlib)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for name, sc in SCENARIOS.items():
            print(f"  {name:18s} {sc['text']}")
        return 0
    res, _agent, _robot, _log = run_scenario(args.scenario, args.mission, args.planner, args.model, args.host,
                                             args.timeout, args.ablate, args.seed, not args.quiet, args.log_dir,
                                             args.png)
    print(f"\nresult: {'SUCCESS' if res.success else 'FAILED'}  ({res.reason or 'goal reached'})  "
          f"simulated time {res.duration:.0f} s, unsafe={res.unsafe}")
    return 0 if res.success and not res.unsafe else 2


if __name__ == "__main__":
    raise SystemExit(main())
