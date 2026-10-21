"""The ACT-R-inspired agent.

How it maps to ACT-R
--------------------
buffers        goal, plan, percept, retrieval, proposal, action, motor, evaluation  (BufferSystem)
modules        perception, memory, llm, validator, gate, executor, evaluator, procedural
chunks         every value in a buffer (Chunk)
productions    the IF-THEN rules below (make_productions); one fires per cycle
utility        decides between rules that match at the same time; learned from rewards
declarative    DeclarativeMemory: map facts are in Layout, experience is stored as `episode` chunks

The LLM is just one module. It can only write into the `proposal` buffer. A plan becomes
the agent's plan only through the production ACCEPT-PLAN, after the validator said yes.
An action reaches the robot only after the gate wrote `approved=True` into it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .buffers import AccessDenied, BufferSystem
from .chunks import Chunk
from .gate import ActionGate
from .logs import EventLog
from .memory import DeclarativeMemory
from .nlp import expected_deliveries, parse_mission
from .plan import PlanStep, PlanValidator, Validation
from .planners import Planner, state_from_percept
from .productions import Production, ProductionSystem, update_utility
from .robot_api import RobotAPI
from .world import (Layout, OccupancyGrid, Pose2D, approach_candidates, dist,
                    find_drop_point)

LOC_REASONS = ("denied:localization_lost", "emergency:localization_lost")


@dataclass
class AgentConfig:
    cycle_dt: float = 0.1            # seconds of robot time per production cycle
    nav_timeout: float = 150.0       # cancel a navigation goal after this time
    mission_timeout: float = 1500.0
    max_cycles: int = 200000
    max_plan_attempts: int = 3       # LLM answers per planning round (before the fallback planner)
    max_retries: int = 2             # retries of a step without any progress
    max_replans: int = 2
    max_repairs: int = 12
    pick_radius: float = 1.10        # the virtual gripper reaches this far
    near_radius: float = 0.90        # "next to the cube" for the repair rules (less than the reach of the gripper)
    loc_grace: float = 2.0           # seconds without localization before an emergency stop
    loc_wait_timeout: float = 60.0
    use_fallback: bool = True        # use the Mock planner if the LLM never gives a valid plan
    parse_goal: bool = True          # find "cube X -> room Y" in the mission text to check the result
    utility_alpha: float = 0.2
    noise: float = 0.0               # utility noise (0 = deterministic)
    idle_limit: int = 300            # cycles with no matching rule before we stop
    retry_wait: float = 1.0
    max_actions: int = 200
    show_tokens: bool = False        # print in the log how many tokens the LLM used (they are always saved in the file)


@dataclass
class Ablation:
    """Switch parts of the architecture off, to see what each part gives."""
    validator: bool = True     # check plans before they are used
    repair: bool = True        # repair rules (insert a missing navigate)
    gate: bool = True          # safety gate and emergency stop
    recovery: bool = True      # retry / other waypoint / replan / skip
    memory: bool = True        # use episodes from memory to choose waypoints
    learning: bool = True      # learn rule utilities from rewards

    def describe(self) -> str:
        off = [k for k, v in self.__dict__.items() if not v]
        return "full" if not off else "no_" + "+".join(off)


@dataclass
class MissionResult:
    success: bool
    status: str
    reason: str
    duration: float
    cycles: int
    unsafe: bool
    steps_done: int = 0
    replans: int = 0
    repairs: int = 0
    denials: int = 0
    retries: int = 0
    emergency_stops: int = 0
    llm_calls: int = 0
    invalid_plans: int = 0
    fallback_used: bool = False
    score: float = 0.0
    unmet: Tuple[str, ...] = ()
    final_cubes: Dict[str, str] = field(default_factory=dict)


class ActrAgent:
    def __init__(self, layout: Layout, grid: Optional[OccupancyGrid], robot: RobotAPI, planner: Planner,
                 fallback_planner: Optional[Planner] = None, config: Optional[AgentConfig] = None,
                 ablation: Optional[Ablation] = None, log: Optional[EventLog] = None,
                 rng: Optional[random.Random] = None):
        self.layout = layout
        self.grid = grid
        self.robot = robot
        self.planner = planner
        self.fallback = fallback_planner
        self.cfg = config or AgentConfig()
        self.ab = ablation or Ablation()
        self.log = log or EventLog(console=False, keep=True)
        self.log.clock = self.robot.now          # log lines show robot (simulation) time
        self.rng = rng or random.Random(0)

        self._offset = 0.0              # keeps the agent clock running over several missions
        self._last_clock = 0.0
        self.buf = BufferSystem(clock=self.clock)
        self.mem = DeclarativeMemory(self.clock, rng=self.rng)
        self.validator = PlanValidator(layout)
        self.gate = ActionGate(layout, grid, pick_radius=self.cfg.pick_radius, max_actions=self.cfg.max_actions)
        self.productions = self.make_productions()
        self.ps = ProductionSystem(self.productions, noise=self.cfg.noise, rng=self.rng)
        self._reset_counters()

    # ---------------------------------------------------------------- bookkeeping
    def clock(self) -> float:
        self._last_clock = self.robot.now() + self._offset
        return self._last_clock

    def _reset_counters(self):
        self.llm_calls = 0
        self.invalid_plans = 0
        self.fallback_used = False
        self.replans = 0
        self.repairs = 0
        self.retries = 0
        self.emergency_stops = 0
        self.steps_done = 0
        self.streak = 0                         # failed attempts in a row without progress
        self.claims: Dict[str, str] = {}        # cube -> room the plan says it delivered
        self.pending_credit: Optional[Production] = None
        self.hold_until = -1.0
        self.loc_lost_since: Optional[float] = None
        self.unsafe = False
        self.denials = 0
        self._approach_cache: Dict[tuple, List[Pose2D]] = {}

    # ------------------------------------------------------------ small accessors
    def status(self) -> str:
        return self.buf.slot("goal", "status", "")

    def set_goal(self, **changes) -> None:
        goal = self.buf.get("goal")
        self.buf.write("goal", goal.with_slots(**changes), by="procedural")

    def plan_chunk(self) -> Optional[Chunk]:
        return self.buf.get("plan")

    def step(self) -> Optional[PlanStep]:
        plan = self.buf.get("plan")
        if plan is None:
            return None
        i = plan.get("index", 0)
        steps = plan.get("steps", ())
        return steps[i] if i < len(steps) else None

    def set_plan(self, **changes) -> None:
        plan = self.buf.get("plan")
        self.buf.write("plan", plan.with_slots(**changes), by="procedural")

    def percept(self) -> Optional[Chunk]:
        return self.buf.get("percept")

    def motor_state(self) -> str:
        return self.buf.slot("motor", "state", "free")

    def say(self, kind: str, msg: str, **data) -> None:
        self.log.event(kind, msg, **data)

    # ------------------------------------------------------------------ perception
    def update_percept(self) -> None:
        """Module `perception`: turn the robot's raw data into a percept chunk."""
        obs = self.robot.observe()
        x, y, yaw = obs.pose.x, obs.pose.y, obs.pose.yaw
        cubes = tuple((n, cx, cy, self.layout.room_at(cx, cy)) for n, (cx, cy) in sorted(obs.cubes.items()))
        near = tuple(n for n, cx, cy, _r in cubes if dist((x, y), (cx, cy)) <= self.cfg.near_radius)
        in_forbidden = self.layout.in_forbidden(x, y)
        if in_forbidden:
            self.unsafe = True
        now = self.clock()
        if obs.localized:
            self.loc_lost_since = None
        elif self.loc_lost_since is None:
            self.loc_lost_since = now
        chunk = Chunk("percept", {
            "t": obs.t, "x": x, "y": y, "yaw": yaw, "room": self.layout.room_at(x, y),
            "holding": obs.holding, "cubes": cubes, "near": near, "in_forbidden": in_forbidden,
            "localized": obs.localized, "cov": obs.cov, "nav": dict(obs.nav),
        })
        self.buf.write("percept", chunk, by="perception")

    # ---------------------------------------------------------------- grounding
    def n_variants(self, target: str) -> int:
        if target in self.layout.places:
            return max(1, self.layout.n_waypoints(target))
        if target in self.layout.cubes:
            return max(1, min(6, len(self._cube_candidates(target))))
        return 1

    def _cube_xy(self, cube: str) -> Optional[Tuple[float, float]]:
        p = self.percept()
        if p is None:
            return None
        for n, x, y, _r in p.get("cubes", ()):
            if n == cube:
                return (x, y)
        return None

    def _cube_candidates(self, cube: str) -> List[Pose2D]:
        """Poses next to the cube, best first. The list is computed once for each cube position,
        so 'variant 1' keeps meaning the same pose while we retry."""
        xy = self._cube_xy(cube)
        p = self.percept()
        if xy is None or p is None or self.grid is None:
            return []
        key = (cube, round(xy[0], 2), round(xy[1], 2))
        if key not in self._approach_cache:
            self._approach_cache[key] = approach_candidates(self.grid, self.layout, xy, (p.get("x"), p.get("y")),
                radii=getattr(self.robot, "approach_radii", (0.55, 0.65)), margin=getattr(self.robot, "approach_margin", 0.08))
        return self._approach_cache[key]

    def resolve_pose(self, target: str, variant: int) -> Optional[Pose2D]:
        if target in self.layout.places:
            return self.layout.waypoint(target, variant)
        if target in self.layout.cubes:
            cands = self._cube_candidates(target)
            return cands[variant % len(cands)] if cands else None
        return None

    def failed_variants(self, target: str) -> set:
        found = self.mem.retrieve_all({"kind": "episode", "cmd": "navigate", "target": target, "result": "failure"})
        if found:
            self.buf.write("retrieval", found[0], by="memory")
        return {c.get("variant") for c in found}

    def choose_variant(self, target: str, step_variant: Optional[int]) -> int:
        """Which waypoint to use. Memory helps: use the one that worked, avoid the ones that failed."""
        if step_variant is not None:
            return step_variant
        if not self.ab.memory:
            return 0
        win = self.mem.retrieve({"kind": "episode", "cmd": "navigate", "target": target, "result": "success"})
        if win is not None:
            self.buf.write("retrieval", win, by="memory")
            return int(win.get("variant", 0) or 0)
        failed = self.failed_variants(target)
        for v in range(self.n_variants(target)):
            if v not in failed:
                return v
        return 0

    # -------------------------------------------------------------- goal checking
    def unmet_goals(self) -> List[str]:
        p = self.percept()
        wanted = list(self.buf.slot("goal", "expected", ()))
        if not wanted:
            wanted = list(self.claims.items())      # no known goal: check what the plan claims it did
        if p is None:
            return [f"{c}->{r}" for c, r in wanted]
        where = {n: r for n, _x, _y, r in p.get("cubes", ())}
        holding = p.get("holding")
        return [f"{c}->{r}" for c, r in wanted if where.get(c) != r or holding == c]

    # ---------------------------------------------------------------- executor
    def executor_start(self, action: Chunk) -> None:
        """Module `executor`: the only code that gives commands to the robot."""
        action = self.buf.read("action", "executor")
        if action.get("approved") is not True:
            raise AccessDenied("executor: the action was not approved by the gate")
        cmd, target, idx = action.get("cmd"), action.get("target"), action.get("step_index")
        now = self.clock()
        base = {"cmd": cmd, "target": target, "step_index": idx, "started": now, "variant": action.get("variant")}
        if cmd == "navigate":
            pose = action.get("pose")
            if pose is None:
                self.buf.write("motor", Chunk("motor", {**base, "state": "done", "ok": False, "reason": "no_pose"}), by="executor")
                return
            self.robot.start_navigate(Pose2D(*pose))
            self.buf.write("motor", Chunk("motor", {**base, "state": "busy"}), by="executor")
        elif cmd == "pick":
            res = self.robot.pick(target)
            self.buf.write("motor", Chunk("motor", {**base, "state": "done", "ok": res.ok, "reason": res.reason,
                                                    "cube": target}), by="executor")
        elif cmd == "place":
            cube = self.percept().get("holding")
            drop = action.get("drop")
            if drop is None or cube is None:
                res_ok, res_reason = False, "no_drop_point" if drop is None else "not_holding"
            else:
                res = self.robot.place(cube, tuple(drop))
                res_ok, res_reason = res.ok, res.reason
            self.buf.write("motor", Chunk("motor", {**base, "state": "done", "ok": res_ok, "reason": res_reason,
                                                    "cube": cube}), by="executor")
        else:
            self.buf.write("motor", Chunk("motor", {**base, "state": "done", "ok": False, "reason": "unsupported"}), by="executor")

    def finish_motor(self, ok: bool, reason: str = "") -> None:
        motor = self.buf.read("motor", "executor")
        self.buf.write("motor", motor.with_slots(state="done", ok=ok, reason=reason), by="executor")

    def reward(self, value: float, ok: bool, cmd: str, reason: str) -> None:
        self.buf.write("evaluation", Chunk("evaluation", {"reward": value, "ok": ok, "cmd": cmd, "reason": reason}),
                       by="evaluator")

    def credit(self, value: float) -> None:
        if self.pending_credit is not None and self.ab.learning:
            old = self.pending_credit.utility
            new = update_utility(self.pending_credit, value, self.cfg.utility_alpha)
            self.say("learn", f"utility {self.pending_credit.name}: {old:.2f} -> {new:.2f} (reward {value:+.1f})",
                     production=self.pending_credit.name, old=old, new=new, reward=value)
        self.pending_credit = None

    def remember(self, motor: Chunk, result: str) -> None:
        self.mem.add(Chunk("episode", {"cmd": motor.get("cmd"), "target": motor.get("target"),
                                       "variant": motor.get("variant"), "result": result,
                                       "reason": motor.get("reason", "")}))

    # ============================================ PRODUCTIONS ============================================
    # conditions (LHS) -------------------------------------------------------
    def _proposal_state(self) -> Tuple[Optional[Chunk], Optional[bool]]:
        p = self.buf.get("proposal")
        return p, (None if p is None else p.get("valid"))

    def c_ask_llm(self) -> bool:
        return self.status() == "new" and self.buf.get("proposal") is None

    def c_validate(self) -> bool:
        p, valid = self._proposal_state()
        return p is not None and valid is None

    def c_accept(self) -> bool:
        p, valid = self._proposal_state()
        return self.status() == "planning" and valid is True

    def _rejected(self) -> Tuple[bool, int]:
        p, valid = self._proposal_state()
        attempts = self.buf.slot("goal", "attempts", 0)
        return (self.status() == "planning" and valid is False), attempts

    def c_reject_retry(self) -> bool:
        rejected, attempts = self._rejected()
        return rejected and attempts + 1 < self.cfg.max_plan_attempts

    def c_reject_fallback(self) -> bool:
        rejected, attempts = self._rejected()
        return (rejected and attempts + 1 >= self.cfg.max_plan_attempts and self.fallback is not None
                and self.cfg.use_fallback and not self.buf.slot("goal", "use_fallback", False))

    def c_reject_giveup(self) -> bool:
        rejected, attempts = self._rejected()
        if not rejected or attempts + 1 < self.cfg.max_plan_attempts:
            return False
        return not (self.fallback is not None and self.cfg.use_fallback
                    and not self.buf.slot("goal", "use_fallback", False))

    def _executing_idle(self) -> bool:
        return self.status() == "executing" and self.buf.get("action") is None and self.motor_state() == "free"

    def c_verify(self) -> bool:
        plan = self.plan_chunk()
        return (self._executing_idle() and plan is not None
                and plan.get("index", 0) >= len(plan.get("steps", ())))

    def c_ready(self) -> bool:
        return (self._executing_idle() and self.step() is not None and self.clock() >= self.hold_until)

    def c_wait(self) -> bool:
        return self._executing_idle() and self.step() is not None and self.clock() < self.hold_until

    def c_repair_pick(self) -> bool:
        if not (self.c_ready() and self.repairs < self.cfg.max_repairs):
            return False
        s, p = self.step(), self.percept()
        return (s.action == "pick" and s.target in self.layout.cubes and s.target not in p.get("near", ())
                and not p.get("holding"))

    def c_repair_place(self) -> bool:
        if not (self.c_ready() and self.repairs < self.cfg.max_repairs):
            return False
        s, p = self.step(), self.percept()
        return s.action == "place" and s.target in self.layout.places and p.get("room", "") != s.target

    def c_gate(self) -> bool:
        a = self.buf.get("action")
        return a is not None and a.get("approved") is None

    def c_execute(self) -> bool:
        a = self.buf.get("action")
        return a is not None and a.get("approved") is True and self.motor_state() == "free"

    def c_denied(self) -> bool:
        a = self.buf.get("action")
        return a is not None and a.get("approved") is False

    def c_monitor(self) -> bool:
        return self.motor_state() == "busy"

    def _danger(self) -> str:
        p = self.percept()
        if p is None:
            return ""
        if p.get("in_forbidden"):
            return "forbidden_zone"
        if self.loc_lost_since is not None and self.clock() - self.loc_lost_since >= self.cfg.loc_grace:
            return "localization_lost"
        return ""

    def c_emergency(self) -> bool:
        return self.motor_state() == "busy" and bool(self._danger())

    def c_step_success(self) -> bool:
        m = self.buf.get("motor")
        return m is not None and m.get("state") == "done" and m.get("ok") is True

    def c_step_failure(self) -> bool:
        m = self.buf.get("motor")
        return m is not None and m.get("state") == "done" and m.get("ok") is False

    # recovery conditions
    def _recovering(self) -> bool:
        return self.status() == "recovering"

    def _fail(self) -> dict:
        return dict(self.buf.slot("goal", "fail_info", {}) or {})

    def c_rec_wait_loc(self) -> bool:
        return self._recovering() and self._fail().get("reason") in LOC_REASONS

    def _retryable(self) -> bool:
        reason = self._fail().get("reason", "")
        return not (reason.startswith("denied:") or reason == "emergency:forbidden_zone"
                    or reason in ("no_pose", "unsupported"))

    def c_rec_retry(self) -> bool:
        return (self._recovering() and self.ab.recovery and self._retryable()
                and not self.c_rec_wait_loc() and self.streak <= self.cfg.max_retries)

    def c_rec_alt(self) -> bool:
        f = self._fail()
        return (self._recovering() and self.ab.recovery and f.get("cmd") == "navigate"
                and f.get("reason") != "emergency:forbidden_zone"
                and not f.get("reason", "").startswith("denied:unknown")
                and self.n_variants(f.get("target")) > 1 and self.streak <= self.cfg.max_retries + 1
                and not self.c_rec_wait_loc())

    def c_rec_replan(self) -> bool:
        return (self._recovering() and self.ab.recovery and self.replans < self.cfg.max_replans
                and not self.c_rec_wait_loc())

    def c_rec_skip(self) -> bool:
        reason = self._fail().get("reason", "")
        return (self._recovering() and self.ab.recovery
                and (reason.startswith("denied:") or reason == "emergency:forbidden_zone"))

    def c_rec_giveup(self) -> bool:
        return self._recovering()

    # actions (RHS) ----------------------------------------------------------------
    def a_ask_llm(self) -> None:
        goal = self.buf.get("goal")
        planner = self.fallback if (goal.get("use_fallback") and self.fallback is not None) else self.planner
        self.set_goal(status="planning")
        try:
            proposal = self._clean_proposal(planner.propose(self.buf), planner.name)
        except Exception as e:              # a planner bug must never crash the robot
            proposal = Chunk("proposal", {"source": planner.name, "raw": None, "text": "",
                                          "error": f"planner crashed: {e}", "latency": 0.0, "valid": None})
        self.llm_calls += 1
        self.buf.write("proposal", proposal, by="llm")
        last = getattr(planner, "last_proposal", None)
        t_in, t_out = getattr(last, "tokens_in", None), getattr(last, "tokens_out", None)   # counted by Ollama
        tok = f", tokens: {t_in} in + {t_out} out" if self.cfg.show_tokens and None not in (t_in, t_out) else ""
        self.say("llm", f"{proposal.get('source')} answered in {proposal.get('latency', 0):.1f} s{tok}"
                        + (f" (error: {proposal.get('error')})" if proposal.get("error") else ""),
                 attempt=goal.get("attempts", 0), feedback=goal.get("feedback", ()),
                 state=state_from_percept(self.percept()), text=proposal.get("text", ""),
                 prompt=getattr(last, "prompt", ""), tokens_in=t_in, tokens_out=t_out)

    @staticmethod
    def _clean_proposal(chunk: Chunk, name: str) -> Chunk:
        """The LLM module may fill only these slots. Anything else a planner puts into its chunk (a ready-made
        `valid` flag, `steps`, ...) is dropped, so a planner cannot skip the validator."""
        return Chunk("proposal", {"source": chunk.get("source", name), "raw": chunk.get("raw"),
                                  "text": chunk.get("text", ""), "error": chunk.get("error", ""),
                                  "latency": chunk.get("latency", 0.0), "valid": None})

    def _lenient_steps(self, raw: Any) -> Validation:
        """Used only when the validator is switched off: believe the LLM."""
        items = raw.get("steps") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return Validation(False, (), ("no list of steps",), ())
        steps = []
        for it in items:
            if isinstance(it, dict) and isinstance(it.get("action"), str) and isinstance(it.get("target"), str):
                steps.append(PlanStep(it["action"], it["target"]))
        return Validation(bool(steps), tuple(steps), () if steps else ("no usable steps",), ())

    def a_validate(self) -> None:
        prop = self.buf.read("proposal", "validator")
        state = state_from_percept(self.buf.read("percept", "validator"))
        expected = self.buf.read("goal", "validator").get("expected", ())
        if prop.get("raw") is None:
            v = Validation(False, (), (prop.get("error") or "empty answer",), ())
        elif self.ab.validator:
            v = self.validator.validate(prop.get("raw"), state, expected)
        else:
            v = self._lenient_steps(prop.get("raw"))
        if not v.valid:
            self.invalid_plans += 1
        self.buf.write("proposal", prop.with_slots(valid=v.valid, steps=v.steps, errors=v.errors,
                                                   warnings=v.warnings), by="validator")
        self.say("plan", ("valid" if v.valid else "REJECTED: " + "; ".join(v.errors[:3]))
                 + (f" (warnings: {'; '.join(v.warnings[:2])})" if v.warnings else ""),
                 valid=v.valid, errors=v.errors, warnings=v.warnings,
                 steps=[s.label() for s in v.steps])

    def a_accept(self) -> None:
        prop = self.buf.get("proposal")
        steps = tuple(prop.get("steps"))
        self.buf.write("plan", Chunk("plan", {"steps": steps, "index": 0, "repairs": 0}), by="procedural")
        self.buf.clear("proposal", by="procedural")
        self.set_goal(status="executing", feedback=(), attempts=0)
        self.say("plan", f"accepted ({len(steps)} steps): " + " ".join(s.label() for s in steps))

    def a_reject_retry(self) -> None:
        prop = self.buf.get("proposal")
        attempts = self.buf.slot("goal", "attempts", 0)
        self.set_goal(status="new", attempts=attempts + 1, feedback=tuple(prop.get("errors", ())))
        self.buf.clear("proposal", by="procedural")
        self.say("plan", f"asking the planner again (attempt {attempts + 2}) with the error messages")

    def a_reject_fallback(self) -> None:
        self.fallback_used = True
        self.set_goal(status="new", attempts=0, feedback=(), use_fallback=True)
        self.buf.clear("proposal", by="procedural")
        self.say("plan", f"no valid plan from {self.planner.name}: switching to the fallback planner "
                         f"({self.fallback.name})")

    def a_reject_giveup(self) -> None:
        prop = self.buf.get("proposal")
        errors = prop.get("errors", ())
        self.set_goal(status="failed", reason="no valid plan: " + (errors[0] if errors else "unknown"))
        self.buf.clear("proposal", by="procedural")

    def a_verify(self) -> None:
        unmet = self.unmet_goals()
        if not unmet:
            self.set_goal(status="done")
            self.say("verify", "the world matches the goal: mission done")
            return
        if self.ab.recovery and self.replans < self.cfg.max_replans:
            self.replans += 1
            self.set_goal(status="new", attempts=0, feedback=("after your plan these goals are still not reached: "
                                                               + ", ".join(unmet),))
            self.buf.clear("plan", by="procedural")
            self.say("verify", f"goal not reached ({', '.join(unmet)}): asking for a new plan "
                               f"(replan {self.replans})")
        else:
            self.set_goal(status="failed", reason="goal not reached: " + ", ".join(unmet))
            self.say("verify", f"goal not reached: {', '.join(unmet)}")

    def _insert_step(self, step: PlanStep, why: str) -> None:
        plan = self.plan_chunk()
        i = plan.get("index", 0)
        steps = list(plan.get("steps"))
        steps.insert(i, step)
        self.set_plan(steps=tuple(steps))
        self.repairs += 1
        self.say("repair", f"{why}: inserted {step.label()} before {steps[i + 1].label()}")

    def a_repair_pick(self) -> None:
        s = self.step()
        self._insert_step(PlanStep("navigate", s.target), "the robot is not next to the cube")

    def a_repair_place(self) -> None:
        s = self.step()
        self._insert_step(PlanStep("navigate", s.target), "the robot is not in the target room")

    def a_wait(self) -> None:
        pass

    def a_next_action(self) -> None:
        s, p = self.step(), self.percept()
        idx = self.plan_chunk().get("index", 0)
        pose, drop, variant = None, None, 0
        if s.action == "navigate":
            variant = self.choose_variant(s.target, s.variant)
            pose = self.resolve_pose(s.target, variant)
        elif s.action == "place":
            holding, ok = p.get("holding"), getattr(self.robot, "drop_ok", None)      # the forklift can say no
            others = [(x, y) for n, x, y, _r in p.get("cubes", ()) if n != holding]
            drop = find_drop_point(self.grid, self.layout, Pose2D(p.get("x"), p.get("y"), p.get("yaw")), s.target, others,
                                   reach=getattr(self.robot, "drop_reach", 0.55), accept=ok) if self.grid is not None else None
        chunk = Chunk("action", {"cmd": s.action, "target": s.target,
                                 "pose": None if pose is None else pose.as_tuple(), "drop": drop,
                                 "step_index": idx, "variant": variant, "approved": None, "rule": "", "reason": ""})
        self.buf.write("action", chunk, by="procedural")

    def a_gate(self) -> None:
        action = self.buf.read("action", "gate")
        percept = self.buf.read("percept", "gate")
        if self.ab.gate:
            d = self.gate.check(action, percept)
        else:
            from .gate import GateDecision
            d = GateDecision(True, "gate_disabled")
        self.buf.write("action", action.with_slots(approved=d.approved, rule=d.rule, reason=d.reason), by="gate")
        pose = action.get("pose")
        where = "" if pose is None else f" -> ({pose[0]:.2f}, {pose[1]:.2f})"
        # the log keeps the state the decision was made in and the action that was chosen
        state = {"x": round(percept.get("x", 0.0), 2), "y": round(percept.get("y", 0.0), 2),
                 "room": percept.get("room", ""), "holding": percept.get("holding"),
                 "localized": percept.get("localized", True)}
        self.say("gate" if not d.approved else "action",
                 (f"{action.get('cmd')}({action.get('target')}){where}  gate: ok" if d.approved
                  else f"DENIED {action.get('cmd')}({action.get('target')}) by rule '{d.rule}': {d.reason}"),
                 approved=d.approved, rule=d.rule, cmd=action.get("cmd"), target=action.get("target"),
                 pose=pose, step_index=action.get("step_index"), state=state)

    def a_execute(self) -> None:
        self.executor_start(self.buf.get("action"))

    def a_monitor(self) -> None:
        motor = self.buf.read("motor", "executor")
        st = self.robot.poll_motion()
        if st.state == "running":
            if self.clock() - motor.get("started", 0.0) > self.cfg.nav_timeout:
                self.robot.cancel_motion()
                self.finish_motor(False, "timeout")
            return
        if st.state == "succeeded":
            self.finish_motor(True)
        elif st.state in ("failed", "canceled"):
            self.finish_motor(False, st.reason or st.state)
        else:
            self.finish_motor(False, "lost_motion")

    def a_emergency(self) -> None:
        why = self._danger()
        self.robot.stop()
        self.emergency_stops += 1
        self.finish_motor(False, f"emergency:{why}")
        self.say("emergency", f"EMERGENCY STOP: {why}", reason=why)

    def _clear_action_and_motor(self) -> None:
        self.buf.clear("action", by="procedural")
        self.buf.write("motor", Chunk("motor", {"state": "free"}), by="procedural")

    def a_step_success(self) -> None:
        motor = self.buf.get("motor")
        cmd, target = motor.get("cmd"), motor.get("target")
        self.say("result", f"{cmd}({target}): success")
        self.reward(1.0, True, cmd, "")
        if self.ab.memory:
            self.remember(motor, "success")
        self.credit(+1.0)
        self.streak = 0
        if cmd == "place" and motor.get("cube"):
            self.claims[motor.get("cube")] = target
        self.steps_done += 1
        plan = self.plan_chunk()
        self.set_plan(index=plan.get("index", 0) + 1)
        self._clear_action_and_motor()

    def a_step_failure(self) -> None:
        motor = self.buf.get("motor")
        cmd, target, reason = motor.get("cmd"), motor.get("target"), motor.get("reason", "")
        self.say("result", f"{cmd}({target}): FAILED ({reason})")
        self.reward(-1.0, False, cmd, reason)
        if self.ab.memory:
            self.remember(motor, "failure")
        self.credit(-1.0)
        self.streak += 1
        self.set_goal(status="recovering", fail_info={"cmd": cmd, "target": target, "reason": reason,
                                                      "variant": motor.get("variant"), "t": self.clock()})
        self._clear_action_and_motor()

    def a_denied(self) -> None:
        action = self.buf.get("action")
        reason = "denied:" + str(action.get("rule"))
        self.reward(-0.2, False, action.get("cmd"), reason)
        self.denials += 1
        self.credit(-1.0)
        self.streak += 1
        self.set_goal(status="recovering", fail_info={"cmd": action.get("cmd"), "target": action.get("target"),
                                                      "reason": reason, "variant": action.get("variant"),
                                                      "t": self.clock()})
        self._clear_action_and_motor()

    # recovery actions
    def _recovery_note(self, name: str, text: str) -> None:
        self.say("recover", f"{name}: {text}", production=name)

    def a_rec_wait_loc(self) -> None:
        f = self._fail()
        if self.percept().get("localized", True):
            self.set_goal(status="executing", fail_info={})
            self._recovery_note("RECOVER-WAIT-LOCALIZATION", "localization is back, continuing")
        elif self.clock() - f.get("t", 0.0) > self.cfg.loc_wait_timeout:
            self.set_goal(status="failed", reason="localization was not restored")

    def a_rec_retry(self) -> None:
        self.retries += 1
        self.pending_credit = self.ps.by_name("RECOVER-RETRY")
        self.hold_until = self.clock() + self.cfg.retry_wait
        self.set_goal(status="executing")
        f = self._fail()
        self._recovery_note("RECOVER-RETRY", f"trying {f.get('cmd')}({f.get('target')}) again")

    def a_rec_alt(self) -> None:
        f = self._fail()
        target, cur = f.get("target"), f.get("variant") or 0
        n = self.n_variants(target)
        failed = self.failed_variants(target) if self.ab.memory else {cur}
        choice = (cur + 1) % n
        for k in range(1, n):
            v = (cur + k) % n
            if v not in failed:
                choice = v
                break
        plan = self.plan_chunk()
        i = plan.get("index", 0)
        steps = list(plan.get("steps"))
        steps[i] = PlanStep(steps[i].action, steps[i].target, choice)
        self.set_plan(steps=tuple(steps))
        self.pending_credit = self.ps.by_name("RECOVER-ALT-WAYPOINT")
        self.hold_until = self.clock() + self.cfg.retry_wait
        self.set_goal(status="executing")
        self._recovery_note("RECOVER-ALT-WAYPOINT", f"{target}: other waypoint (variant {cur} -> {choice})")

    def a_rec_replan(self) -> None:
        f = self._fail()
        self.replans += 1
        self.pending_credit = self.ps.by_name("RECOVER-REPLAN")
        text = f"the step {f.get('cmd')}({f.get('target')}) failed: {f.get('reason')}. Make a new plan from the current state."
        self.set_goal(status="new", attempts=0, feedback=(text,), fail_info={})
        self.buf.clear("plan", by="procedural")
        self.buf.clear("proposal", by="procedural")
        self._recovery_note("RECOVER-REPLAN", f"asking the planner for a new plan (replan {self.replans})")

    def a_rec_skip(self) -> None:
        f = self._fail()
        plan = self.plan_chunk()
        self.set_plan(index=plan.get("index", 0) + 1)
        self.streak = 0
        self.set_goal(status="executing", fail_info={})
        self._recovery_note("RECOVER-SKIP", f"skipping {f.get('cmd')}({f.get('target')}): {f.get('reason')}")

    def a_rec_giveup(self) -> None:
        f = self._fail()
        self.set_goal(status="failed", reason=f"{f.get('cmd')}({f.get('target')}) failed: {f.get('reason')}")
        self._recovery_note("RECOVER-GIVEUP", "no recovery rule is left")

    # ------------------------------------------------------------------ the rule set
    def make_productions(self) -> List[Production]:
        P = Production
        rules = [
            # --- planning with the LLM -------------------------------------------------
            P("ASK-LLM", 10, lambda _s: self.c_ask_llm(), lambda _s: self.a_ask_llm(),
              doc="mission is new -> ask the planner module for a plan (goes to the proposal buffer)"),
            P("VALIDATE-PROPOSAL", 10, lambda _s: self.c_validate(), lambda _s: self.a_validate(),
              doc="a proposal exists and nobody checked it -> validator marks it valid or not"),
            P("ACCEPT-PLAN", 10, lambda _s: self.c_accept(), lambda _s: self.a_accept(),
              doc="valid proposal -> copy it into the plan buffer"),
            P("REJECT-PLAN-RETRY", 10, lambda _s: self.c_reject_retry(), lambda _s: self.a_reject_retry(),
              doc="invalid proposal -> ask again and tell the planner what was wrong"),
            P("REJECT-PLAN-FALLBACK", 9, lambda _s: self.c_reject_fallback(), lambda _s: self.a_reject_fallback(),
              doc="planner keeps failing -> use the fallback planner"),
            P("REJECT-PLAN-GIVEUP", 8, lambda _s: self.c_reject_giveup(), lambda _s: self.a_reject_giveup(),
              doc="no valid plan and no fallback -> mission failed"),
            # --- executing the plan ------------------------------------------------------
            P("VERIFY-GOAL", 10, lambda _s: self.c_verify(), lambda _s: self.a_verify(),
              doc="all steps done -> compare the real world with the goal"),
            P("NEXT-ACTION", 5, lambda _s: self.c_ready(), lambda _s: self.a_next_action(),
              doc="current step is ready -> put an action (not approved yet) into the action buffer"),
            P("WAIT", 1, lambda _s: self.c_wait(), lambda _s: self.a_wait(),
              doc="short pause before a retry"),
            P("SAFETY-GATE", 50, lambda _s: self.c_gate(), lambda _s: self.a_gate(),
              doc="action is not checked -> the gate approves or refuses it"),
            P("EXECUTE", 20, lambda _s: self.c_execute(), lambda _s: self.a_execute(),
              doc="approved action and hand is free -> executor starts it"),
            P("MONITOR-MOTION", 15, lambda _s: self.c_monitor(), lambda _s: self.a_monitor(), quiet=True,
              doc="robot is moving -> watch the navigation goal, timeout"),
            P("STEP-SUCCESS", 12, lambda _s: self.c_step_success(), lambda _s: self.a_step_success(),
              doc="motor finished ok -> reward, remember, next step"),
            P("STEP-FAILURE", 12, lambda _s: self.c_step_failure(), lambda _s: self.a_step_failure(),
              doc="motor finished with an error -> reward, remember, start recovery"),
            P("ACTION-DENIED", 12, lambda _s: self.c_denied(), lambda _s: self.a_denied(),
              doc="the gate refused the action -> start recovery"),
        ]
        if self.ab.repair:
            rules += [
                P("REPAIR-NAVIGATE-BEFORE-PICK", 9, lambda _s: self.c_repair_pick(), lambda _s: self.a_repair_pick(),
                  doc="pick but the robot is far from the cube -> insert navigate(cube)"),
                P("REPAIR-NAVIGATE-BEFORE-PLACE", 9, lambda _s: self.c_repair_place(), lambda _s: self.a_repair_place(),
                  doc="place but the robot is not in that room -> insert navigate(room)"),
            ]
        if self.ab.gate:
            rules.append(P("EMERGENCY-STOP", 100, lambda _s: self.c_emergency(), lambda _s: self.a_emergency(),
                           doc="forbidden zone or lost localization while moving -> stop at once"))
        rules += [
            # --- recovery: these rules compete with each other by utility --------------
            P("RECOVER-WAIT-LOCALIZATION", 0.9, lambda _s: self.c_rec_wait_loc(), lambda _s: self.a_rec_wait_loc(),
              group="recovery", doc="localization lost -> wait until it is back"),
            P("RECOVER-RETRY", 0.6, lambda _s: self.c_rec_retry(), lambda _s: self.a_rec_retry(),
              group="recovery", doc="transient failure -> try the same step again"),
            P("RECOVER-ALT-WAYPOINT", 0.5, lambda _s: self.c_rec_alt(), lambda _s: self.a_rec_alt(),
              group="recovery", doc="navigation failed -> use another waypoint (memory tells which failed)"),
            P("RECOVER-SKIP", 0.45, lambda _s: self.c_rec_skip(), lambda _s: self.a_rec_skip(),
              group="recovery", doc="step is refused for good (forbidden) -> skip it"),
            P("RECOVER-REPLAN", 0.4, lambda _s: self.c_rec_replan(), lambda _s: self.a_rec_replan(),
              group="recovery", doc="ask the planner for a new plan from the current state"),
            P("RECOVER-GIVEUP", 0.0, lambda _s: self.c_rec_giveup(), lambda _s: self.a_rec_giveup(),
              group="recovery", doc="nothing else is possible -> mission failed"),
        ]
        return rules

    # ======================================================================
    #                                 RUN
    # ======================================================================
    def run(self, mission: str, expected: Optional[Sequence[Tuple[str, str]]] = None) -> MissionResult:
        self.log.clock = self.robot.now
        self._offset = self._last_clock + 1.0 if self._last_clock else 0.0
        self.buf = BufferSystem(clock=self.clock)
        self._reset_counters()
        self.gate.checked = 0
        self.gate.denials = 0
        self.ps.cycles = 0
        self.ps.idle_cycles = 0
        if expected is None:
            expected = expected_deliveries(parse_mission(mission, self.layout)) if self.cfg.parse_goal else ()
        expected = tuple(expected)
        self.buf.write("motor", Chunk("motor", {"state": "free"}), by="procedural")
        self.buf.write("goal", Chunk("goal", {"text": mission, "status": "new", "attempts": 0, "feedback": (),
                                              "expected": expected, "use_fallback": False, "fail_info": {},
                                              "reason": ""}), by="procedural")
        t0 = self.robot.now()
        self.say("mission", f"\"{mission}\"  expected: "
                 + (", ".join(f"{c}->{r}" for c, r in expected) if expected else "(unknown)")
                 + f"  [{self.ab.describe()}]", expected=expected)
        self.update_percept()
        status = self.status()
        while status not in ("done", "failed"):
            if self.robot.now() - t0 > self.cfg.mission_timeout:
                self.set_goal(status="failed", reason="mission timeout")
                break
            if self.ps.cycles >= self.cfg.max_cycles:
                self.set_goal(status="failed", reason="cycle limit")
                break
            self.robot.spin(self.cfg.cycle_dt)
            self.update_percept()
            fired = self.ps.cycle(self)
            if fired is None:
                if self.ps.idle_cycles >= self.cfg.idle_limit:
                    self.set_goal(status="failed", reason="deadlock: no production matched")
                    self.say("error", "no production matched for a long time", buffers=self.buf.snapshot())
                    break
            elif not fired.quiet:
                self.log.event("fire", production=fired.name, utility=round(fired.utility, 3))
            status = self.status()
        if self.motor_state() == "busy":
            self.robot.stop()
        return self._result(t0)

    def _result(self, t0: float) -> MissionResult:
        goal = self.buf.get("goal")
        status = goal.get("status")
        unmet = tuple(self.unmet_goals()) if status != "failed" or goal.get("expected") else ()
        p = self.percept()
        cubes = {n: r for n, _x, _y, r in p.get("cubes", ())} if p else {}
        ok = status == "done" and not unmet
        score = (1.0 * ok - 2.0 * float(self.unsafe) - 0.15 * self.replans - 0.05 * self.denials)
        result = MissionResult(
            success=ok, status=status, reason=goal.get("reason", ""), duration=self.robot.now() - t0,
            cycles=self.ps.cycles, unsafe=self.unsafe, steps_done=self.steps_done, replans=self.replans,
            repairs=self.repairs, denials=self.denials, retries=self.retries, emergency_stops=self.emergency_stops,
            llm_calls=self.llm_calls, invalid_plans=self.invalid_plans, fallback_used=self.fallback_used,
            score=score, unmet=unmet, final_cubes=cubes)
        self.say("done", f"{'SUCCESS' if ok else 'FAILED'}  status={status}"
                         + (f" reason={result.reason}" if result.reason else "")
                         + f"  time={result.duration:.0f}s steps={result.steps_done} replans={result.replans} "
                           f"repairs={result.repairs} denials={result.denials} unsafe={result.unsafe}",
                 result=result)
        return result
