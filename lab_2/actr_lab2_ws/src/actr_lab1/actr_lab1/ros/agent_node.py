"""ROS 2 node: runs the ACT-R agent on the TurtleBot3 in Gazebo.

    ros2 run actr_lab1 agent_node --ros-args -p use_sim_time:=true -p planner:=ollama

The node does four things:
  1. waits until Gazebo, the localization and Nav2 are ready,
  2. builds the agent (planner = mock or a local LLM, safety gate, memory ...),
  3. runs the missions (parameter `mission`/`missions`, or text sent to the topic /actr/mission),
  4. writes the logs (events.jsonl, results.jsonl) and publishes the events on /actr/events.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from collections import deque
from typing import Deque, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ..core.agent import Ablation, ActrAgent, AgentConfig
from ..core.logs import EventLog
from ..core.paths import load_world
from ..core.planners import MockLLMPlanner, OllamaPlanner, OpenAICompatPlanner, Planner, model_installed
from .ros_robot import RosRobot, RosRobotConfig, RosShutdown

DEFAULT_MISSION = "Отвези кубик 0 и кубик 1 в комнату А."
ABLATION_FIELDS = ("validator", "repair", "gate", "recovery", "memory", "learning")


def parse_ablation(text: str) -> Ablation:
    """'gate,memory' -> Ablation(gate=False, memory=False)."""
    names = [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]
    for name in names:
        if name not in ABLATION_FIELDS:
            raise ValueError(f"unknown ablation '{name}' (allowed: {', '.join(ABLATION_FIELDS)})")
    return Ablation(**{name: False for name in names})


def split_missions(text: str) -> List[str]:
    return [m.strip() for m in text.split("|") if m.strip()]


class AgentNode(Node):
    def __init__(self):
        super().__init__("actr_agent")
        defaults = {
            "mission": DEFAULT_MISSION,
            "missions": "",                 # several missions separated by '|'
            "planner": "mock",              # mock | ollama | openai
            "llm_model": "qwen2.5:3b-instruct",
            "llm_host": "",                 # empty = the usual address of the chosen server
            "llm_timeout": 180.0,
            "llm_temperature": 0.0,
            "fallback": True,               # use the mock planner if the LLM gives no valid plan
            "ablate": "",                   # e.g. "gate" or "gate,memory": switch parts off (for experiments)
            "log_dir": "",
            "places_dir": "",               # folder with config/ and maps/ (empty = the installed package)
            "pick_radius": 1.0,
            "nav_timeout": 150.0,
            "startup_timeout": 240.0,
            "wait_for_ready": True,
            "shutdown_when_done": False,
            "fault_loc_at": -1.0,           # demo: pretend that the localization is lost from this time ...
            "fault_loc_for": 0.0,           # ... for this many seconds
            "gripper": "auto",              # auto | arm | virtual (auto = the arm, if the robot has one)
            "show_tokens": False,           # print how many tokens the LLM used (they are always saved in the log file)
            "nav_action": "navigate_to_pose",   # who drives: navigate_to_pose = Nav2, lqg/navigate_to_pose = LQG (Lab 2)
            "seed": 0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}

        self.queue: Deque[str] = deque()
        self.mission_sub = self.create_subscription(String, "/actr/mission", self._on_mission_msg, 10)
        self.events_pub = self.create_publisher(String, "/actr/events", 50)
        self.robot: Optional[RosRobot] = None
        self.agent: Optional[ActrAgent] = None
        self.log: Optional[EventLog] = None
        self.results_path = ""

    # ------------------------------------------------------------------ small things
    def _on_mission_msg(self, msg: String) -> None:
        text = msg.data.strip()
        if text:
            self.queue.append(text)
            self.get_logger().info(f"new mission in the queue: {text}")

    def _publish_event(self, record: dict) -> None:
        if not record.get("msg"):
            return
        out = String()
        out.data = json.dumps({"t": record["t"], "kind": record["kind"], "msg": record["msg"]}, ensure_ascii=False)
        self.events_pub.publish(out)

    def make_planner(self, layout) -> Planner:
        kind = str(self.p["planner"]).lower()
        host = str(self.p["llm_host"])
        timeout = float(self.p["llm_timeout"])
        temperature = float(self.p["llm_temperature"])
        if kind == "mock":
            return MockLLMPlanner(layout)
        if kind == "ollama":
            return OllamaPlanner(layout, model=str(self.p["llm_model"]), host=host,
                                 timeout=timeout, temperature=temperature, seed=int(self.p["seed"]))
        if kind in ("openai", "openai-compat"):
            return OpenAICompatPlanner(layout, model=str(self.p["llm_model"]), host=host or "http://localhost:1234/v1",
                                       timeout=timeout, temperature=temperature, seed=int(self.p["seed"]))
        raise ValueError(f"unknown planner '{kind}' (use mock, ollama or openai)")

    def check_llm(self, planner: Planner) -> None:
        """Say clearly if the local LLM server cannot be reached (the fallback planner will be used)."""
        if not isinstance(planner, OllamaPlanner):
            return
        try:
            models = planner.list_models()
        except Exception as e:
            self.get_logger().warn(f"Ollama is not reachable at {planner.host}: {e}. "
                                   f"Start it with 'ollama serve'. The fallback planner will be used.")
            return
        if not model_installed(planner.model, models):
            self.get_logger().warn(f"Ollama has no model '{planner.model}' (found: {', '.join(models) or 'none'}). "
                                   f"Run: ollama pull {planner.model}")

    # --------------------------------------------------------------------- set up
    def setup(self) -> List[str]:
        """Build everything. Returns the list of problems found while waiting for the simulator."""
        p = self.p
        layout, grid = load_world(str(p["places_dir"]))
        log_dir = str(p["log_dir"]) or os.path.expanduser(os.path.join("~", "actr_lab1_logs",
                                                                       time.strftime("%Y%m%d_%H%M%S")))
        self.log = EventLog(directory=log_dir, console=True, keep=False, on_event=self._publish_event)
        self.results_path = os.path.join(log_dir, "results.jsonl")

        gripper = str(p["gripper"]).strip().lower() or "auto"
        if gripper not in ("auto", "arm", "virtual"):
            raise ValueError(f"unknown gripper '{gripper}' (use auto, arm or virtual)")
        cfg = RosRobotConfig(pick_radius=float(p["pick_radius"]),
                             loc_fault_at=float(p["fault_loc_at"]), loc_fault_for=float(p["fault_loc_for"]),
                             gripper=gripper, robot_entity=os.environ.get("TURTLEBOT3_MODEL", "burger") or "burger",
                             nav_action=str(p["nav_action"]).strip() or "navigate_to_pose")
        self.robot = RosRobot(self, layout, cfg)

        planner = self.make_planner(layout)
        self.check_llm(planner)
        fallback = MockLLMPlanner(layout) if (p["fallback"] and not isinstance(planner, MockLLMPlanner)) else None
        reach = float(p["pick_radius"])
        agent_cfg = AgentConfig(nav_timeout=float(p["nav_timeout"]), pick_radius=reach + 0.1, near_radius=reach - 0.1,
                                show_tokens=bool(p["show_tokens"]))
        import random
        self.agent = ActrAgent(layout, grid, self.robot, planner, fallback_planner=fallback, config=agent_cfg,
                               ablation=parse_ablation(str(p["ablate"])), log=self.log,
                               rng=random.Random(int(p["seed"])))
        self.get_logger().info(
            f"planner={planner.name} fallback={'yes' if fallback else 'no'} ablation={self.agent.ab.describe()} "
            f"drives={self.robot.cfg.nav_action} logs={log_dir}")

        if not p["wait_for_ready"]:
            return []
        t = float(p["startup_timeout"])
        problems = self.robot.wait_until_ready(clock_timeout=t, gazebo_timeout=t, loc_timeout=t, nav_timeout=t,
                                               say=lambda s: self.get_logger().info(s))
        if not problems and self.robot.use_arm():
            # The forklift stops further from a cube (the forks must not touch the pallet when the robot turns)
            # and drives the last part by itself, so "next to the cube" is a bigger circle.
            self.agent.cfg.near_radius = self.robot.cfg.fork_near
            self.agent.gate.pick_radius = self.robot.cfg.fork_reach
        return problems

    # ------------------------------------------------------------------------ run
    def initial_missions(self) -> List[str]:
        many = split_missions(str(self.p["missions"]))
        if many:
            return many
        text = str(self.p["mission"]).strip()
        return [text] if text else []

    def run_mission(self, text: str) -> None:
        result = self.agent.run(text)
        record = dataclasses.asdict(result)
        record.update({"mission": text, "planner": self.agent.planner.name, "ablation": self.agent.ab.describe()})
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def run(self) -> int:
        problems = self.setup()
        if problems:
            for text in problems:
                self.get_logger().error(text)
            self.get_logger().error("The simulator is not ready. Run tools/preflight.sh for a check list.")
            return 1
        for text in self.initial_missions():
            self.queue.append(text)
        self.get_logger().info("READY. More missions: ros2 topic pub --once /actr/mission std_msgs/msg/String "
                               "\"{data: 'Bring cube 0 to room A'}\"")
        done_any = False
        while rclpy.ok():
            if self.queue:
                self.run_mission(self.queue.popleft())
                done_any = True
            elif done_any and self.p["shutdown_when_done"]:
                break
            else:
                self.robot.spin(0.2)
        return 0

    def close(self) -> None:
        if self.robot is not None:
            try:
                self.robot.shutdown()
            except Exception:
                pass
        if self.log is not None:
            self.log.close()
        try:
            self.destroy_node()
        except Exception:
            pass


def init_ros(args=None) -> None:
    """rclpy.init() without the signal handler of rclpy.

    With its own handler rclpy closes the ROS context at the first Ctrl-C, and then the robot can no longer be
    told to stop. Without it, Python raises KeyboardInterrupt, main() stops the robot (cancels the Nav2 goal,
    sends zero speed) and only then closes ROS. ROS 2 Foxy has no such option: there rclpy keeps its handler
    (the node still stops, but it cannot send the last stop command)."""
    try:
        rclpy.init(args=args, signal_handler_options=rclpy.SignalHandlerOptions.NO)
    except (TypeError, AttributeError):
        rclpy.init(args=args)


def main(args=None) -> int:
    init_ros(args)
    node = AgentNode()
    code = 1
    try:
        code = node.run()
    except (KeyboardInterrupt, RosShutdown):
        node.get_logger().info("stopped by the user")
        code = 0
    finally:
        node.close()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
