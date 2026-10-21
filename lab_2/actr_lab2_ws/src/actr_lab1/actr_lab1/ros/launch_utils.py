"""Helpers for the launch files (kept here so that they can be tested without ROS).

Only the function `share_dir` imports ament; everything else is plain Python.
"""
from __future__ import annotations

import glob
import os
from typing import Callable, Dict, Optional

# Launch arguments: (name, default, type, help). They are declared in all.launch.py, sim.launch.py and
# agent.launch.py, so the same list is kept here.
SIM_ARGS = [
    ("gui", "true", bool, "open the Gazebo window"),
    ("nav2", "true", bool, "start Nav2 (map server, AMCL, planner ...)"),
    ("rviz", "false", bool, "open RViz with the Nav2 view"),
    ("nav2_params", "", str, "your own Nav2 parameter file (empty = the one of TurtleBot3)"),
    ("arm", "true", bool, "add the forklift (forks + lift) to the robot; false = no forklift, virtual gripper"),
]

AGENT_ARGS = [
    ("mission", "Отвези кубик 0 и кубик 1 в комнату А.", str, "mission text (Russian or English)"),
    ("missions", "", str, "several missions separated by |  (replaces 'mission')"),
    ("planner", "mock", str, "mock | ollama | openai"),
    ("llm_model", "qwen2.5:3b-instruct", str, "model name on the LLM server"),
    ("llm_host", "", str, "address of the LLM server (empty = $OLLAMA_HOST, or http://localhost:11434 for ollama)"),
    ("llm_timeout", "180.0", float, "seconds to wait for one LLM answer"),
    ("fallback", "true", bool, "use the mock planner when the LLM gives no valid plan"),
    ("ablate", "", str, "switch parts off: validator,repair,gate,recovery,memory,learning"),
    ("log_dir", "", str, "folder for events.jsonl and results.jsonl (empty = ~/actr_lab1_logs/<time>)"),
    ("shutdown_when_done", "false", bool, "stop the node after the last mission"),
    ("fault_loc_at", "-1.0", float, "demo: simulation time when the localization is declared lost (-1 = never)"),
    ("fault_loc_for", "0.0", float, "demo: how long it stays lost (seconds)"),
    ("gripper", "auto", str, "auto | arm | virtual  (auto = use the forklift if the robot in Gazebo has one)"),
    ("show_tokens", "false", bool, "print how many tokens the LLM used for each answer (counted by Ollama)"),
    ("nav_action", "navigate_to_pose", str,
     "action that drives the robot: navigate_to_pose (standard Nav2) or lqg/navigate_to_pose (LQG, Lab 2)"),
]

INSTALL_HINT = (
    "sudo apt install ros-$ROS_DISTRO-turtlebot3-gazebo ros-$ROS_DISTRO-turtlebot3-navigation2 "
    "ros-$ROS_DISTRO-navigation2 ros-$ROS_DISTRO-nav2-bringup ros-$ROS_DISTRO-gazebo-ros-pkgs"
)


def share_dir(package: str) -> str:
    from ament_index_python.packages import get_package_share_directory
    return get_package_share_directory(package)


def append_path(existing: str, *extra: str) -> str:
    """Add folders to a PATH-like variable (GAZEBO_MODEL_PATH) without repeating them."""
    parts = [p for p in existing.split(os.pathsep) if p]
    for e in extra:
        if e and e not in parts:
            parts.append(e)
    return os.pathsep.join(parts)


# Where Ubuntu puts the models of Gazebo Classic 11 (apt package gazebo). Our world does not need them any more
# (it writes the ground and the sun itself), but they are added when they exist, so that a model of the
# standard library still loads.
GAZEBO_SYSTEM_MODEL_DIRS = ("/usr/share/gazebo-11/models", "/usr/share/gazebo/models")


def gazebo_model_path(existing: str, turtlebot_models: str, system_dirs=GAZEBO_SYSTEM_MODEL_DIRS) -> str:
    """The value of GAZEBO_MODEL_PATH: what is there + the TurtleBot3 models + the models of Gazebo (if they exist)."""
    return append_path(existing, turtlebot_models, *[d for d in system_dirs if os.path.isdir(d)])


def _first_existing(candidates) -> Optional[str]:
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def tb3_files(model: str = "burger", share: Optional[Callable[[str], str]] = None,
              ros_distro: str = "") -> Dict[str, str]:
    """Where the TurtleBot3 files are. Raises FileNotFoundError with an install hint."""
    share = share or share_dir
    ros_distro = ros_distro or os.environ.get("ROS_DISTRO", "")
    problems = []
    try:
        gz = share("turtlebot3_gazebo")
    except Exception:
        raise FileNotFoundError("package turtlebot3_gazebo not found. Install: " + INSTALL_HINT)
    sdf = os.path.join(gz, "models", f"turtlebot3_{model}", "model.sdf")
    urdf = os.path.join(gz, "urdf", f"turtlebot3_{model}.urdf")
    for name, path in (("model.sdf", sdf), ("urdf", urdf)):
        if not os.path.isfile(path):
            problems.append(f"{name} not found: {path}")
    if problems:
        raise FileNotFoundError("; ".join(problems) + f". Is TURTLEBOT3_MODEL='{model}' correct? (burger is tested)")

    params = None
    try:
        nav = share("turtlebot3_navigation2")
        params = _first_existing([
            os.path.join(nav, "param", ros_distro, f"{model}.yaml"),
            os.path.join(nav, "param", f"{model}.yaml"),
        ]) or _first_existing(sorted(glob.glob(os.path.join(nav, "param", "**", f"{model}.yaml"), recursive=True)))
    except Exception:
        pass
    if params is None:
        try:
            params = _first_existing([os.path.join(share("nav2_bringup"), "params", "nav2_params.yaml")])
        except Exception:
            params = None
    if params is None:
        raise FileNotFoundError("no Nav2 parameter file found (turtlebot3_navigation2 or nav2_bringup). Install: "
                                + INSTALL_HINT)
    return {"sdf": sdf, "urdf": urdf, "models_dir": os.path.join(gz, "models"), "nav2_params": params}
