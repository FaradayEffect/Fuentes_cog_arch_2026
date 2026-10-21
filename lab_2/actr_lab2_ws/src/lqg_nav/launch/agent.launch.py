"""Lab 2: the ACT-R agent of Lab 1. By default the LQG controller drives; controller:=nav2 gives the standard
Nav2 controller (to compare the two in the same simulation).

    ros2 launch lqg_nav agent.launch.py planner:=ollama shutdown_when_done:=true
    ros2 launch lqg_nav agent.launch.py planner:=ollama shutdown_when_done:=true controller:=nav2
All the arguments of actr_lab1 agent.launch.py work as well (mission, planner, show_tokens ...).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from actr_lab1.ros.launch_utils import AGENT_ARGS
from lqg_nav.launch_args import CONTROLLER_ARG, nav_action

PASSED = [a for a in AGENT_ARGS if a[0] != "nav_action"]       # nav_action comes from `controller`


def _agent(context, *args, **kwargs):
    controller = LaunchConfiguration("controller").perform(context)
    action = nav_action(controller)
    arguments = {name: LaunchConfiguration(name) for name, *_ in PASSED}
    arguments["nav_action"] = action
    lab1 = os.path.join(get_package_share_directory("actr_lab1"), "launch", "agent.launch.py")
    return [LogInfo(msg=f"[lqg_nav] the robot is driven by: {controller} (action /{action})"),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(lab1), launch_arguments=arguments.items())]


def generate_launch_description():
    declared = [DeclareLaunchArgument(name, default_value=default, description=text)
                for name, default, _t, text in [CONTROLLER_ARG] + PASSED]
    return LaunchDescription(declared + [OpaqueFunction(function=_agent)])
