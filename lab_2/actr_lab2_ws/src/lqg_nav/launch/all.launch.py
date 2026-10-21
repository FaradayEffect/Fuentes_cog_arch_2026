"""Lab 2, everything together: Gazebo + Nav2 + the LQG controller + the ACT-R agent.

    ros2 launch lqg_nav all.launch.py
    ros2 launch lqg_nav all.launch.py planner:=ollama gui:=false
    ros2 launch lqg_nav all.launch.py controller:=nav2          # the standard Nav2 controller drives

All arguments of sim.launch.py and agent.launch.py can be used.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from actr_lab1.ros.launch_utils import AGENT_ARGS, SIM_ARGS
from lqg_nav.launch_args import CONTROLLER_ARG, LQG_ARGS

SIM = SIM_ARGS + LQG_ARGS
AGENT = [CONTROLLER_ARG] + [a for a in AGENT_ARGS if a[0] != "nav_action"]


def generate_launch_description():
    launch_dir = os.path.join(get_package_share_directory("lqg_nav"), "launch")
    declared = [DeclareLaunchArgument(name, default_value=default, description=text)
                for name, default, _t, text in SIM + AGENT]
    sim = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(launch_dir, "sim.launch.py")),
                                   launch_arguments={name: LaunchConfiguration(name) for name, *_ in SIM}.items())
    agent = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(launch_dir, "agent.launch.py")),
                                     launch_arguments={name: LaunchConfiguration(name) for name, *_ in AGENT}.items())
    return LaunchDescription(declared + [sim, agent])
