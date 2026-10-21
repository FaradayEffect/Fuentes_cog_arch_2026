"""Everything together: Gazebo + Nav2 + the ACT-R agent.

    ros2 launch actr_lab1 all.launch.py
    ros2 launch actr_lab1 all.launch.py planner:=ollama gui:=false
    ros2 launch actr_lab1 all.launch.py mission:="Bring cube 1 to room A"

All arguments of sim.launch.py and agent.launch.py can be used.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from actr_lab1.ros.launch_utils import AGENT_ARGS, SIM_ARGS


def generate_launch_description():
    launch_dir = os.path.join(get_package_share_directory("actr_lab1"), "launch")
    sim = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(launch_dir, "sim.launch.py")))
    agent = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(launch_dir, "agent.launch.py")))
    declared = [DeclareLaunchArgument(name, default_value=default, description=text)
                for name, default, _t, text in SIM_ARGS + AGENT_ARGS]
    return LaunchDescription(declared + [sim, agent])
