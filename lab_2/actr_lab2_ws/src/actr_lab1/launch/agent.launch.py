"""The ACT-R agent node (it waits for Gazebo and Nav2 by itself).

    ros2 launch actr_lab1 agent.launch.py                                   # mock planner, default mission
    ros2 launch actr_lab1 agent.launch.py planner:=ollama llm_model:=qwen2.5:3b-instruct
    ros2 launch actr_lab1 agent.launch.py mission:="Bring cube 0 to room B"
    ros2 launch actr_lab1 agent.launch.py ablate:=gate                      # experiment: safety gate OFF
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from actr_lab1.ros.launch_utils import AGENT_ARGS


def generate_launch_description():
    declared = [DeclareLaunchArgument(name, default_value=default, description=text)
                for name, default, _t, text in AGENT_ARGS]
    params = {name: ParameterValue(LaunchConfiguration(name), value_type=t) for name, _d, t, _x in AGENT_ARGS}
    config = os.path.join(get_package_share_directory("actr_lab1"), "config", "agent.yaml")
    node = Node(package="actr_lab1", executable="agent_node", name="actr_agent", output="screen",
                emulate_tty=True, parameters=[config, {"use_sim_time": True}, params])
    return LaunchDescription(declared + [node])
