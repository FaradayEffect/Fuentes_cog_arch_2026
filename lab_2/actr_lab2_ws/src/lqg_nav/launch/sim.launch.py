"""Lab 2: the simulation of Lab 1 (Gazebo, TurtleBot3 with the forklift, Nav2) plus the LQG controller.

    ros2 launch lqg_nav sim.launch.py                        # Gazebo window, Nav2, LQG controller, monitor
    ros2 launch lqg_nav sim.launch.py steering:=lqr          # comparison: the LQR without the Kalman filter
    ros2 launch lqg_nav sim.launch.py pose_noise:=0.02 yaw_noise:=0.03     # experiment: a noisy pose
All the arguments of actr_lab1 sim.launch.py work as well (gui, nav2, rviz, arm, nav2_params).
Nav2 is started as in Lab 1: its planner gives the path to the LQG controller, and its own controller stays
idle unless the agent is started with controller:=nav2.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from actr_lab1.ros.launch_utils import SIM_ARGS
from lqg_nav.launch_args import LQG_ARGS


def generate_launch_description():
    declared = [DeclareLaunchArgument(name, default_value=default, description=text)
                for name, default, _t, text in SIM_ARGS + LQG_ARGS]
    lab1 = os.path.join(get_package_share_directory("actr_lab1"), "launch", "sim.launch.py")
    sim = IncludeLaunchDescription(PythonLaunchDescriptionSource(lab1),
                                   launch_arguments={name: LaunchConfiguration(name) for name, *_ in SIM_ARGS}.items())
    config = os.path.join(get_package_share_directory("lqg_nav"), "config", "lqg.yaml")
    params = {name: ParameterValue(LaunchConfiguration(name), value_type=t) for name, _d, t, _x in LQG_ARGS}
    params["forklift"] = ParameterValue(LaunchConfiguration("arm"), value_type=bool)
    with_nav2 = IfCondition(LaunchConfiguration("nav2"))
    navigator = Node(package="lqg_nav", executable="lqg_navigator", name="lqg_navigator", output="screen",
                     emulate_tty=True, parameters=[config, {"use_sim_time": True}, params], condition=with_nav2)
    monitor = Node(package="lqg_nav", executable="track_monitor", name="track_monitor", output="screen",
                   emulate_tty=True, parameters=[{"use_sim_time": True}], condition=with_nav2)
    return LaunchDescription(declared + [sim, navigator, monitor])
