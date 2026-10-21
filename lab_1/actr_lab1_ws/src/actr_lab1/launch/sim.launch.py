"""Gazebo (our world) + TurtleBot3 Burger + Nav2 with the map of the same world.

    ros2 launch actr_lab1 sim.launch.py                 # Gazebo window + Nav2
    ros2 launch actr_lab1 sim.launch.py gui:=false      # no Gazebo window (faster)
    ros2 launch actr_lab1 sim.launch.py nav2:=false     # only the world and the robot (first check)
    ros2 launch actr_lab1 sim.launch.py rviz:=true      # also open RViz
    ros2 launch actr_lab1 sim.launch.py arm:=false      # the robot without the forklift (old virtual gripper)

The agent is started by agent.launch.py (or all.launch.py starts both).
"""
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            RegisterEventHandler, SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from actr_lab1.core.world import Layout
from actr_lab1.ros.arm_model import REACH, ArmModelError, write_arm_model
from actr_lab1.ros.launch_utils import SIM_ARGS, gazebo_model_path, tb3_files
from actr_lab1.ros.nav2_params import FORKLIFT_INFLATION, INFLATION_RADIUS, ROBOT_RADIUS, SELF_FILTER, write_patched


def _is_true(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in ("true", "1", "yes")


def _start_nav2_after_spawn(actions):
    """Event handler: `actions` start when spawn_entity.py has finished. If it failed (the robot is NOT in Gazebo),
    Nav2 must not start: a message is printed and the whole launch stops."""
    def handler(event, context):
        code = getattr(event, "returncode", 0)
        if code in (0, None):
            return actions or None
        return [LogInfo(msg=f"[actr_lab1] spawn_entity.py failed (exit code {code}): the robot is not in Gazebo. "
                            "Look at the lines above it. Stopping the launch."),
                EmitEvent(event=Shutdown(reason="spawn_entity.py failed"))]
    return handler


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory("actr_lab1")
    layout = Layout.load(os.path.join(pkg, "config", "places.json"))
    model = os.environ.get("TURTLEBOT3_MODEL", "burger")
    files = tb3_files(model)
    world = os.path.join(pkg, "worlds", "rooms.world")
    map_yaml = os.path.join(pkg, "maps", "rooms.yaml")
    sx, sy, syaw = layout.robot_start.x, layout.robot_start.y, layout.robot_start.yaw

    actions = []
    actions.append(LogInfo(msg=f"[actr_lab1] robot: TurtleBot3 {model}, start pose ({sx}, {sy}, {syaw})"))
    # a new private folder for every launch (a fixed name in /tmp could belong to another user)
    tmp = tempfile.mkdtemp(prefix="actr_lab1_")

    # The forklift (forks + lift + counterweight) is added to a copy of the TurtleBot3 model. If that fails,
    # the robot simply has no forklift and the agent uses the virtual gripper.
    sdf, radius, inflation, self_filter = files["sdf"], ROBOT_RADIUS, INFLATION_RADIUS, 0.0
    if _is_true(context, "arm"):
        arm_sdf = os.path.join(tmp, "model.sdf")
        try:
            write_arm_model(files["sdf"], arm_sdf)
            sdf, radius, inflation, self_filter = arm_sdf, REACH, FORKLIFT_INFLATION, SELF_FILTER
            actions.append(LogInfo(msg="[actr_lab1] forklift: forks and a lift are added to the robot"))
        except (ArmModelError, OSError) as e:
            actions.append(LogInfo(msg=f"[actr_lab1] the forklift could not be added ({e}). "
                                       "The robot has no forklift; the agent will use the virtual gripper."))

    # Gazebo must find the TurtleBot3 meshes. No online model database (it can block the start).
    actions.append(SetEnvironmentVariable("GAZEBO_MODEL_PATH",
                                          gazebo_model_path(os.environ.get("GAZEBO_MODEL_PATH", ""),
                                                            files["models_dir"])))
    actions.append(SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""))
    actions.append(SetEnvironmentVariable("TURTLEBOT3_MODEL", model))

    gazebo_launch = os.path.join(get_package_share_directory("gazebo_ros"), "launch")
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_launch, "gzserver.launch.py")),
        launch_arguments={"world": world, "verbose": "true"}.items()))
    if _is_true(context, "gui"):
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_launch, "gzclient.launch.py"))))

    with open(files["urdf"], "r") as f:
        robot_description = f.read()
    actions.append(Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
                        parameters=[{"use_sim_time": True, "robot_description": robot_description}]))

    spawn = Node(package="gazebo_ros", executable="spawn_entity.py", output="screen",
                 arguments=["-entity", model, "-file", sdf, "-x", str(sx), "-y", str(sy), "-z", "0.01",
                            "-Y", str(syaw)])
    actions.append(spawn)

    after_spawn = []
    if _is_true(context, "nav2"):
        source = LaunchConfiguration("nav2_params").perform(context) or files["nav2_params"]
        patched = os.path.join(tmp, "nav2_params.yaml")
        # with the forklift the robot is longer (Nav2 keeps it further from the walls), and the lidar must
        # not take its own forks and load for obstacles
        notes = write_patched(source, patched, layout.robot_start, robot_radius=radius, inflation_radius=inflation,
                              self_filter=self_filter)
        after_spawn.append(LogInfo(msg="[actr_lab1] Nav2 parameters: " + source + " -> " + patched
                                       + "  (" + "; ".join(notes) + ")"))
        nav2_launch = os.path.join(get_package_share_directory("nav2_bringup"), "launch", "bringup_launch.py")
        after_spawn.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={"map": map_yaml, "use_sim_time": "true", "params_file": patched,
                              "autostart": "true"}.items()))
    if _is_true(context, "rviz"):
        rviz_config = os.path.join(get_package_share_directory("nav2_bringup"), "rviz", "nav2_default_view.rviz")
        after_spawn.append(Node(package="rviz2", executable="rviz2", output="screen",
                                arguments=["-d", rviz_config], parameters=[{"use_sim_time": True}]))
    # Nav2 starts when the robot exists in Gazebo (its lifecycle manager is impatient). If the spawn fails,
    # nothing else starts and the launch stops with a clear message (also when nav2:=false).
    actions.append(RegisterEventHandler(OnProcessExit(target_action=spawn,
                                                      on_exit=_start_nav2_after_spawn(after_spawn))))
    return actions


def generate_launch_description():
    declared = [DeclareLaunchArgument(name, default_value=default, description=text)
                for name, default, _t, text in SIM_ARGS]
    return LaunchDescription(declared + [OpaqueFunction(function=launch_setup)])
