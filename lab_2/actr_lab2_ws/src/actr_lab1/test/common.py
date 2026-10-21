"""Shared helpers for the tests. Run them from src/actr_lab1:

    python3 -m unittest discover -s test -v
"""
import os
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from actr_lab1.core.world import Layout, OccupancyGrid  # noqa: E402

_cache = {}


def load_world():
    """Layout and grid are read once (the grid inflation is the slow part)."""
    if "world" not in _cache:
        layout = Layout.load(os.path.join(PKG_ROOT, "config", "places.json"))
        grid = OccupancyGrid.from_yaml(layout.map_yaml_path())
        _cache["world"] = (layout, grid)
    return _cache["world"]


# A small copy of the structure of turtlebot3_gazebo/models/turtlebot3_burger/model.sdf (ROS 2 Humble):
# the same links, joints and plugin names, without meshes. Used to test the arm that is added to it.
TB3_BURGER_SDF = """<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="turtlebot3_burger">
    <pose>0.0 0.0 0.0 0.0 0.0 0.0</pose>
    <link name="base_footprint"/>
    <link name="base_link">
      <inertial><pose>-0.032 0 0.070 0 0 0</pose><mass>0.8</mass>
        <inertia><ixx>0.007</ixx><iyy>0.007</iyy><izz>0.004</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name="base_collision"><pose>-0.032 0 0.070 0 0 0</pose>
        <geometry><box><size>0.140 0.140 0.140</size></box></geometry></collision>
      <visual name="base_visual"><pose>-0.032 0 0 0 0 0</pose>
        <geometry><box><size>0.140 0.140 0.140</size></box></geometry></visual>
    </link>
    <link name="base_scan">
      <sensor name="hls_lfcd_lds" type="ray"><pose>-0.032 0 0.171 0 0 0</pose>
        <ray><range><min>0.120000</min><max>3.5</max></range></ray>
        <plugin name="turtlebot3_laserscan" filename="libgazebo_ros_ray_sensor.so">
          <ros><remapping>~/out:=scan</remapping></ros>
        </plugin>
      </sensor>
    </link>
    <link name="wheel_left_link"/>
    <link name="wheel_right_link"/>
    <joint name="base_joint" type="fixed"><parent>base_footprint</parent><child>base_link</child></joint>
    <joint name="wheel_left_joint" type="revolute"><parent>base_link</parent><child>wheel_left_link</child>
      <axis><xyz>0 0 1</xyz></axis></joint>
    <joint name="wheel_right_joint" type="revolute"><parent>base_link</parent><child>wheel_right_link</child>
      <axis><xyz>0 0 1</xyz></axis></joint>
    <joint name="lidar_joint" type="fixed"><parent>base_link</parent><child>base_scan</child></joint>
    <plugin name="turtlebot3_diff_drive" filename="libgazebo_ros_diff_drive.so">
      <left_joint>wheel_left_joint</left_joint><right_joint>wheel_right_joint</right_joint>
      <command_topic>cmd_vel</command_topic>
    </plugin>
  </model>
</sdf>
"""
