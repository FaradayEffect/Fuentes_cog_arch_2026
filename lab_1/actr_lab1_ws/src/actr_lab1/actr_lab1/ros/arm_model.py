"""The manipulator of the robot: a small forklift in front of the TurtleBot3 Burger.

TurtleBot3 Burger has no arm, so sim.launch.py adds one to its Gazebo model (model.sdf) before the
robot is spawned. It works like a real forklift:
  * two flat forks go under a small pallet (each cube stands on its own pallet, see world_layout.py),
  * the forks are thin at the tip (a wedge made of 4 steps) and thicker at the back, so they enter
    the openings of the pallet easily,
  * a lift (prismatic joint fork_lift_joint) moves the forks and their backrest up and down,
  * a counterweight inside the robot keeps it from tipping forward with a load, like on a real forklift.

The load is held only by its weight and friction on the forks (no hidden force). The lift is moved by
a standard plugin of gazebo_ros_pkgs, so nothing has to be compiled:
  libgazebo_ros_joint_pose_trajectory.so   topic /arm/set_joint_trajectory

Plain Python (no ROS import), so it can be tested without ROS.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Tuple

# --- numbers of the forklift (metres) --------------------------------------------------
# The origin of the TurtleBot3 model is 1 cm above the floor (it is spawned with z = 0.01), so a
# height above the floor is "model z + 0.01". The numbers below are heights ABOVE THE FLOOR.
SPAWN_Z = 0.01
FORK_Y = 0.05125          # the two forks are at y = -FORK_Y and +FORK_Y (middle of the pallet openings)
FORK_W = 0.03             # width of one fork
FORK_X0 = 0.075           # back end of the forks (in front of the robot centre)
FORK_X1 = 0.315           # tip of the forks
TAPER_X = 0.255           # the wedge starts here (the load lies on the flat part before it)
FORK_T = 0.015            # thickness of the flat part
TAPER_T = (0.0125, 0.010, 0.007, 0.004)   # thickness of the 4 steps of the wedge, the last one is the tip
FORK_LOW = 0.012          # bottom of the forks above the floor when the lift is down
LIFT_DOWN = 0.0           # positions of fork_lift_joint
LIFT_UP = 0.04            # the forks go 4 cm up: the pallet comes about 2 cm off the floor
LIFT_MIN = -0.002
FULL_IN = 0.21            # robot centre -> pallet centre when the forks are all the way in (1 cm free behind)
REACH = 0.38              # robot radius for Nav2: holds the robot, forks and a load (35 cm), plus the 1-1.5 cm
                          # that another pallet sticks out under its cube (the lidar sees only the cube)
COUNTERWEIGHT = 0.25      # kg, inside the robot at the back

ARM_NS = "/arm"
LIFT_JOINT = "fork_lift_joint"
LIFT_LINK = "fork_lift"

ORANGE = "1 0.5 0 1"
GREY = "0.3 0.3 0.3 1"
DARK = "0.15 0.15 0.15 1"


def _model_z(height_above_floor: float) -> float:
    return height_above_floor - SPAWN_Z


def fork_boxes(lift: float = LIFT_DOWN) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Collision boxes of the forks and the backrest as (centre, size) in the robot frame, heights above
    the floor, for a lift position. The same boxes are used in the SDF and in the tests."""
    boxes = []
    z0 = FORK_LOW + lift
    for s in (-1, 1):
        y = s * FORK_Y
        boxes.append(((0.5 * (FORK_X0 + TAPER_X), y, z0 + FORK_T / 2), (TAPER_X - FORK_X0, FORK_W, FORK_T)))
        step = (FORK_X1 - TAPER_X) / len(TAPER_T)
        for i, t in enumerate(TAPER_T):
            x = TAPER_X + (i + 0.5) * step
            boxes.append(((x, y, z0 + t / 2), (step, FORK_W, t)))
        boxes.append(((FORK_X0 - 0.003, y, z0 + 0.05), (0.006, FORK_W, 0.10)))      # heel: up to the backrest
    boxes.append(((0.066, 0.0, z0 + 0.06), (0.008, 0.16, 0.12)))                       # backrest
    return boxes


def _box_xml(kind: str, name: str, centre, size, colour: str = "") -> str:
    cx, cy, cz = centre
    sx, sy, sz = size
    body = f"""<{kind} name="{name}">
  <pose>{cx - FORK_X0:.4f} {cy:.4f} {cz - FORK_LOW:.4f} 0 0 0</pose>
  <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>"""
    if kind == "visual":
        body += f"\n  <material><ambient>{colour}</ambient><diffuse>{colour}</diffuse></material>"
    else:
        body += "\n  <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>"
    return body + f"\n</{kind}>"


def _lift_link_xml() -> str:
    parts = []
    for i, (c, s) in enumerate(fork_boxes(LIFT_DOWN)):
        parts.append(_box_xml("collision", f"fork_collision_{i}", c, s))
        parts.append(_box_xml("visual", f"fork_visual_{i}", c, s, ORANGE))
    # the carriage that slides on the mast (only a picture)
    parts.append(_box_xml("visual", "carriage_visual", (0.059, 0.0, FORK_LOW + 0.09), (0.008, 0.05, 0.05), ORANGE))
    inner = "\n".join(parts)
    # The origin of the link is the back end of the forks, at their bottom, when the lift is down.
    return f"""<link name="{LIFT_LINK}">
  <pose>{FORK_X0} 0 {_model_z(FORK_LOW):.4f} 0 0 0</pose>
  <gravity>false</gravity>
  <inertial>
    <pose>0.06 0 0.03 0 0 0</pose>
    <mass>0.03</mass>
    <inertia><ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
  </inertial>
{inner}
</link>"""


# A grey mast in front of the robot (only a picture, it is added to base_link). It stays below the lidar.
MAST_VISUAL = f"""<visual name="fork_mast_visual">
  <pose>0.050 0 {_model_z(0.0925):.4f} 0 0 0</pose>
  <geometry><box><size>0.012 0.05 0.165</size></box></geometry>
  <material><ambient>{GREY}</ambient><diffuse>{GREY}</diffuse></material>
</visual>"""

LIFT_JOINT_XML = f"""<joint name="{LIFT_JOINT}" type="prismatic">
  <parent>base_link</parent>
  <child>{LIFT_LINK}</child>
  <axis>
    <xyz>0 0 1</xyz>
    <limit><lower>{LIFT_MIN}</lower><upper>{LIFT_UP}</upper><effort>50</effort><velocity>0.2</velocity></limit>
    <dynamics><damping>5.0</damping><friction>10.0</friction></dynamics>
  </axis>
</joint>"""

COUNTERWEIGHT_LINK_XML = f"""<link name="fork_counterweight">
  <pose>-0.075 0 {_model_z(0.03):.4f} 0 0 0</pose>
  <inertial>
    <mass>{COUNTERWEIGHT}</mass>
    <inertia><ixx>2.2e-4</ixx><iyy>3.2e-5</iyy><izz>2.3e-4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
  </inertial>
  <visual name="counterweight_visual">
    <geometry><box><size>0.03 0.10 0.025</size></box></geometry>
    <material><ambient>{DARK}</ambient><diffuse>{DARK}</diffuse></material>
  </visual>
</link>"""

COUNTERWEIGHT_JOINT_XML = """<joint name="fork_counterweight_joint" type="fixed">
  <parent>base_link</parent>
  <child>fork_counterweight</child>
</joint>"""

LIFT_PLUGIN_XML = f"""<plugin name="fork_lift_control" filename="libgazebo_ros_joint_pose_trajectory.so">
  <ros><namespace>{ARM_NS}</namespace></ros>
  <update_rate>50</update_rate>
</plugin>"""


class ArmModelError(ValueError):
    """The model file is not the one we expect (no <model> or no base_link)."""


def add_arm(sdf_text: str) -> str:
    """Return the SDF text of the robot with the forklift added. Raises ArmModelError."""
    try:
        root = ET.fromstring(sdf_text)
    except ET.ParseError as e:
        raise ArmModelError(f"the robot model is not valid XML: {e}") from e
    model = root if root.tag == "model" else root.find("model")
    if model is None:
        raise ArmModelError("no <model> in the robot file")
    base = next((ln for ln in model.findall("link") if ln.get("name") == "base_link"), None)
    if base is None:
        raise ArmModelError("the robot model has no link 'base_link'")
    if any(ln.get("name") == LIFT_LINK for ln in model.findall("link")):
        return sdf_text                        # the forklift is already there
    base.append(ET.fromstring(MAST_VISUAL))
    for part in (_lift_link_xml(), LIFT_JOINT_XML, COUNTERWEIGHT_LINK_XML, COUNTERWEIGHT_JOINT_XML, LIFT_PLUGIN_XML):
        model.append(ET.fromstring(part))
    return '<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def write_arm_model(source_path: str, target_path: str) -> None:
    """Read the TurtleBot3 model.sdf, add the forklift and write the result to target_path."""
    with open(source_path, encoding="utf-8") as f:
        text = add_arm(f.read())
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(text)


def fork_top(lift: float) -> float:
    """Height of the top of the forks (flat part) above the floor for a lift position."""
    return FORK_LOW + lift + FORK_T
