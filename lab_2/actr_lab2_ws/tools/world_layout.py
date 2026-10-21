"""Layout of the test world. All numbers are in meters.

The Gazebo world, the Nav2 map and config/places.json are generated from this
file by tools/generate_world.py, so they always agree with each other.

Picture (top view, x to the right, y up):

    +----------------+----------------+-----------------------+
    |                |                |                  ///// |
    |   room A       |      hall      |   room B         ///// |  /// = restricted zone
    |   (shelf)      |   (start)      |   (table)              |
    |                |                |                        |
    +----------------+----------------+-----------------------+
    The door of room A is in the upper part of its wall,
    the door of room B is in the lower part of its wall.
"""
from dataclasses import dataclass
import math

# --- general numbers ---------------------------------------------------------
RESOLUTION = 0.05        # map resolution, m per pixel
WALL_T = 0.20            # wall thickness
WALL_H = 1.0             # wall height (only for the picture; the lidar is at 0.18 m)
ROBOT_RADIUS = 0.105     # TurtleBot3 Burger
CUBE_SIZE = 0.22         # as in the first version: the lidar (0.18 m above the floor) sees the cubes
CUBE_MASS = 0.05         # kg, cube and pallet together (light, like an empty cardboard box)
# Every cube stands on a small pallet, so the forks of the robot can go under it (like a real forklift).
# 9 blocks (3 x 3) under a deck: the openings between the blocks go through on all 4 sides.
PALLET_SIZE = 0.24
PALLET_BLOCK = 0.035     # the blocks are 3.5 x 3.5 cm ...
PALLET_BLOCK_H = 0.045   # ... and 4.5 cm high: the openings for the forks are 6.75 cm wide and 4.5 cm high
PALLET_DECK_H = 0.012
PALLET_H = PALLET_BLOCK_H + PALLET_DECK_H
PALLET_BLOCKS = (-0.1025, 0.0, 0.1025)   # x and y of the block centres (the outer blocks are flush with the edge)
MAP_PAD = 0.5            # free margin around the building in the map image

# --- building ------------------------------------------------------------------
X_MIN, X_MAX = -6.5, 7.0
Y_MIN, Y_MAX = -4.0, 4.0
WALL_A_X = -2.5          # wall between room A and the hall
WALL_B_X = 2.5           # wall between the hall and room B
DOOR_A = (0.8, 2.2)      # y range of the opening in the wall of room A (width 1.4)
DOOR_B = (-2.2, -1.0)    # y range of the opening in the wall of room B (width 1.2)


@dataclass(frozen=True)
class Box:
    name: str
    cx: float
    cy: float
    sx: float
    sy: float
    height: float
    color: tuple  # r, g, b, a


GRAY = (0.80, 0.80, 0.80, 1.0)
BROWN = (0.55, 0.35, 0.20, 1.0)


def _walls():
    t = WALL_T
    w = X_MAX - X_MIN
    h = Y_MAX - Y_MIN
    walls = [
        Box("wall_south", (X_MIN + X_MAX) / 2, Y_MIN, w + t, t, WALL_H, GRAY),
        Box("wall_north", (X_MIN + X_MAX) / 2, Y_MAX, w + t, t, WALL_H, GRAY),
        Box("wall_west", X_MIN, 0.0, t, h, WALL_H, GRAY),
        Box("wall_east", X_MAX, 0.0, t, h, WALL_H, GRAY),
    ]
    for tag, x, door in (("a", WALL_A_X, DOOR_A), ("b", WALL_B_X, DOOR_B)):
        lo, hi = door
        walls.append(Box(f"wall_{tag}_south", x, (Y_MIN + lo) / 2, t, lo - Y_MIN, WALL_H, GRAY))
        walls.append(Box(f"wall_{tag}_north", x, (hi + Y_MAX) / 2, t, Y_MAX - hi, WALL_H, GRAY))
    return walls


WALLS = _walls()

OBSTACLES = [
    Box("pillar_hall", -0.3, -1.8, 0.5, 0.5, 1.0, BROWN),
    Box("table_b", 4.6, 0.6, 1.2, 0.6, 0.7, BROWN),
    Box("shelf_a", -4.5, 3.6, 2.0, 0.4, 1.2, BROWN),
]

# --- rooms ---------------------------------------------------------------------
def _rect(x0, x1, y0, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


_in = WALL_T / 2
ROOMS = {
    "room_a": {
        "kind": "room",
        "polygon": _rect(X_MIN + _in, WALL_A_X - _in, Y_MIN + _in, Y_MAX - _in),
        "waypoints": [[-4.5, 0.0, math.pi], [-4.5, -2.2, math.pi]],
        "aliases": ["room a", "room_a", "комната а", "комнату а", "комнате а", "комнаты а"],
        "floor_color": (0.65, 0.90, 0.65, 1.0),
    },
    "hall": {
        "kind": "hall",
        "polygon": _rect(WALL_A_X + _in, WALL_B_X - _in, Y_MIN + _in, Y_MAX - _in),
        "waypoints": [[0.0, 0.0, 0.0], [1.0, -2.8, 0.0]],
        "aliases": ["hall", "холл", "зал", "start", "старт"],
        "floor_color": (0.85, 0.85, 0.85, 1.0),
    },
    "room_b": {
        "kind": "room",
        "polygon": _rect(WALL_B_X + _in, X_MAX - _in, Y_MIN + _in, Y_MAX - _in),
        "waypoints": [[4.4, -1.2, 0.0], [3.6, 2.4, 0.0]],
        "aliases": ["room b", "room_b", "комната б", "комнату б", "комнате б", "комнаты б"],
        "floor_color": (0.65, 0.80, 0.95, 1.0),
    },
}

# A zone where the robot is not allowed to go. It is NOT an obstacle in the map:
# only the ACT-R safety gate keeps the robot out of it.
RESTRICTED = {
    "restricted": {
        "kind": "restricted",
        "polygon": _rect(5.2, 6.9, 2.0, 3.9),
        "waypoints": [[6.0, 3.0, 0.0]],
        "aliases": ["restricted", "restricted area", "restricted zone", "forbidden zone",
                    "запретная зона", "запретную зону", "запретной зоне", "закрытая зона"],
        "floor_color": (1.0, 0.0, 0.0, 0.45),
    }
}

# --- cubes and robot -----------------------------------------------------------
CUBES = {
    "cube_0": {"start": [1.2, 2.0], "color": (1.0, 0.85, 0.0, 1.0)},    # in the hall
    "cube_1": {"start": [5.2, -2.6], "color": (0.9, 0.2, 0.7, 1.0)},    # in room B
}
ROBOT_START = [0.0, 0.0, 0.0]       # x, y, yaw (in the hall)
GOAL_ROOM = "room_a"                # default place where the cubes must be delivered

# --- map image area ------------------------------------------------------------
MAP_X0 = X_MIN - WALL_T / 2 - MAP_PAD
MAP_X1 = X_MAX + WALL_T / 2 + MAP_PAD
MAP_Y0 = Y_MIN - WALL_T / 2 - MAP_PAD
MAP_Y1 = Y_MAX + WALL_T / 2 + MAP_PAD
