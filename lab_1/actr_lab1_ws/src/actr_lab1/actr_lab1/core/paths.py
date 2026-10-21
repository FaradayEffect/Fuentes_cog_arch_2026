"""Where are the data files (config/places.json, maps/, worlds/)?

Three cases:
  * you run from the source folder (python3 -m actr_lab1.demo, unit tests),
  * the package is installed with colcon (the files are in install/actr_lab1/share/actr_lab1),
  * you point to a folder yourself with the environment variable ACTR_LAB1_HOME.
"""
from __future__ import annotations

import os
from typing import Tuple

from .world import Layout, OccupancyGrid

PACKAGE = "actr_lab1"


def _has_data(folder: str) -> bool:
    return bool(folder) and os.path.isfile(os.path.join(folder, "config", "places.json"))


def share_dir() -> str:
    env = os.environ.get("ACTR_LAB1_HOME", "")
    if _has_data(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))                  # .../actr_lab1/core
    source_root = os.path.dirname(os.path.dirname(here))               # .../src/actr_lab1
    if _has_data(source_root):
        return source_root
    try:
        from ament_index_python.packages import get_package_share_directory
        folder = get_package_share_directory(PACKAGE)
        if _has_data(folder):
            return folder
    except Exception:           # no ROS, or the package is not installed
        pass
    raise FileNotFoundError(
        "cannot find config/places.json. Run from the source folder, build the package with colcon "
        "and source install/setup.bash, or set ACTR_LAB1_HOME to the folder that has config/ and maps/.")


def places_file(folder: str = "") -> str:
    return os.path.join(folder or share_dir(), "config", "places.json")


def load_world(folder: str = "") -> Tuple[Layout, OccupancyGrid]:
    """Layout (names, rooms, cubes) and the occupancy grid (the same map that Nav2 uses)."""
    layout = Layout.load(places_file(folder))
    grid = OccupancyGrid.from_yaml(layout.map_yaml_path())
    return layout, grid
