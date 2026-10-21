import json
import math
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import common
from actr_lab1.core.world import (OccupancyGrid, Pose2D, approach_candidates, dist, find_drop_point,
                                  point_in_polygon)

layout, grid = common.load_world()
ROOT = os.path.dirname(os.path.dirname(common.PKG_ROOT))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import generate_world  # noqa: E402
import world_layout  # noqa: E402

CLEAR = layout.robot_radius + 0.08


class GeometryTests(unittest.TestCase):
    def test_point_in_polygon(self):
        sq = [[0, 0], [2, 0], [2, 2], [0, 2]]
        self.assertTrue(point_in_polygon(1, 1, sq))
        self.assertFalse(point_in_polygon(3, 1, sq))
        self.assertFalse(point_in_polygon(1, -0.1, sq))
        l_shape = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
        self.assertTrue(point_in_polygon(0.5, 1.5, l_shape))
        self.assertFalse(point_in_polygon(1.5, 1.5, l_shape))

    def test_rooms_and_zones(self):
        self.assertEqual(layout.room_at(0, 0), "hall")
        self.assertEqual(layout.room_at(-4.5, 0), "room_a")
        self.assertEqual(layout.room_at(4.5, -1), "room_b")
        self.assertEqual(layout.room_at(-2.5, 3.0), "")            # inside a wall
        self.assertTrue(layout.in_forbidden(6.0, 3.0))
        self.assertFalse(layout.in_forbidden(4.4, -1.2))

    def test_waypoint_variants_wrap_around(self):
        self.assertEqual(layout.waypoint("room_a", 0), layout.waypoint("room_a", 2))
        self.assertNotEqual(layout.waypoint("room_a", 0), layout.waypoint("room_a", 1))
        self.assertIsNone(layout.waypoint("garage"))


class MapTests(unittest.TestCase):
    def test_map_matches_the_layout(self):
        self.assertAlmostEqual(grid.res, world_layout.RESOLUTION)
        self.assertEqual(grid.width, int(round((world_layout.MAP_X1 - world_layout.MAP_X0) / grid.res)))
        self.assertEqual(grid.height, int(round((world_layout.MAP_Y1 - world_layout.MAP_Y0) / grid.res)))
        self.assertAlmostEqual(grid.ox, world_layout.MAP_X0, places=2)
        self.assertAlmostEqual(grid.oy, world_layout.MAP_Y0, places=2)

    def test_walls_and_obstacles_are_occupied(self):
        for b in world_layout.WALLS + world_layout.OBSTACLES:
            self.assertFalse(grid.is_free(b.cx, b.cy), b.name)

    def test_open_floor_is_free(self):
        for xy in [(0, 0), (-4.5, 0), (4.4, -1.2), (6.0, 3.0)]:
            self.assertTrue(grid.is_free(*xy, radius=CLEAR), xy)

    def test_outside_is_not_free(self):
        self.assertFalse(grid.is_free(-7.0, 0.0))
        self.assertFalse(grid.is_free(100, 100))

    def test_start_pose_is_free(self):
        self.assertTrue(grid.is_free(*layout.robot_start.xy(), radius=CLEAR))

    def test_waypoints_are_inside_their_place_and_free(self):
        for name, place in layout.places.items():
            for wp in place.waypoints:
                self.assertTrue(place.contains(wp.x, wp.y), (name, wp))
                self.assertTrue(grid.is_free(wp.x, wp.y, CLEAR), (name, wp))

    def test_cubes_start_on_free_floor_in_a_room(self):
        for name, c in layout.cubes.items():
            self.assertTrue(grid.is_free(*c.start, radius=layout.cube_size / 2), name)
            self.assertNotEqual(layout.room_at(*c.start), "", name)
            self.assertFalse(layout.in_forbidden(*c.start), name)

    def test_restricted_zone_is_not_an_obstacle(self):          # only the gate keeps the robot out
        for wp in layout.places["restricted"].waypoints:
            self.assertTrue(grid.is_free(wp.x, wp.y, CLEAR))

    def test_doors_are_wide_enough(self):
        # walk along each interior wall and measure the free gap that is wide enough for the robot
        for wall_x, (lo, hi) in ((world_layout.WALL_A_X, world_layout.DOOR_A), (world_layout.WALL_B_X, world_layout.DOOR_B)):
            self.assertGreaterEqual(hi - lo, 1.2)
            mid = (lo + hi) / 2
            self.assertTrue(grid.is_free(wall_x, mid, CLEAR))
            self.assertFalse(grid.is_free(wall_x, hi + 0.3))     # the wall is there above the door
            self.assertFalse(grid.is_free(wall_x, lo - 0.3))


class ReachabilityTests(unittest.TestCase):
    def test_every_waypoint_is_reachable_from_the_start(self):
        for name, place in layout.places.items():
            for i, wp in enumerate(place.waypoints):
                path = grid.astar(layout.robot_start.xy(), wp.xy(), CLEAR)
                self.assertIsNotNone(path, f"{name} {i}")

    def test_every_waypoint_is_reachable_from_every_room(self):
        starts = [p.waypoints[0] for p in layout.places.values()]
        for a in starts:
            for name, place in layout.places.items():
                for wp in place.waypoints:
                    self.assertIsNotNone(grid.astar(a.xy(), wp.xy(), CLEAR), (a, name, wp))

    def test_paths_between_rooms_go_through_the_doors(self):
        path = grid.astar((-4.5, 0.0), (4.4, -1.2), CLEAR)
        self.assertIsNotNone(path)
        self.assertTrue(any(abs(x - world_layout.WALL_A_X) < 0.2 and world_layout.DOOR_A[0] < y < world_layout.DOOR_A[1]
                            for x, y in path))
        self.assertTrue(any(abs(x - world_layout.WALL_B_X) < 0.2 and world_layout.DOOR_B[0] < y < world_layout.DOOR_B[1]
                            for x, y in path))

    def test_a_goal_in_a_wall_has_no_path_to_the_wall_itself(self):
        path = grid.astar((0, 0), (-2.5, 3.0), CLEAR)
        # the planner moves the goal to the nearest free cell; it must not end inside the wall
        if path is not None:
            self.assertTrue(grid.is_free(path[-2][0], path[-2][1], CLEAR))

    def test_closed_door_means_no_path(self):
        # copy the grid and close door A with a wall: room A must become unreachable
        blocked = bytearray(grid.blocked)
        for iy in range(grid.height):
            for ix in range(grid.width):
                x, y = grid.to_world(ix, iy)
                if abs(x - world_layout.WALL_A_X) < 0.12 and world_layout.DOOR_A[0] - 0.05 < y < world_layout.DOOR_A[1] + 0.05:
                    blocked[iy * grid.width + ix] = 1
        closed = OccupancyGrid(grid.width, grid.height, grid.res, (grid.ox, grid.oy), blocked, grid.unknown)
        self.assertIsNone(closed.astar((0, 0), (-4.5, 0.0), CLEAR))
        self.assertIsNotNone(closed.astar((0, 0), (4.4, -1.2), CLEAR))


class GroundingTests(unittest.TestCase):
    def test_approach_poses_face_the_cube_and_are_free(self):
        for name, c in layout.cubes.items():
            cands = approach_candidates(grid, layout, c.start, layout.robot_start.xy())
            self.assertGreaterEqual(len(cands), 4, name)
            for p in cands:
                self.assertTrue(grid.is_free(p.x, p.y, CLEAR))
                self.assertTrue(0.5 < dist(p.xy(), c.start) < 0.66)
                heading = math.atan2(c.start[1] - p.y, c.start[0] - p.x)
                self.assertAlmostEqual(math.cos(heading - p.yaw), 1.0, places=6)
            d = [dist(p.xy(), layout.robot_start.xy()) for p in cands]
            self.assertEqual(d, sorted(d))                        # closest first
            for p in cands:
                self.assertIsNotNone(grid.astar(layout.robot_start.xy(), p.xy(), CLEAR), (name, p))

    def test_drop_point_is_in_the_room_and_free(self):
        wp = layout.waypoint("room_a")
        drop = find_drop_point(grid, layout, wp, "room_a", [])
        self.assertIsNotNone(drop)
        self.assertTrue(layout.places["room_a"].contains(*drop))
        self.assertTrue(grid.is_free(*drop, layout.cube_size / 2 + 0.05))

    def test_two_cubes_do_not_land_on_each_other(self):
        wp = layout.waypoint("room_a")
        first = find_drop_point(grid, layout, wp, "room_a", [])
        second = find_drop_point(grid, layout, wp, "room_a", [first])
        self.assertIsNotNone(second)
        self.assertGreaterEqual(dist(first, second), 0.40)

    def test_no_drop_point_outside_the_room(self):
        self.assertIsNone(find_drop_point(grid, layout, Pose2D(0, 0, 0), "room_a", []))

    def test_no_drop_point_for_an_unknown_room(self):
        self.assertIsNone(find_drop_point(grid, layout, Pose2D(0, 0, 0), "garage", []))

    def test_drop_point_never_in_the_forbidden_zone(self):
        drop = find_drop_point(grid, layout, Pose2D(5.4, 2.6, 0.0), "room_b", [])
        if drop is not None:
            self.assertFalse(layout.in_forbidden(*drop))


class GeneratedFilesTests(unittest.TestCase):
    def test_generated_files_are_up_to_date(self):
        """The files in the package must be exactly what tools/generate_world.py makes now."""
        with tempfile.TemporaryDirectory() as tmp:
            made, _size = generate_world.generate(tmp)
            real = generate_world.paths(common.PKG_ROOT)
            for key in ("world", "pgm", "yaml", "places"):
                with open(made[key], "rb") as a, open(real[key], "rb") as b:
                    self.assertEqual(a.read(), b.read(), f"{key} is out of date: run tools/generate_world.py")

    def test_world_file_is_valid_sdf_xml(self):
        root = ET.parse(os.path.join(common.PKG_ROOT, "worlds", "rooms.world")).getroot()
        self.assertEqual(root.tag, "sdf")
        world = root.find("world")
        models = {m.get("name"): m for m in world.findall("model")}
        for name in ["building", "obstacles", "floor_marks"] + list(layout.cubes):
            self.assertIn(name, models)
        for name in layout.cubes:                                  # cubes must move: not static
            self.assertIsNone(models[name].find("static"))
        self.assertEqual(models["building"].find("static").text, "true")
        plugin = world.find("plugin")
        self.assertEqual(plugin.get("filename"), "libgazebo_ros_state.so")
        self.assertEqual(plugin.find("ros/namespace").text, "/gazebo")

    def test_cube_poses_in_the_world_match_places_json(self):
        root = ET.parse(os.path.join(common.PKG_ROOT, "worlds", "rooms.world")).getroot()
        for m in root.find("world").findall("model"):
            if m.get("name") in layout.cubes:
                x, y, z = [float(v) for v in m.find("pose").text.split()[:3]]
                self.assertAlmostEqual(x, layout.cubes[m.get("name")].start[0])
                self.assertAlmostEqual(y, layout.cubes[m.get("name")].start[1])
                self.assertAlmostEqual(z, layout.cube_rest_z, delta=0.002)       # on its pallet, on the floor

    def test_wall_boxes_in_the_world_match_the_layout(self):
        root = ET.parse(os.path.join(common.PKG_ROOT, "worlds", "rooms.world")).getroot()
        building = [m for m in root.find("world").findall("model") if m.get("name") == "building"][0]
        cols = building.findall("link/collision")
        self.assertEqual(len(cols), len(world_layout.WALLS))

    def test_places_json_is_complete(self):
        with open(os.path.join(common.PKG_ROOT, "config", "places.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(set(data["places"]), {"hall", "room_a", "room_b", "restricted"})
        self.assertEqual(set(data["cubes"]), {"cube_0", "cube_1"})
        self.assertEqual(data["forbidden"], ["restricted"])
        self.assertTrue(os.path.exists(layout.map_yaml_path()))

    def test_map_yaml_is_what_nav2_expects(self):
        with open(os.path.join(common.PKG_ROOT, "maps", "rooms.yaml")) as f:
            text = f.read()
        for key in ("image: rooms.pgm", "resolution: 0.05", "origin:", "negate: 0", "occupied_thresh: 0.65",
                    "free_thresh: 0.196"):
            self.assertIn(key, text)
        with open(os.path.join(common.PKG_ROOT, "maps", "rooms.pgm"), "rb") as f:
            self.assertTrue(f.read(2) == b"P5")


if __name__ == "__main__":
    unittest.main()
