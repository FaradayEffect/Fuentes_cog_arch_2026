import unittest

import common
from actr_lab1.core.chunks import Chunk
from actr_lab1.core.gate import ActionGate

layout, grid = common.load_world()


def percept(x=0.0, y=0.0, room="hall", holding=None, localized=True, cubes=None):
    cubes = cubes if cubes is not None else (("cube_0", 1.2, 2.0, "hall"), ("cube_1", 5.2, -2.6, "room_b"))
    return Chunk("percept", {"x": x, "y": y, "yaw": 0.0, "room": room, "holding": holding,
                             "localized": localized, "cubes": cubes})


def nav(target, pose, **kw):
    return Chunk("action", {"cmd": "navigate", "target": target, "pose": pose, **kw})


class GateTests(unittest.TestCase):
    def setUp(self):
        self.g = ActionGate(layout, grid, pick_radius=1.1)

    def check(self, action, p=None):
        return self.g.check(action, p or percept())

    def test_navigate_to_a_room_is_fine(self):
        d = self.check(nav("room_a", (-4.5, 0.0, 3.14)))
        self.assertTrue(d.approved, d.reason)

    def test_unknown_command(self):
        d = self.check(Chunk("action", {"cmd": "fly", "target": "room_a"}))
        self.assertFalse(d.approved)
        self.assertEqual(d.rule, "allowed_actions")

    def test_unknown_target(self):
        d = self.check(nav("garage", (0, 0, 0)))
        self.assertEqual(d.rule, "unknown_target")

    def test_no_pose(self):
        self.assertEqual(self.check(nav("room_a", None)).rule, "no_pose")

    def test_forbidden_place(self):
        d = self.check(nav("restricted", (6.0, 3.0, 0.0)))
        self.assertFalse(d.approved)
        self.assertEqual(d.rule, "forbidden_zone")

    def test_pose_inside_forbidden_polygon_even_if_the_place_is_allowed(self):
        d = self.check(nav("room_b", (6.0, 3.0, 0.0)))
        self.assertEqual(d.rule, "forbidden_zone")

    def test_goal_inside_a_wall(self):
        d = self.check(nav("room_a", (-2.5, 3.0, 0.0)))        # the wall between room A and the hall
        self.assertEqual(d.rule, "target_not_free")

    def test_goal_outside_the_map(self):
        self.assertEqual(self.check(nav("room_a", (50.0, 0.0, 0.0))).rule, "target_not_free")

    def test_no_navigation_without_localization(self):         # Seminar 2, homework 1
        d = self.check(nav("room_a", (-4.5, 0.0, 0.0)), percept(localized=False))
        self.assertEqual(d.rule, "localization_lost")

    def test_pick_ok(self):
        d = self.check(Chunk("action", {"cmd": "pick", "target": "cube_0"}), percept(x=0.9, y=1.5))
        self.assertTrue(d.approved, d.reason)

    def test_pick_too_far(self):
        d = self.check(Chunk("action", {"cmd": "pick", "target": "cube_0"}), percept(x=0.0, y=0.0))
        self.assertEqual(d.rule, "too_far")

    def test_pick_while_holding(self):
        d = self.check(Chunk("action", {"cmd": "pick", "target": "cube_0"}), percept(x=0.9, y=1.5, holding="cube_1"))
        self.assertEqual(d.rule, "already_holding")

    def test_pick_something_that_is_not_a_cube(self):
        self.assertEqual(self.check(Chunk("action", {"cmd": "pick", "target": "room_a"})).rule, "unknown_target")

    def test_pick_a_cube_lying_in_the_restricted_zone(self):
        cubes = (("cube_0", 6.0, 3.0, "room_b"), ("cube_1", 5.2, -2.6, "room_b"))
        d = self.check(Chunk("action", {"cmd": "pick", "target": "cube_0"}), percept(x=6.0, y=2.5, cubes=cubes))
        self.assertEqual(d.rule, "forbidden_zone")

    def test_pick_a_cube_nobody_sees(self):
        d = self.check(Chunk("action", {"cmd": "pick", "target": "cube_0"}), percept(cubes=()))
        self.assertEqual(d.rule, "cube_not_seen")

    def test_place_ok(self):
        d = self.check(Chunk("action", {"cmd": "place", "target": "room_a", "drop": (-5.0, 0.0)}),
                       percept(x=-4.5, y=0.0, room="room_a", holding="cube_0"))
        self.assertTrue(d.approved, d.reason)

    def test_place_without_cube(self):
        d = self.check(Chunk("action", {"cmd": "place", "target": "room_a", "drop": (-5.0, 0.0)}),
                       percept(x=-4.5, y=0.0, room="room_a"))
        self.assertEqual(d.rule, "not_holding")

    def test_place_in_the_wrong_room(self):
        d = self.check(Chunk("action", {"cmd": "place", "target": "room_a", "drop": (-5.0, 0.0)}),
                       percept(x=0.0, y=0.0, room="hall", holding="cube_0"))
        self.assertEqual(d.rule, "wrong_room")

    def test_place_without_a_drop_point(self):
        d = self.check(Chunk("action", {"cmd": "place", "target": "room_a", "drop": None}),
                       percept(x=-4.5, y=0.0, room="room_a", holding="cube_0"))
        self.assertEqual(d.rule, "no_drop_point")

    def test_place_in_the_restricted_place(self):
        d = self.check(Chunk("action", {"cmd": "place", "target": "restricted", "drop": (6.0, 3.0)}),
                       percept(x=6.0, y=3.0, room="room_b", holding="cube_0"))
        self.assertEqual(d.rule, "forbidden_zone")

    def test_stop_is_never_refused(self):
        d = self.check(Chunk("action", {"cmd": "stop"}), percept(localized=False))
        self.assertTrue(d.approved)

    def test_action_budget(self):
        g = ActionGate(layout, grid, max_actions=3)
        ok = [g.check(nav("room_a", (-4.5, 0.0, 0.0)), percept()).approved for _ in range(5)]
        self.assertEqual(ok, [True, True, True, False, False])
        self.assertEqual(g.denials, 2)

    def test_denials_are_counted(self):
        self.check(nav("garage", (0, 0, 0)))
        self.check(nav("restricted", (6.0, 3.0, 0.0)))
        self.assertEqual(self.g.denials, 2)


if __name__ == "__main__":
    unittest.main()
