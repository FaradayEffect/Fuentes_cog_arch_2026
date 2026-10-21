import unittest

import common
from actr_lab1.core.nlp import Task, expected_deliveries, parse_mission, resolve_name
from actr_lab1.core.plan import PlanStep, PlanValidator

layout, grid = common.load_world()


def D(cube, room):
    return Task("deliver", room, cube)


class MissionParsingTests(unittest.TestCase):
    def test_main_mission_russian(self):
        t = parse_mission("Отвези кубик 0 и кубик 1 в комнату А.", layout)
        self.assertEqual(t, [D("cube_0", "room_a"), D("cube_1", "room_a")])

    def test_english(self):
        self.assertEqual(parse_mission("Bring cube 1 to room A.", layout), [D("cube_1", "room_a")])

    def test_cyrillic_and_latin_room_letters(self):
        self.assertEqual(parse_mission("Отвези кубик 0 в комнату Б", layout), [D("cube_0", "room_b")])
        self.assertEqual(parse_mission("take cube 0 to room B", layout), [D("cube_0", "room_b")])

    def test_two_wishes_with_different_rooms(self):
        t = parse_mission("Cube 0 to room B, then cube 1 to room A", layout)
        self.assertEqual(t, [D("cube_0", "room_b"), D("cube_1", "room_a")])

    def test_two_pairs_in_one_sentence(self):
        self.assertEqual(parse_mission("Take cube 0 to room B and cube 1 to room A", layout),
                         [D("cube_0", "room_b"), D("cube_1", "room_a")])
        self.assertEqual(parse_mission("Отвези кубик 0 в комнату Б и кубик 1 в комнату А.", layout),
                         [D("cube_0", "room_b"), D("cube_1", "room_a")])
        self.assertEqual(parse_mission("Отвези кубик 1 в комнату А, а кубик 0 в комнату Б", layout),
                         [D("cube_1", "room_a"), D("cube_0", "room_b")])

    def test_one_destination_for_two_cubes_is_still_one_piece(self):
        self.assertEqual(parse_mission("Bring cube 0 and cube 1 to room B", layout),
                         [D("cube_0", "room_b"), D("cube_1", "room_b")])

    def test_professor_example(self):
        text = "Доедь до холла, возьми кубик 0, отвези в комнату А, доедь в комнату Б, возьми кубик 1, отвези в комнату А"
        t = parse_mission(text, layout)
        self.assertEqual(t, [Task("visit", "hall"), D("cube_0", "room_a"), Task("visit", "room_b"),
                             D("cube_1", "room_a")])

    def test_both_cubes(self):
        self.assertEqual(parse_mission("Take both cubes to room A", layout),
                         [D("cube_0", "room_a"), D("cube_1", "room_a")])
        self.assertEqual(parse_mission("Отвези оба кубика в комнату А", layout),
                         [D("cube_0", "room_a"), D("cube_1", "room_a")])

    def test_list_of_numbers(self):
        self.assertEqual(parse_mission("Отвези кубик 0 и 1 в комнату А", layout),
                         [D("cube_0", "room_a"), D("cube_1", "room_a")])

    def test_place_before_the_cube_is_a_stop(self):
        t = parse_mission("Go to the restricted area and bring cube 0 to room A", layout)
        self.assertEqual(t, [Task("visit", "restricted"), D("cube_0", "room_a")])

    def test_visit_only(self):
        self.assertEqual(parse_mission("Go to the hall.", layout), [Task("visit", "hall")])
        self.assertEqual(parse_mission("Поезжай в запретную зону", layout), [Task("visit", "restricted")])

    def test_unknown_room_is_kept_so_validation_can_complain(self):
        self.assertEqual(parse_mission("Привези кубик 0 в комнату В", layout), [D("cube_0", "room_c")])

    def test_default_room_when_none_is_given(self):
        self.assertEqual(parse_mission("Bring cube 1", layout), [D("cube_1", layout.goal_room)])

    def test_nonsense_gives_nothing(self):
        self.assertEqual(parse_mission("Hello robot", layout), [])
        self.assertEqual(parse_mission("", layout), [])

    def test_expected_deliveries(self):
        t = parse_mission("Cube 0 to room B, then go to the hall", layout)
        self.assertEqual(expected_deliveries(t), (("cube_0", "room_b"),))

    def test_resolve_name(self):
        for text, name in [("Room A", "room_a"), ("room_a", "room_a"), ("комната А", "room_a"),
                           ("Комнату Б", "room_b"), ("cube0", "cube_0"), ("Cube 1", "cube_1"),
                           ("кубик 1", "cube_1"), ("cube_0", "cube_0"), ("hall", "hall"),
                           ("restricted", "restricted")]:
            self.assertEqual(resolve_name(text, layout), name, text)
        self.assertIsNone(resolve_name("garage", layout))
        self.assertIsNone(resolve_name("cube 0 room a", layout))
        self.assertIsNone(resolve_name(5, layout))


def steps(*pairs):
    return {"steps": [{"action": a, "target": t} for a, t in pairs]}


GOOD = steps(("navigate", "cube_0"), ("pick", "cube_0"), ("navigate", "room_a"), ("place", "room_a"))
START = {"holding": None, "robot_place": "hall", "cube_places": {"cube_0": "hall", "cube_1": "room_b"}}


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.v = PlanValidator(layout)

    def test_good_plan(self):
        r = self.v.validate(GOOD, START, expected=[("cube_0", "room_a")])
        self.assertTrue(r.valid, r.errors)
        self.assertEqual(r.steps[0], PlanStep("navigate", "cube_0"))
        self.assertEqual(r.warnings, ())

    def test_not_a_dict(self):
        self.assertFalse(self.v.validate("hello", START).valid)
        self.assertFalse(self.v.validate({"plan": []}, START).valid)
        self.assertFalse(self.v.validate({"steps": "x"}, START).valid)

    def test_empty_plan(self):
        r = self.v.validate({"steps": []}, START)
        self.assertFalse(r.valid)
        self.assertIn("no steps", r.errors[0])

    def test_empty_plan_is_fine_when_the_goal_is_already_true(self):
        state = {"holding": None, "robot_place": "hall", "cube_places": {"cube_0": "room_a"}}
        self.assertTrue(self.v.validate({"steps": []}, state, expected=[("cube_0", "room_a")]).valid)
        # but an empty plan is wrong when the goal is not reached yet
        self.assertFalse(self.v.validate({"steps": []}, START, expected=[("cube_0", "room_a")]).valid)

    def test_action_not_allowed(self):
        r = self.v.validate(steps(("fly", "room_a")), START)
        self.assertFalse(r.valid)
        self.assertIn("not allowed", r.errors[0])

    def test_synonyms_are_accepted(self):
        r = self.v.validate(steps(("go", "cube_0"), ("take", "cube_0"), ("move_to", "room_a"), ("drop", "room_a")), START)
        self.assertTrue(r.valid, r.errors)
        self.assertEqual([s.action for s in r.steps], ["navigate", "pick", "navigate", "place"])

    def test_unknown_target(self):
        r = self.v.validate(steps(("navigate", "room_c")), START)
        self.assertFalse(r.valid)
        self.assertIn("unknown target", r.errors[0])
        self.assertIn("room_a", r.errors[0])          # the message lists the known names

    def test_names_are_normalised(self):
        r = self.v.validate(steps(("navigate", "Cube 0"), ("pick", "cube0"), ("navigate", "комната А"),
                                  ("place", "Room A")), START)
        self.assertTrue(r.valid, r.errors)
        self.assertEqual([s.target for s in r.steps], ["cube_0", "cube_0", "room_a", "room_a"])

    def test_text_steps(self):
        r = self.v.validate({"steps": ["navigate(cube_0)", "pick(cube_0)"]}, START)
        self.assertTrue(r.valid, r.errors)
        self.assertFalse(self.v.validate({"steps": ["go there"]}, START).valid)

    def test_pick_needs_a_cube(self):
        r = self.v.validate(steps(("pick", "room_a")), START)
        self.assertFalse(r.valid)

    def test_robot_carries_one_cube(self):
        r = self.v.validate(steps(("pick", "cube_0"), ("pick", "cube_1")), START)
        self.assertFalse(r.valid)
        self.assertIn("already holds cube_0", " ".join(r.errors))

    def test_place_without_cube(self):
        r = self.v.validate(steps(("navigate", "room_a"), ("place", "room_a")), START)
        self.assertFalse(r.valid)
        self.assertIn("holds no cube", " ".join(r.errors))

    def test_initial_holding_is_used(self):
        state = dict(START, holding="cube_0")
        r = self.v.validate(steps(("navigate", "room_a"), ("place", "room_a")), state)
        self.assertTrue(r.valid, r.errors)

    def test_missing_navigate_is_only_a_warning(self):
        r = self.v.validate(steps(("pick", "cube_0"), ("place", "room_a")), START)
        self.assertTrue(r.valid)
        self.assertEqual(len(r.warnings), 2)

    def test_forbidden_place_is_not_rejected_here(self):         # the gate refuses it later
        r = self.v.validate(steps(("navigate", "restricted")), START)
        self.assertTrue(r.valid)

    def test_coverage(self):
        r = self.v.validate(GOOD, START, expected=[("cube_0", "room_a"), ("cube_1", "room_a")])
        self.assertFalse(r.valid)
        self.assertIn("does not deliver cube_1 to room_a", r.errors[0])
        r = self.v.validate(GOOD, START, expected=[("cube_0", "room_b")])
        self.assertFalse(r.valid)

    def test_too_many_steps(self):
        r = self.v.validate({"steps": [{"action": "navigate", "target": "hall"}] * 40}, START)
        self.assertFalse(r.valid)

    def test_ends_holding_is_a_warning(self):
        r = self.v.validate(steps(("navigate", "cube_0"), ("pick", "cube_0")), START)
        self.assertTrue(r.valid)
        self.assertTrue(any("still holds" in w for w in r.warnings))

    def test_alternative_target_keys(self):
        r = self.v.validate({"steps": [{"action": "navigate", "room": "room_a"}]}, START)
        self.assertTrue(r.valid, r.errors)


if __name__ == "__main__":
    unittest.main()
