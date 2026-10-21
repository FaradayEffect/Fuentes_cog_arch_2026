"""The experiment (a short version), the tracking measures and the table script."""
import importlib.util
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import common
from lqg_nav import experiment as ex
from lqg_nav.metrics import Mission, distance_to_polyline


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout, cls.grid = ex.load_world()
        cls.routes = ex.routes(cls.layout, cls.grid)

    def test_routes_on_the_map(self):
        self.assertEqual(len(self.routes), len(ex.ROUTES))
        for name, start, goal, points in self.routes:
            self.assertEqual(points[0], start[:2], name)
            self.assertEqual(points[-1], goal[:2], name)
            for x, y in points[1:-1]:
                self.assertTrue(self.grid.is_free(x, y, ex.ROBOT_RADIUS), name)

    def test_one_run_of_each_controller(self):
        for c in ex.CONTROLLERS:
            row = ex.run_one(c, self.routes[0], "amcl", 0, self.grid)
            self.assertTrue(row["success"], row)
            self.assertLess(row["rms_cte_cm"], 5.0)
            self.assertLess(row["final_xy_cm"], 10.0)

    def test_same_noise_for_the_three_controllers(self):
        """A paired comparison: with the same seed the measured noise is the same sequence."""
        a = ex.run_one("lqg", self.routes[3], "high", 7, self.grid)
        b = ex.run_one("lqg", self.routes[3], "high", 7, self.grid)
        self.assertEqual(a, b)

    def test_main_writes_the_results(self):
        out = tempfile.mkdtemp()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ex.main(["--seeds", "1", "--out", out]), 0)
        rows = [json.loads(line) for line in common.read_text(os.path.join(out, "runs.jsonl")).splitlines()]
        self.assertEqual(len(rows), len(ex.ROUTES) * len(ex.NOISE) * len(ex.CONTROLLERS))
        self.assertTrue(all(r["success"] for r in rows))
        summary = common.read_text(os.path.join(out, "summary.md"))
        self.assertIn("| lqg |", summary)
        self.assertIn("not Gazebo", summary)
        try:
            import matplotlib  # noqa: F401
            self.assertTrue(os.path.isfile(os.path.join(out, "compare.png")))
        except ImportError:
            pass

    def test_summary_numbers(self):
        rows = [{"noise": "none", "controller": "lqg", "success": True, "rms_cte_cm": 1.0, "max_cte_cm": 2.0,
                 "time_s": 10.0, "final_xy_cm": 1.0, "jerk": 0.5},
                {"noise": "none", "controller": "lqg", "success": False, "rms_cte_cm": 3.0, "max_cte_cm": 4.0,
                 "time_s": 99.0, "final_xy_cm": 50.0, "jerk": 1.5}]
        s = ex.summarize(rows)["none"]["lqg"]
        self.assertEqual((s["runs"], s["success"]), (2, 1))
        self.assertEqual(s["rms_cte_cm"], 2.0)
        self.assertEqual(s["time_s"], 10.0)                # the time only of the missions that succeeded
        self.assertIn("| none (0 cm, 0.0 deg) | lqg | 1/2 |", ex.format_table({"none": {"lqg": s}}))


class MetricsTests(unittest.TestCase):
    def test_distance_to_polyline(self):
        line = [(0, 0), (1, 0), (1, 1)]
        self.assertAlmostEqual(distance_to_polyline(line, 0.5, 0.2), 0.2)
        self.assertAlmostEqual(distance_to_polyline(line, 1.3, 0.5), 0.3)
        self.assertAlmostEqual(distance_to_polyline(line, -1.0, 0.0), 1.0)
        self.assertAlmostEqual(distance_to_polyline([(2, 2)], 2, 3), 1.0)
        self.assertTrue(math.isnan(distance_to_polyline([], 0, 0)))

    def test_mission(self):
        m = Mission("m", 0.0)
        m.sample(0.0, 0.0, 0.1, 0.0, [(0, 0), (1, 0)], "lqg")            # not navigating yet: ignored
        m.start_step(1.0)
        for k in range(10):
            t = 1.0 + 0.1 * k
            m.command(t, 0.2, 0.1 if k % 2 else -0.1)
            m.sample(t, 0.02 * k, 0.05, 0.0, [(0, 0), (1, 0)], "lqg")
        m.end_step(2.0)
        m.command(2.5, 0.0, 5.0)                                          # after the step: ignored
        s = m.summary(3.0, "SUCCESS")
        self.assertEqual((s["controller"], s["nav_steps"], s["nav_time_s"], s["result"]), ("lqg", 1, 1.0, "SUCCESS"))
        self.assertAlmostEqual(s["rms_cte_cm"], 5.0)
        self.assertAlmostEqual(s["jerk"], 2.0)
        self.assertAlmostEqual(s["driven_m"], 0.18)
        self.assertEqual(Mission("x", 0).summary(1.0)["controller"], "unknown")


class TrackingTableTests(unittest.TestCase):
    def load(self):
        path = os.path.join(common.WS, "tools", "tracking_table.py")
        spec = importlib.util.spec_from_file_location("tracking_table", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_table(self):
        mod = self.load()
        rows = [{"controller": "lqg", "result": "SUCCESS", "nav_time_s": 100, "rms_cte_cm": 1, "max_cte_cm": 3,
                 "jerk": 0.5},
                {"controller": "lqg", "result": "FAILED", "nav_time_s": 300, "rms_cte_cm": 3, "max_cte_cm": 5,
                 "jerk": 0.7},
                {"controller": "nav2", "result": "SUCCESS", "nav_time_s": 90, "rms_cte_cm": 4, "max_cte_cm": 9,
                 "jerk": 1.5}]
        text = mod.table(rows)
        self.assertIn("| lqg | 2 | 1 | 100.0 | 2.0 | 4.0 | 0.6 |", text)
        self.assertIn("| nav2 | 1 | 1 | 90.0 | 4.0 | 9.0 | 1.5 |", text)
        path = os.path.join(tempfile.mkdtemp(), "t.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(mod.main(["x", path]), 0)
        self.assertIn("| nav2 |", out.getvalue())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(mod.main(["x", path + ".missing"]), 1)
        mod.GAZEBO_MD = os.path.join(tempfile.mkdtemp(), "gazebo.md")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(mod.main(["x", "--save", path]), 0)
        saved = common.read_text(mod.GAZEBO_MD)
        self.assertIn("| nav2 | 1 | 1 | 90.0 | 4.0 | 9.0 | 1.5 |", saved)
        self.assertIn("track_monitor", saved)

    def test_the_gazebo_file_has_a_table_or_says_how_to_make_it(self):
        text = common.read_text(os.path.join(common.WS, "results", "lqg", "gazebo.md"))
        self.assertTrue("| controller |" in text or "tools/tracking_table.py --save" in text)


if __name__ == "__main__":
    unittest.main()
