"""Files of the package lqg_nav: package.xml, setup.py, the parameters and the three launch files."""
import ast
import glob
import os
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from dataclasses import fields

import yaml

import common
from common import LAB1, PKG, read_text
import fake_launch
from actr_lab1.ros.launch_utils import AGENT_ARGS, SIM_ARGS
from lqg_nav.launch_args import CONTROLLER_ARG, CONTROLLERS, LQG_ARGS, nav_action
from lqg_nav.tracker import TrackerConfig


def py_files():
    out = []
    for folder in ("lqg_nav", "launch", "test"):
        out += glob.glob(os.path.join(PKG, folder, "**", "*.py"), recursive=True)
    return sorted(out + [os.path.join(PKG, "setup.py")])


class PackageFilesTests(unittest.TestCase):
    def test_package_xml_and_setup_py(self):
        root = ET.parse(os.path.join(PKG, "package.xml")).getroot()
        self.assertEqual(root.findtext("name"), "lqg_nav")
        self.assertEqual(root.find("export/build_type").text, "ament_python")
        text = read_text(os.path.join(PKG, "setup.py"))
        self.assertEqual(re.search(r'version="([^"]+)"', text).group(1), root.findtext("version"))
        self.assertIn('package_name = "lqg_nav"', text)
        self.assertTrue(os.path.isfile(os.path.join(PKG, "resource", "lqg_nav")))
        cfg = read_text(os.path.join(PKG, "setup.cfg"))
        self.assertIn("script_dir=$base/lib/lqg_nav", cfg)

    def test_entry_points_exist(self):
        text = read_text(os.path.join(PKG, "setup.py"))
        points = re.findall(r'"(\w+) = ([\w.]+):(\w+)"', text)
        self.assertEqual({p[0] for p in points}, {"lqg_navigator", "track_monitor", "lqg_experiment"})
        for _name, module, func in points:
            path = os.path.join(PKG, *module.split(".")) + ".py"
            tree = ast.parse(read_text(path))
            self.assertIn(func, [n.name for n in tree.body if isinstance(n, ast.FunctionDef)], module)

    def test_every_imported_ros_package_is_a_dependency(self):
        root = ET.parse(os.path.join(PKG, "package.xml")).getroot()
        deps = {e.text for e in root if e.tag in ("exec_depend", "depend")}
        imported = set()
        for path in py_files():
            if os.sep + "test" + os.sep in path:
                continue
            for node in ast.walk(ast.parse(read_text(path))):
                if isinstance(node, ast.Import):
                    imported |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])
        stdlib = set(getattr(sys, "stdlib_module_names", ())) or {
            "argparse", "collections", "dataclasses", "glob", "json", "math", "os", "random", "statistics", "sys",
            "threading", "time", "typing", "__future__"}
        imported = {m for m in imported if m not in stdlib} - {"lqg_nav", "setuptools", "matplotlib", "__future__"}
        for name in sorted(imported):
            self.assertIn(name, deps, f"{name} is imported but not in package.xml")

    def test_old_python_syntax_only(self):
        """ROS 2 Foxy runs Python 3.8."""
        for path in py_files():
            try:
                ast.parse(read_text(path), filename=path, feature_version=(3, 8))
            except SyntaxError as e:
                self.fail(f"{path}: {e}")

    def test_core_does_not_import_ros(self):
        for name in ("lqg", "path", "tracker", "unicycle", "experiment", "metrics", "launch_args"):
            tree = ast.parse(read_text(os.path.join(PKG, "lqg_nav", name + ".py")))
            for node in ast.walk(tree):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else [])
                for m in mods:
                    self.assertNotIn(m.split(".")[0], ("rclpy", "tf2_ros", "nav2_msgs", "geometry_msgs"), name)

    def test_yaml_parameters_are_numbers_and_known(self):
        data = yaml.safe_load(read_text(os.path.join(PKG, "config", "lqg.yaml")))
        params = data["lqg_navigator"]["ros__parameters"]
        known = {f.name for f in fields(TrackerConfig)} | {"rate", "replan_period"}
        for name, value in params.items():
            self.assertIn(name, known, f"lqg.yaml sets '{name}', which the node does not declare")
            self.assertIn(type(value), (int, float), f"{name}: {value} is not a number")
        self.assertNotIn("steering", params)          # a launch argument, not set twice


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self._ctx = fake_launch.install({"lqg_nav": PKG, "actr_lab1": LAB1})
        self._ctx.__enter__()
        self.addCleanup(self._ctx.__exit__, None, None, None)

    def launch(self, name):
        return fake_launch.load_launch_file(os.path.join(PKG, "launch", name)).generate_launch_description()

    def test_controllers(self):
        self.assertEqual(nav_action("lqg"), "lqg/navigate_to_pose")
        self.assertEqual(nav_action(" NAV2 "), "navigate_to_pose")
        with self.assertRaises(ValueError):
            nav_action("dwb")
        self.assertEqual(CONTROLLER_ARG[1], "lqg")
        self.assertEqual(set(CONTROLLERS), {"lqg", "nav2"})

    def test_sim_launch(self):
        d = self.launch("sim.launch.py")
        declared = [e.name for e in d.entities if isinstance(e, fake_launch.DeclareLaunchArgument)]
        self.assertEqual(declared, [a[0] for a in SIM_ARGS + LQG_ARGS])
        include = next(e for e in d.entities if isinstance(e, fake_launch.IncludeLaunchDescription))
        self.assertEqual(include.source.path, os.path.join(LAB1, "launch", "sim.launch.py"))
        self.assertTrue(os.path.isfile(include.source.path))
        self.assertEqual(set(include.arguments), {a[0] for a in SIM_ARGS})
        nodes = {e.executable: e for e in d.entities if isinstance(e, fake_launch.Node)}
        self.assertEqual(set(nodes), {"lqg_navigator", "track_monitor"})
        nav = nodes["lqg_navigator"]
        config, fixed, dynamic = nav.parameters
        self.assertTrue(os.path.isfile(config))
        self.assertIs(fixed["use_sim_time"], True)
        self.assertEqual(dynamic["forklift"].value.name, "arm")
        self.assertIs(dynamic["forklift"].value_type, bool)
        for name, _d, typ, _h in LQG_ARGS:
            self.assertIs(dynamic[name].value_type, typ)
        for node in nodes.values():
            self.assertEqual(node.kwargs["condition"].predicate.name, "nav2")

    def test_agent_launch(self):
        d = self.launch("agent.launch.py")
        declared = [e.name for e in d.entities if isinstance(e, fake_launch.DeclareLaunchArgument)]
        self.assertEqual(declared[0], "controller")
        self.assertEqual(set(declared[1:]), {a[0] for a in AGENT_ARGS} - {"nav_action"})
        func = next(e for e in d.entities if isinstance(e, fake_launch.OpaqueFunction)).function
        for controller, action in (("lqg", "lqg/navigate_to_pose"), ("nav2", "navigate_to_pose")):
            out = func(fake_launch.FakeContext({"controller": controller}))
            include = next(e for e in out if isinstance(e, fake_launch.IncludeLaunchDescription))
            self.assertEqual(include.source.path, os.path.join(LAB1, "launch", "agent.launch.py"))
            self.assertEqual(include.arguments["nav_action"], action)
            self.assertEqual(set(include.arguments), {a[0] for a in AGENT_ARGS})
        with self.assertRaises(ValueError):
            func(fake_launch.FakeContext({"controller": "dwb"}))

    def test_all_launch(self):
        d = self.launch("all.launch.py")
        declared = [e.name for e in d.entities if isinstance(e, fake_launch.DeclareLaunchArgument)]
        self.assertEqual(len(declared), len(set(declared)))
        self.assertIn("controller", declared)
        self.assertNotIn("nav_action", declared)
        includes = [e for e in d.entities if isinstance(e, fake_launch.IncludeLaunchDescription)]
        self.assertEqual([os.path.basename(e.source.path) for e in includes], ["sim.launch.py", "agent.launch.py"])
        self.assertEqual(set(includes[0].arguments) | set(includes[1].arguments), set(declared))

    def test_no_unused_imports_in_launch_files(self):
        for path in glob.glob(os.path.join(PKG, "launch", "*.py")):
            tree = ast.parse(read_text(path))
            imported = {}
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        imported[(a.asname or a.name).split(".")[0]] = node.lineno
            used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            for name, line in imported.items():
                self.assertIn(name, used, f"{os.path.basename(path)}:{line} imports '{name}' but never uses it")


class AgentParameterTests(unittest.TestCase):
    """The only change in the agent of Lab 1: the name of the action that drives the robot."""

    def test_the_agent_has_the_parameter_and_the_robot_uses_it(self):
        names = [a[0] for a in AGENT_ARGS]
        self.assertIn("nav_action", names)
        self.assertEqual(dict((a[0], a[1]) for a in AGENT_ARGS)["nav_action"], "navigate_to_pose")
        text = read_text(os.path.join(LAB1, "actr_lab1", "ros", "agent_node.py"))
        self.assertIn('"nav_action": "navigate_to_pose"', text)
        self.assertIn('nav_action=str(p["nav_action"])', text)
        robot = read_text(os.path.join(LAB1, "actr_lab1", "ros", "ros_robot.py"))
        self.assertIn("ActionClient(node, NavigateToPose, self.cfg.nav_action)", robot)


if __name__ == "__main__":
    unittest.main()
