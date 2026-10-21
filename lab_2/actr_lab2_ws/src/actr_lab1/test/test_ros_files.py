"""Checks of the package files, the launch files and the Nav2 parameter patch. No ROS needed.

The launch files are executed with a fake of the `launch` library (fake_launch.py): they really run,
but the objects they make are simple recorders.
"""
import ast
import glob
import importlib.util
import math
import os
import re
import sys
import sysconfig
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

try:
    import yaml
except ImportError:                  # PyYAML is part of every ROS 2 installation, but not of every Python
    raise unittest.SkipTest("PyYAML is not installed (pip install pyyaml), so the ROS file checks are skipped")

import common  # noqa: F401
import fake_launch
import fake_ros
from actr_lab1.core.world import Pose2D
from actr_lab1.ros import launch_utils, nav2_params


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def is_stdlib(name):
    """True for modules of the Python standard library (sys.stdlib_module_names exists only from 3.10 on)."""
    names = getattr(sys, "stdlib_module_names", None)
    if names is not None:
        return name in names
    if name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    origin = getattr(spec, "origin", None) or ""
    if origin in ("built-in", "frozen"):
        return True
    return origin.startswith(sysconfig.get_paths()["stdlib"] + os.sep) and "-packages" not in origin


PKG = common.PKG_ROOT
layout, grid = common.load_world()

NAV2_SAMPLE = """
amcl:
  ros__parameters:
    use_sim_time: False
    base_frame_id: "base_footprint"
    laser_min_range: -1.0
    set_initial_pose: false
    initial_pose: {x: 0.0, y: 0.0, z: 0.0, yaw: 0.0}
bt_navigator:
  ros__parameters:
    use_sim_time: False
local_costmap:
  local_costmap:
    ros__parameters:
      use_sim_time: False
      robot_radius: 0.22
      plugins: ["voxel_layer", "inflation_layer"]
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        observation_sources: scan
        scan:
          topic: /scan
          data_type: "LaserScan"
          obstacle_min_range: 0.0
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 1.0
global_costmap:
  global_costmap:
    ros__parameters:
      use_sim_time: False
      robot_radius: 0.22
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        observation_sources: scan
        scan:
          topic: /scan
          data_type: "LaserScan"
      inflation_layer:
        cost_scaling_factor: 3.0
        inflation_radius: 1.0
"""


def py_files():
    out = []
    for folder in ("actr_lab1", "launch", "test"):
        out += glob.glob(os.path.join(PKG, folder, "**", "*.py"), recursive=True)
    out.append(os.path.join(PKG, "setup.py"))
    return sorted(out)


class PackageFilesTests(unittest.TestCase):
    def test_package_xml(self):
        root = ET.parse(os.path.join(PKG, "package.xml")).getroot()
        self.assertEqual(root.tag, "package")
        self.assertEqual(root.get("format"), "3")
        self.assertEqual(root.findtext("name"), "actr_lab1")
        self.assertEqual(root.find("export/build_type").text, "ament_python")
        self.assertTrue(root.findtext("version"))
        self.assertTrue(root.findtext("license"))

    def test_setup_py_matches_package_xml(self):
        text = read_text(os.path.join(PKG, "setup.py"))
        version = re.search(r'version="([^"]+)"', text).group(1)
        root = ET.parse(os.path.join(PKG, "package.xml")).getroot()
        self.assertEqual(version, root.findtext("version"))
        self.assertIn('package_name = "actr_lab1"', text)

    def test_resource_marker_and_setup_cfg(self):
        self.assertTrue(os.path.isfile(os.path.join(PKG, "resource", "actr_lab1")))
        cfg = read_text(os.path.join(PKG, "setup.cfg"))
        self.assertIn("script_dir=$base/lib/actr_lab1", cfg)
        self.assertIn("install_scripts=$base/lib/actr_lab1", cfg)

    def test_entry_points_exist(self):
        text = read_text(os.path.join(PKG, "setup.py"))
        points = re.findall(r'"(\w+) = ([\w.]+):(\w+)"', text)
        self.assertGreaterEqual(len(points), 4)
        for script, module, func in points:
            path = os.path.join(PKG, *module.split(".")) + ".py"
            self.assertTrue(os.path.isfile(path), f"{script}: {path} does not exist")
            tree = ast.parse(read_text(path))
            names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            self.assertIn(func, names, f"{script}: {module} has no function {func}")

    def test_data_files_exist(self):
        for folder, pattern in (("launch", "*.py"), ("config", "*"), ("worlds", "*"), ("maps", "*")):
            files = glob.glob(os.path.join(PKG, folder, pattern))
            self.assertTrue(files, f"nothing to install from {folder}/")
        for needed in ("launch/sim.launch.py", "launch/agent.launch.py", "launch/all.launch.py",
                       "config/places.json", "config/agent.yaml", "worlds/rooms.world", "maps/rooms.yaml",
                       "maps/rooms.pgm"):
            self.assertTrue(os.path.isfile(os.path.join(PKG, needed)), needed)

    def test_every_imported_ros_package_is_a_dependency(self):
        root = ET.parse(os.path.join(PKG, "package.xml")).getroot()
        deps = {e.text for e in root if e.tag in ("exec_depend", "depend", "build_depend")}
        deps.add("python3-yaml")
        ros_roots = set()
        for path in py_files():
            if os.sep + "test" + os.sep in path:
                continue
            tree = ast.parse(read_text(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    ros_roots |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    ros_roots.add(node.module.split(".")[0])
        ros_roots = {r for r in ros_roots if not is_stdlib(r)}
        ros_roots -= {"actr_lab1", "setuptools", "matplotlib", "numpy"}
        mapping = {"yaml": "python3-yaml"}
        for name in sorted(ros_roots):
            self.assertIn(mapping.get(name, name), deps, f"{name} is imported but not in package.xml")

    def test_no_unused_imports_in_launch_files(self):
        for path in glob.glob(os.path.join(PKG, "launch", "*.py")):
            tree = ast.parse(read_text(path))
            imported = {}
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imported[(a.asname or a.name).split(".")[0]] = node.lineno
                elif isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        imported[a.asname or a.name] = node.lineno
            used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
                   {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
            for name, line in imported.items():
                self.assertIn(name, used, f"{os.path.basename(path)}:{line} imports '{name}' but never uses it")

    def test_old_python_syntax_only(self):
        """ROS 2 Foxy runs Python 3.8: no match statements, no `X | Y` types in code, etc."""
        for path in py_files():
            with open(path, encoding="utf-8") as f:
                try:
                    ast.parse(f.read(), filename=path, feature_version=(3, 8))
                except SyntaxError as e:
                    self.fail(f"{path}: {e}")

    def test_core_does_not_import_ros(self):
        ros = {"rclpy", "launch", "launch_ros", "ament_index_python", "geometry_msgs", "nav2_msgs", "gazebo_msgs",
               "tf2_ros", "std_msgs", "action_msgs", "actr_lab1.ros"}
        for path in glob.glob(os.path.join(PKG, "actr_lab1", "core", "*.py")):
            tree = ast.parse(read_text(path))
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module]
                for m in mods:
                    root = m.split(".")[0]
                    if root in ros and root != "ament_index_python":
                        self.fail(f"{path} imports {m}")
                    if root == "ament_index_python":
                        # allowed only inside a function with try/except (paths.py)
                        self.assertTrue(path.endswith("paths.py"), f"{path} imports {m}")


class WorldFileTests(unittest.TestCase):
    """The Gazebo world must load without Gazebo's own model folder and without the internet."""

    def setUp(self):
        self.root = ET.parse(os.path.join(PKG, "worlds", "rooms.world")).getroot()
        self.world = self.root.find("world")

    def test_no_include_of_external_models(self):
        text = read_text(os.path.join(PKG, "worlds", "rooms.world"))
        self.assertNotIn("model://", text)
        self.assertEqual(list(self.world.iter("include")), [])

    def test_ground_and_sun_are_written_in_the_file(self):
        models = {m.get("name"): m for m in self.world.findall("model")}
        ground = models["ground_plane"]
        self.assertEqual(ground.findtext("static"), "true")
        self.assertIsNotNone(ground.find("link/collision/geometry/plane"))          # the robot must not fall
        self.assertEqual(ground.findtext("link/collision/geometry/plane/normal"), "0 0 1")
        sun = next(l for l in self.world.findall("light") if l.get("name") == "sun")
        self.assertEqual(sun.get("type"), "directional")
        self.assertIsNotNone(sun.find("direction"))

    def test_the_state_plugin_and_the_cubes_are_there(self):
        plugin = self.world.find("plugin")
        self.assertEqual(plugin.get("filename"), "libgazebo_ros_state.so")
        self.assertEqual(plugin.findtext("ros/namespace"), "/gazebo")
        names = {m.get("name") for m in self.world.findall("model")}
        self.assertTrue({"building", "obstacles", "cube_0", "cube_1"} <= names)
        self.assertEqual(set(layout.cubes), {n for n in names if n.startswith("cube_")})

    def test_the_file_is_what_the_generator_writes(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(PKG)), "tools"))
        try:
            import generate_world
        finally:
            sys.path.pop(0)
        self.assertEqual(read_text(os.path.join(PKG, "worlds", "rooms.world")), generate_world.build_world())


class Nav2ParamsTests(unittest.TestCase):
    def test_patch(self):
        data = yaml.safe_load(NAV2_SAMPLE)
        notes = nav2_params.patch_params(data, (1.5, -0.5, 1.0))
        amcl = data["amcl"]["ros__parameters"]
        self.assertIs(amcl["set_initial_pose"], True)
        self.assertEqual(amcl["initial_pose"], {"x": 1.5, "y": -0.5, "z": 0.0, "yaw": 1.0})
        for section in ("local_costmap", "global_costmap"):
            p = data[section][section]["ros__parameters"]
            self.assertEqual(p["robot_radius"], nav2_params.ROBOT_RADIUS)
            self.assertEqual(p["inflation_layer"]["inflation_radius"], nav2_params.INFLATION_RADIUS)
            self.assertEqual(p["inflation_layer"]["cost_scaling_factor"], 3.0)       # not touched
            self.assertIs(p["use_sim_time"], True)
        self.assertIs(data["bt_navigator"]["ros__parameters"]["use_sim_time"], True)
        self.assertTrue(any("robot_radius" in n and "2 place" in n for n in notes), notes)

    def test_the_lidar_skips_the_forks_and_the_load_only_with_the_forklift(self):
        data = yaml.safe_load(NAV2_SAMPLE)
        nav2_params.patch_params(data, (0, 0, 0))
        self.assertEqual(data["amcl"]["ros__parameters"]["laser_min_range"], -1.0)          # first version
        self.assertNotIn("obstacle_min_range", data["global_costmap"]["global_costmap"]["ros__parameters"]
                         ["obstacle_layer"]["scan"])
        data = yaml.safe_load(NAV2_SAMPLE)
        notes = nav2_params.patch_params(data, (0, 0, 0), robot_radius=0.36, self_filter=0.4)
        self.assertEqual(data["amcl"]["ros__parameters"]["laser_min_range"], 0.4)
        local = data["local_costmap"]["local_costmap"]["ros__parameters"]
        glob_ = data["global_costmap"]["global_costmap"]["ros__parameters"]
        self.assertEqual(local["voxel_layer"]["scan"]["obstacle_min_range"], 0.4)
        self.assertEqual(glob_["obstacle_layer"]["scan"]["obstacle_min_range"], 0.4)       # added where it was missing
        self.assertTrue(any("2 lidar source" in n for n in notes), notes)

    def test_start_pose_can_be_a_pose_object_or_a_tuple(self):
        # the launch file passes layout.robot_start (a Pose2D); this was a real bug once
        for start in (Pose2D(1.5, -0.5, 1.0), (1.5, -0.5, 1.0), [1.5, -0.5, 1.0]):
            data = yaml.safe_load(NAV2_SAMPLE)
            nav2_params.patch_params(data, start)
            self.assertEqual(data["amcl"]["ros__parameters"]["initial_pose"],
                             {"x": 1.5, "y": -0.5, "z": 0.0, "yaw": 1.0}, start)
        with self.assertRaises(ValueError):
            nav2_params.patch_params(yaml.safe_load(NAV2_SAMPLE), (1.0, 2.0))

    def test_file_without_amcl_still_works(self):
        data = {"controller_server": {"ros__parameters": {"use_sim_time": False}}}
        notes = nav2_params.patch_params(data, (0, 0, 0))
        self.assertTrue(any("amcl section not found" in n for n in notes))
        self.assertIs(data["controller_server"]["ros__parameters"]["use_sim_time"], True)

    def test_write_patched_file(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.yaml"), os.path.join(d, "out", "patched.yaml")
            write_text(src, NAV2_SAMPLE)
            notes = nav2_params.write_patched(src, dst, layout.robot_start.as_tuple())
            self.assertTrue(notes)
            text = read_text(dst)
            self.assertTrue(text.startswith("# Generated by actr_lab1"))
            back = yaml.safe_load(text)
            self.assertIs(back["amcl"]["ros__parameters"]["set_initial_pose"], True)
            self.assertIsInstance(back["local_costmap"]["local_costmap"]["ros__parameters"]["robot_radius"], float)
            self.assertEqual(back["local_costmap"]["local_costmap"]["ros__parameters"]["plugins"],
                             ["voxel_layer", "inflation_layer"])


class LaunchUtilsTests(unittest.TestCase):
    def make_install(self, root, with_nav=True, with_sdf=True, param_layout="humble"):
        gz = os.path.join(root, "turtlebot3_gazebo")
        os.makedirs(os.path.join(gz, "urdf"))
        os.makedirs(os.path.join(gz, "models", "turtlebot3_burger"))
        write_text(os.path.join(gz, "urdf", "turtlebot3_burger.urdf"), "<robot/>")
        if with_sdf:
            write_text(os.path.join(gz, "models", "turtlebot3_burger", "model.sdf"), "<sdf/>")
        nav = os.path.join(root, "turtlebot3_navigation2")
        folder = os.path.join(nav, "param", "humble") if param_layout == "humble" else os.path.join(nav, "param")
        if with_nav:
            os.makedirs(folder)
            write_text(os.path.join(folder, "burger.yaml"), NAV2_SAMPLE)
        b = os.path.join(root, "nav2_bringup", "params")
        os.makedirs(b)
        write_text(os.path.join(b, "nav2_params.yaml"), NAV2_SAMPLE)
        return {"turtlebot3_gazebo": gz, "turtlebot3_navigation2": nav, "nav2_bringup": os.path.join(root, "nav2_bringup")}

    def lookup(self, folders):
        def share(name):
            if name not in folders:
                raise KeyError(name)
            return folders[name]
        return share

    def test_append_path(self):
        self.assertEqual(launch_utils.append_path("", "/a"), "/a")
        self.assertEqual(launch_utils.append_path("/x:/y", "/a", "/x"), os.pathsep.join(["/x", "/y", "/a"]).replace(":", os.pathsep))
        self.assertEqual(launch_utils.append_path("/x", ""), "/x")

    def test_gazebo_model_path(self):
        with tempfile.TemporaryDirectory() as t:
            system = os.path.join(t, "gazebo-11", "models")
            os.makedirs(system)
            missing = os.path.join(t, "no", "such", "folder")
            got = launch_utils.gazebo_model_path("/mine", "/tb3/models", system_dirs=(system, missing))
            self.assertEqual(got.split(os.pathsep), ["/mine", "/tb3/models", system])       # a missing folder is left out
            again = launch_utils.gazebo_model_path(got, "/tb3/models", system_dirs=(system,))
            self.assertEqual(again, got)                                                     # nothing is repeated
            self.assertEqual(launch_utils.gazebo_model_path("", "/tb3/models", system_dirs=()), "/tb3/models")

    def test_files_with_distro_folder(self):
        with tempfile.TemporaryDirectory() as d:
            f = launch_utils.tb3_files("burger", self.lookup(self.make_install(d)), ros_distro="humble")
            self.assertTrue(f["sdf"].endswith(os.path.join("turtlebot3_burger", "model.sdf")))
            self.assertTrue(f["urdf"].endswith("turtlebot3_burger.urdf"))
            self.assertIn(os.path.join("param", "humble", "burger.yaml"), f["nav2_params"])
            self.assertTrue(f["models_dir"].endswith("models"))

    def test_files_with_plain_param_folder(self):
        with tempfile.TemporaryDirectory() as d:
            f = launch_utils.tb3_files("burger", self.lookup(self.make_install(d, param_layout="plain")),
                                       ros_distro="foxy")
            self.assertTrue(f["nav2_params"].endswith(os.path.join("param", "burger.yaml")))

    def test_distro_folder_found_by_search(self):
        with tempfile.TemporaryDirectory() as d:
            folders = self.make_install(d)
            f = launch_utils.tb3_files("burger", self.lookup(folders), ros_distro="")
            self.assertTrue(f["nav2_params"].endswith("burger.yaml"))

    def test_fallback_to_the_nav2_bringup_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = launch_utils.tb3_files("burger", self.lookup(self.make_install(d, with_nav=False)), ros_distro="humble")
            self.assertTrue(f["nav2_params"].endswith("nav2_params.yaml"))

    def test_missing_model_file_gives_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as cm:
                launch_utils.tb3_files("burger", self.lookup(self.make_install(d, with_sdf=False)), ros_distro="humble")
            self.assertIn("model.sdf", str(cm.exception))
            self.assertIn("TURTLEBOT3_MODEL", str(cm.exception))

    def test_missing_package_gives_an_install_hint(self):
        with self.assertRaises(FileNotFoundError) as cm:
            launch_utils.tb3_files("burger", self.lookup({}), ros_distro="humble")
        self.assertIn("apt install", str(cm.exception))


class LaunchFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        gz = os.path.join(root, "turtlebot3_gazebo")
        os.makedirs(os.path.join(gz, "urdf"))
        os.makedirs(os.path.join(gz, "models", "turtlebot3_burger"))
        write_text(os.path.join(gz, "urdf", "turtlebot3_burger.urdf"), "<robot name='burger'/>")
        write_text(os.path.join(gz, "models", "turtlebot3_burger", "model.sdf"), "<sdf/>")
        nav = os.path.join(root, "turtlebot3_navigation2")
        os.makedirs(os.path.join(nav, "param", "humble"))
        write_text(os.path.join(nav, "param", "humble", "burger.yaml"), NAV2_SAMPLE)
        self.folders = {"actr_lab1": PKG, "turtlebot3_gazebo": gz, "turtlebot3_navigation2": nav,
                        "gazebo_ros": os.path.join(root, "gazebo_ros"), "nav2_bringup": os.path.join(root, "nav2_bringup")}
        env = {"ROS_DISTRO": "humble"}
        patches = [mock.patch.dict(os.environ, env), mock.patch("tempfile.gettempdir", return_value=root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop("TURTLEBOT3_MODEL", None)
        os.environ.pop("GAZEBO_MODEL_PATH", None)
        ctx = fake_launch.install(self.folders)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)

    def launch(self, name):
        return fake_launch.load_launch_file(os.path.join(PKG, "launch", name))

    def sim_actions(self, **args):
        mod = self.launch("sim.launch.py")
        description = mod.generate_launch_description()
        declared = [e for e in description.entities if isinstance(e, fake_launch.DeclareLaunchArgument)]
        opaque = [e for e in description.entities if isinstance(e, fake_launch.OpaqueFunction)]
        self.assertEqual(len(opaque), 1)
        configs = {d.name: d.default_value for d in declared}
        configs.update(args)
        return declared, opaque[0].function(fake_launch.FakeContext(configs))

    # --- sim.launch.py ---------------------------------------------------------------
    def test_sim_launch_default(self):
        declared, actions = self.sim_actions()
        self.assertEqual({d.name for d in declared}, {"gui", "nav2", "rviz", "nav2_params", "arm"})
        env = {a.name: a.value for a in actions if isinstance(a, fake_launch.SetEnvironmentVariable)}
        self.assertIn(os.path.join(self.folders["turtlebot3_gazebo"], "models"), env["GAZEBO_MODEL_PATH"].split(os.pathsep))
        self.assertEqual(env["GAZEBO_MODEL_DATABASE_URI"], "")
        self.assertEqual(env["TURTLEBOT3_MODEL"], "burger")
        includes = [a for a in actions if isinstance(a, fake_launch.IncludeLaunchDescription)]
        sources = [i.source.path for i in includes]
        self.assertTrue(any(s.endswith("gzserver.launch.py") for s in sources))
        self.assertTrue(any(s.endswith("gzclient.launch.py") for s in sources))
        server = next(i for i in includes if i.source.path.endswith("gzserver.launch.py"))
        self.assertEqual(server.arguments["world"], os.path.join(PKG, "worlds", "rooms.world"))
        nodes = [a for a in actions if isinstance(a, fake_launch.Node)]
        rsp = next(n for n in nodes if n.package == "robot_state_publisher")
        params = rsp.parameters[0]
        self.assertIs(params["use_sim_time"], True)
        self.assertEqual(params["robot_description"], "<robot name='burger'/>")
        spawn = next(n for n in nodes if n.executable == "spawn_entity.py")
        args = spawn.arguments
        self.assertEqual(args[args.index("-entity") + 1], "burger")
        self.assertTrue(args[args.index("-file") + 1].endswith("model.sdf"))
        self.assertEqual(float(args[args.index("-x") + 1]), layout.robot_start.x)
        self.assertEqual(float(args[args.index("-y") + 1]), layout.robot_start.y)
        self.assertEqual(float(args[args.index("-Y") + 1]), layout.robot_start.yaw)

    def test_nav2_starts_after_the_robot_is_spawned(self):
        _declared, actions = self.sim_actions()
        spawn = next(a for a in actions if isinstance(a, fake_launch.Node) and a.executable == "spawn_entity.py")
        handlers = [a for a in actions if isinstance(a, fake_launch.RegisterEventHandler)]
        self.assertEqual(len(handlers), 1)
        handler = handlers[0].event_handler
        self.assertIs(handler.target_action, spawn)
        started = handler.fire(0)                       # the spawn ended well (exit code 0)
        bringup = [e for e in started if isinstance(e, fake_launch.IncludeLaunchDescription)]
        self.assertEqual(len(bringup), 1)
        a = bringup[0].arguments
        self.assertTrue(bringup[0].source.path.endswith(os.path.join("launch", "bringup_launch.py")))
        self.assertEqual(a["map"], os.path.join(PKG, "maps", "rooms.yaml"))
        self.assertEqual(a["use_sim_time"], "true")
        self.assertEqual(a["autostart"], "true")
        patched = yaml.safe_load(read_text(a["params_file"]))
        self.assertIs(patched["amcl"]["ros__parameters"]["set_initial_pose"], True)
        self.assertEqual(patched["amcl"]["ros__parameters"]["initial_pose"]["x"], layout.robot_start.x)
        self.assertFalse(any(isinstance(e, fake_launch.Node) for e in started))      # no rviz

    def test_options(self):
        _d, actions = self.sim_actions(gui="false", rviz="true")
        sources = [a.source.path for a in actions if isinstance(a, fake_launch.IncludeLaunchDescription)]
        self.assertFalse(any(s.endswith("gzclient.launch.py") for s in sources))
        handler = next(a for a in actions if isinstance(a, fake_launch.RegisterEventHandler)).event_handler
        rviz = [e for e in handler.fire(0) if isinstance(e, fake_launch.Node)]
        self.assertEqual([n.executable for n in rviz], ["rviz2"])

    def test_without_nav2_nothing_starts_after_the_spawn(self):
        _d, actions = self.sim_actions(nav2="false")
        handler = next(a for a in actions if isinstance(a, fake_launch.RegisterEventHandler)).event_handler
        self.assertEqual(handler.fire(0), [])

    def test_a_failed_spawn_stops_the_launch_and_nav2_does_not_start(self):
        for nav2 in ("true", "false"):
            _d, actions = self.sim_actions(nav2=nav2, rviz="true")
            handler = next(a for a in actions if isinstance(a, fake_launch.RegisterEventHandler)).event_handler
            started = handler.fire(1)
            self.assertFalse(any(isinstance(e, (fake_launch.IncludeLaunchDescription, fake_launch.Node))
                                 for e in started), nav2)
            messages = [e.msg for e in started if isinstance(e, fake_launch.LogInfo)]
            self.assertTrue(any("spawn_entity.py failed (exit code 1)" in m for m in messages), messages)
            emitted = [e.event for e in started if isinstance(e, fake_launch.EmitEvent)]
            self.assertEqual(len(emitted), 1)
            self.assertIsInstance(emitted[0], fake_launch.Shutdown)

    def test_the_patched_nav2_parameters_get_a_new_private_folder_every_time(self):
        def params_file():
            _d, actions = self.sim_actions()
            handler = next(a for a in actions if isinstance(a, fake_launch.RegisterEventHandler)).event_handler
            return next(e for e in handler.fire(0) if isinstance(e, fake_launch.IncludeLaunchDescription)) \
                .arguments["params_file"]
        first, second = params_file(), params_file()
        self.assertNotEqual(os.path.dirname(first), os.path.dirname(second))
        self.assertNotEqual(os.path.basename(first), "actr_lab1_nav2_params.yaml")
        self.assertEqual(os.stat(os.path.dirname(first)).st_mode & 0o077, 0)          # only the owner may enter
        self.assertTrue(os.path.isfile(first))

    def test_own_nav2_parameter_file(self):
        own = os.path.join(self.tmp.name, "own.yaml")
        write_text(own, NAV2_SAMPLE.replace("0.22", "0.5"))
        _d, actions = self.sim_actions(nav2_params=own)
        logs = " ".join(a.msg for a in [e for e in actions if isinstance(e, fake_launch.LogInfo)])
        handler = next(a for a in actions if isinstance(a, fake_launch.RegisterEventHandler)).event_handler
        self.assertTrue(any("own.yaml" in e.msg for e in handler.fire(0) if isinstance(e, fake_launch.LogInfo)))

    # --- the forklift -------------------------------------------------------------------
    def spawn_file_and_radius(self, **args):
        _d, actions = self.sim_actions(**args)
        spawn = next(a for a in actions if isinstance(a, fake_launch.Node) and a.executable == "spawn_entity.py")
        sdf = spawn.arguments[spawn.arguments.index("-file") + 1]
        handler = next(a for a in actions if isinstance(a, fake_launch.RegisterEventHandler)).event_handler
        bringup = next(e for e in handler.fire(0) if isinstance(e, fake_launch.IncludeLaunchDescription))
        self.params = yaml.safe_load(read_text(bringup.arguments["params_file"]))
        radius = self.params["local_costmap"]["local_costmap"]["ros__parameters"]["robot_radius"]
        logs = " ".join(a.msg for a in actions if isinstance(a, fake_launch.LogInfo))
        return sdf, radius, logs

    def use_real_looking_model(self):
        write_text(os.path.join(self.folders["turtlebot3_gazebo"], "models", "turtlebot3_burger", "model.sdf"),
                   common.TB3_BURGER_SDF)

    def test_the_forklift_is_added_to_the_robot(self):
        from actr_lab1.ros import arm_model
        from actr_lab1.ros import nav2_params as n2
        self.use_real_looking_model()
        original = os.path.join(self.folders["turtlebot3_gazebo"], "models", "turtlebot3_burger", "model.sdf")
        sdf, radius, logs = self.spawn_file_and_radius()
        self.assertNotEqual(sdf, original)                       # a copy with the forklift
        self.assertTrue(sdf.endswith("model.sdf"))
        text = read_text(sdf)
        for needed in (arm_model.LIFT_LINK, arm_model.LIFT_JOINT, "fork_counterweight",
                       "libgazebo_ros_joint_pose_trajectory.so", "libgazebo_ros_diff_drive.so"):
            self.assertIn(needed, text)
        self.assertNotIn("vacuum", text)
        self.assertEqual(read_text(original), common.TB3_BURGER_SDF)   # the installed file is not changed
        self.assertAlmostEqual(radius, arm_model.REACH)                 # Nav2 knows that the robot is longer
        local = self.params["local_costmap"]["local_costmap"]["ros__parameters"]
        self.assertAlmostEqual(local["inflation_layer"]["inflation_radius"], n2.FORKLIFT_INFLATION)
        # the lidar sees only the near face of the carried cube (forks and pallet are lower than the lidar)
        near_face = math.hypot(arm_model.FULL_IN - layout.cube_size / 2 + 0.032, layout.cube_size / 2)
        self.assertGreater(n2.SELF_FILTER, near_face + 0.05)
        self.assertLess(n2.SELF_FILTER, 0.30)                    # the blind circle around the robot stays small
        self.assertEqual(self.params["amcl"]["ros__parameters"]["laser_min_range"], n2.SELF_FILTER)
        self.assertEqual(local["voxel_layer"]["scan"]["obstacle_min_range"], n2.SELF_FILTER)
        self.assertIn("forklift: forks and a lift are added to the robot", logs)

    def test_without_the_forklift(self):
        from actr_lab1.ros import nav2_params as n2
        self.use_real_looking_model()
        original = os.path.join(self.folders["turtlebot3_gazebo"], "models", "turtlebot3_burger", "model.sdf")
        sdf, radius, logs = self.spawn_file_and_radius(arm="false")
        self.assertEqual(sdf, original)
        self.assertAlmostEqual(radius, n2.ROBOT_RADIUS)
        self.assertEqual(self.params["amcl"]["ros__parameters"]["laser_min_range"], -1.0)    # as in the first version
        self.assertNotIn("forklift:", logs)

    def test_a_model_that_cannot_get_the_forklift_is_spawned_without_it(self):
        from actr_lab1.ros import nav2_params as n2
        original = os.path.join(self.folders["turtlebot3_gazebo"], "models", "turtlebot3_burger", "model.sdf")
        sdf, radius, logs = self.spawn_file_and_radius()         # the model of setUp is only "<sdf/>"
        self.assertEqual(sdf, original)
        self.assertAlmostEqual(radius, n2.ROBOT_RADIUS)
        self.assertIn("the forklift could not be added", logs)
        self.assertIn("virtual gripper", logs)

    def test_other_robot_model_is_used_if_the_environment_says_so(self):
        os.environ["TURTLEBOT3_MODEL"] = "waffle"
        with self.assertRaises(FileNotFoundError) as cm:
            self.sim_actions()
        self.assertIn("waffle", str(cm.exception))

    # --- agent.launch.py -----------------------------------------------------------------
    def test_agent_launch(self):
        mod = self.launch("agent.launch.py")
        description = mod.generate_launch_description()
        declared = [e for e in description.entities if isinstance(e, fake_launch.DeclareLaunchArgument)]
        node = next(e for e in description.entities if isinstance(e, fake_launch.Node))
        self.assertEqual([d.name for d in declared], [a[0] for a in launch_utils.AGENT_ARGS])
        self.assertEqual((node.package, node.executable), ("actr_lab1", "agent_node"))
        self.assertEqual(node.kwargs["name"], "actr_agent")
        config, fixed, dynamic = node.parameters
        self.assertTrue(config.endswith(os.path.join("config", "agent.yaml")))
        self.assertIs(fixed["use_sim_time"], True)
        self.assertEqual(set(dynamic), {a[0] for a in launch_utils.AGENT_ARGS})
        for name, default, typ, _help in launch_utils.AGENT_ARGS:
            self.assertIs(dynamic[name].value_type, typ, name)
            self.assertEqual(dynamic[name].value.name, name)

    def test_launch_arguments_are_parameters_of_the_node(self):
        """A launch argument that the node does not declare would be ignored without any message."""
        world = fake_ros.FakeWorld(layout, grid)
        with fake_ros.install(world):
            from actr_lab1.ros import agent_node
            node = agent_node.AgentNode()
            declared = dict(node.params)
        for name, default, typ, _help in launch_utils.AGENT_ARGS:
            self.assertIn(name, declared, f"launch argument '{name}' is not a node parameter")
            value = declared[name]
            self.assertIs(type(value), typ, f"{name}: launch type {typ.__name__}, node default {type(value).__name__}")
            if typ is bool:
                self.assertEqual(str(value).lower(), default, name)

    def test_agent_yaml(self):
        data = yaml.safe_load(read_text(os.path.join(PKG, "config", "agent.yaml")))
        self.assertEqual(list(data), ["actr_agent"])
        params = data["actr_agent"]["ros__parameters"]
        world = fake_ros.FakeWorld(layout, grid)
        with fake_ros.install(world):
            from actr_lab1.ros import agent_node
            declared = dict(agent_node.AgentNode().params)
        launch_names = {a[0] for a in launch_utils.AGENT_ARGS}
        for name, value in params.items():
            self.assertIn(name, declared, f"agent.yaml sets '{name}', which the node does not declare")
            self.assertIs(type(value), type(declared[name]), name)
            self.assertNotIn(name, launch_names, f"'{name}' is set twice (yaml and launch argument)")

    # --- all.launch.py ------------------------------------------------------------------
    def test_all_launch(self):
        mod = self.launch("all.launch.py")
        description = mod.generate_launch_description()
        declared = [e.name for e in description.entities if isinstance(e, fake_launch.DeclareLaunchArgument)]
        self.assertEqual(declared, [a[0] for a in launch_utils.SIM_ARGS + launch_utils.AGENT_ARGS])
        includes = [e.source.path for e in description.entities if isinstance(e, fake_launch.IncludeLaunchDescription)]
        self.assertEqual([os.path.basename(p) for p in includes], ["sim.launch.py", "agent.launch.py"])
        for p in includes:
            self.assertTrue(os.path.isfile(p), p)


if __name__ == "__main__":
    unittest.main()
