"""Tests of the shell scripts: tools/preflight.sh, docker/run.sh and docker/entrypoint.sh.

The scripts run for real, but `docker`, `ros2` and `gzserver` are small fake programs that
write down what they were asked, so these tests check the LOGIC of the scripts. They cannot show
that real Docker, WSLg or ROS accept the same commands.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import common
from fake_llm_server import FakeLLMServer

PKG = common.PKG_ROOT
REPO = os.path.dirname(os.path.dirname(PKG))
PREFLIGHT = os.path.join(REPO, "tools", "preflight.sh")
RUN_SH = os.path.join(REPO, "docker", "run.sh")
ENTRYPOINT = os.path.join(REPO, "docker", "entrypoint.sh")
DOCKERFILE = os.path.join(REPO, "docker", "Dockerfile")
# Inside the Docker image only src/, tools/ and entrypoint.sh are copied, so docker/run.sh and the
# Dockerfile are not there: the tests that need them are skipped in that case.
HAVE_DOCKER_FILES = os.path.exists(RUN_SH) and os.path.exists(DOCKERFILE)
NO_DOCKER_FILES = "docker/run.sh and the Dockerfile are not here (normal inside the Docker image)"


def write_exec(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o755)


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def run(cmd, env, **kw):
    return subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, timeout=120, **kw)


@unittest.skipUnless(shutil.which("bash"), "bash is needed")
@unittest.skipUnless(HAVE_DOCKER_FILES, NO_DOCKER_FILES)
class ScriptFilesTests(unittest.TestCase):
    def test_syntax(self):
        for path in (PREFLIGHT, RUN_SH, ENTRYPOINT):
            r = subprocess.run(["bash", "-n", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            self.assertEqual(r.returncode, 0, f"{path}: {r.stderr}")

    def test_scripts_start_with_a_shebang_and_are_executable(self):
        for path in (PREFLIGHT, RUN_SH, ENTRYPOINT):
            self.assertTrue(read_text(path).startswith("#!/usr/bin/env bash"), path)
            self.assertTrue(os.access(path, os.X_OK), f"{path} is not executable")

    def test_no_windows_line_endings(self):
        for path in (PREFLIGHT, RUN_SH, ENTRYPOINT, DOCKERFILE):
            with open(path, "rb") as f:
                self.assertNotIn(b"\r", f.read(), path)

    def test_help(self):
        for path, word in ((PREFLIGHT, "--runtime"), (RUN_SH, "llm-start")):
            args = ["--help"]
            r = run(["bash", path] + args, dict(os.environ))
            self.assertEqual(r.returncode, 0)
            self.assertIn(word, r.stdout)
            self.assertNotIn("set -u", r.stdout)


@unittest.skipUnless(HAVE_DOCKER_FILES, NO_DOCKER_FILES)
class DockerfileTests(unittest.TestCase):
    def setUp(self):
        self.text = read_text(DOCKERFILE)

    def test_distro_is_an_argument_that_is_declared_before_from(self):
        lines = [line.strip() for line in self.text.splitlines() if line.strip() and not line.startswith("#")]
        self.assertEqual(lines[0], "ARG ROS_DISTRO=humble")
        self.assertTrue(lines[1].startswith("FROM osrf/ros:${ROS_DISTRO}-desktop"))
        self.assertIn("ARG ROS_DISTRO", lines[2:3])          # needed again after FROM

    def test_installs_what_the_launch_files_need(self):
        from actr_lab1.ros.launch_utils import INSTALL_HINT
        for package in INSTALL_HINT.split():
            if package.startswith("ros-$ROS_DISTRO-"):
                self.assertIn(package.replace("$ROS_DISTRO", "${ROS_DISTRO}"), self.text, package)
        for needed in ("python3-colcon-common-extensions", "python3-yaml", "colcon build", "TURTLEBOT3_MODEL=burger",
                       "ENTRYPOINT", "COPY src /ws/src", "COPY tools /ws/tools"):
            self.assertIn(needed, self.text, needed)

    def test_home_folder_exists_and_any_user_can_write_in_it(self):
        """run.sh starts the container as the user of the host with HOME=/tmp/actr_home and mounts the log folder
        below it. If the image had no such folder, Docker would create it as root and ROS could not write there."""
        self.assertIn("RUN install -d -m 1777 /tmp/actr_home", self.text)
        run_sh = read_text(RUN_SH)
        self.assertIn("HOME=/tmp/actr_home", run_sh)
        self.assertIn(":/tmp/actr_home/actr_lab1_logs", run_sh)
        self.assertIn("/tmp/actr_home", read_text(ENTRYPOINT))

    def test_files_that_are_copied_exist_and_are_not_ignored(self):
        self.assertTrue(os.path.isdir(os.path.join(REPO, "src")))
        self.assertTrue(os.path.isfile(os.path.join(REPO, "tools", "preflight.sh")))
        self.assertTrue(os.path.isfile(ENTRYPOINT))
        ignored = [line.strip() for line in read_text(os.path.join(REPO, ".dockerignore")).splitlines()]
        for must_not in ("src", "tools", "docker"):
            self.assertNotIn(must_not, ignored)
        self.assertIn("**/__pycache__", ignored)

    def test_gitignore_keeps_secrets_and_logs_out(self):
        text = read_text(os.path.join(REPO, ".gitignore"))
        for entry in (".env", "*.key", "actr_lab1_logs/", "__pycache__/"):
            self.assertIn(entry, text)
        # the results of the experiments are part of the report, so they go to git
        self.assertNotIn("\nresults/", "\n" + text)


@unittest.skipUnless(shutil.which("bash"), "bash is needed")
@unittest.skipUnless(HAVE_DOCKER_FILES, NO_DOCKER_FILES)
class RunScriptTests(unittest.TestCase):
    FAKE_DOCKER = r'''#!/usr/bin/env bash
{ echo "--- docker"; for a in "$@"; do printf '%s\n' "$a"; done; } >> "$FAKE_LOG"
case "$1" in
  info) [ "${FAKE_DOCKER_DOWN:-0}" = 1 ] && exit 1; exit 0 ;;
  image) [ "${FAKE_IMAGE:-0}" = 1 ] && exit 0 || exit 1 ;;
  network) if [ "$2" = inspect ]; then [ "${FAKE_NET:-0}" = 1 ] && exit 0 || exit 1; fi; exit 0 ;;
  ps) for n in ${FAKE_RUNNING:-}; do echo "$n"; done; exit 0 ;;
  run) [ "${FAKE_RUN_FAILS:-0}" = 1 ] && exit 1; exit 0 ;;
esac
exit 0
'''

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.repo = os.path.join(t, "repo")
        os.makedirs(os.path.join(self.repo, "docker"))
        shutil.copy(RUN_SH, os.path.join(self.repo, "docker", "run.sh"))
        self.script = os.path.join(self.repo, "docker", "run.sh")
        self.home = os.path.join(t, "home")
        os.makedirs(self.home)
        self.wslg = os.path.join(t, "wslg")
        self.x11 = os.path.join(t, "x11")
        os.makedirs(self.wslg)
        os.makedirs(self.x11)
        self.log = os.path.join(t, "docker.log")
        self.docker = os.path.join(t, "docker")
        write_exec(self.docker, self.FAKE_DOCKER)

    def tearDown(self):
        self.tmp.cleanup()

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items()
               if k not in ("OLLAMA_HOST", "DISPLAY", "DEV", "GPU", "ROS_DISTRO", "LAB_ROS_DISTRO")}
        env.update(DOCKER=self.docker, FAKE_LOG=self.log, HOME=self.home, WSLG_DIR=self.wslg, X11_DIR=self.x11,
                   FAKE_IMAGE="1", FAKE_NET="1", DISPLAY=":0")
        env.update(extra)
        return env

    def script_run(self, *args, **extra):
        return run(["bash", self.script] + list(args), self.env(**extra))

    def calls(self):
        if not os.path.exists(self.log):
            return []
        out, current = [], None
        for line in read_text(self.log).splitlines():
            if line == "--- docker":
                current = []
                out.append(current)
            else:
                current.append(line)
        return out

    def call(self, first):
        found = [c for c in self.calls() if c and c[0] == first]
        self.assertTrue(found, f"no 'docker {first}' call in {self.calls()}")
        return found[-1]

    @staticmethod
    def pair(args, flag, value):
        return any(args[i] == flag and args[i + 1] == value for i in range(len(args) - 1))

    # ---------------------------------------------------------------- build
    def test_build(self):
        r = self.script_run("build")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.call("build"),
                         ["build", "-f", "docker/Dockerfile", "--build-arg", "ROS_DISTRO=humble", "-t",
                          "actr_lab1:humble", "."])

    def test_build_for_foxy(self):
        r = self.script_run("build", LAB_ROS_DISTRO="foxy")
        self.assertEqual(r.returncode, 0, r.stderr)
        args = self.call("build")
        self.assertIn("ROS_DISTRO=foxy", args)
        self.assertIn("actr_lab1:foxy", args)
        self.script_run("run", "true", LAB_ROS_DISTRO="foxy")
        self.assertIn("actr_lab1:foxy", self.call("run"))

    def test_the_ros_distro_of_this_computer_does_not_choose_the_image(self):
        """If ROS is installed in WSL itself, ROS_DISTRO is 'jazzy' or so: the image must stay humble."""
        self.script_run("build", ROS_DISTRO="jazzy")
        self.assertIn("actr_lab1:humble", self.call("build"))
        self.assertIn("ROS_DISTRO=humble", self.call("build"))
        self.script_run("run", "true", ROS_DISTRO="jazzy")
        self.assertIn("actr_lab1:humble", self.call("run"))

    # ------------------------------------------------------------------ run
    def test_run_a_launch_command_in_a_new_container(self):
        r = self.script_run("run", "ros2", "launch", "actr_lab1", "all.launch.py", "gui:=false", "planner:=ollama")
        self.assertEqual(r.returncode, 0, r.stderr)
        a = self.call("run")
        self.assertIn("--rm", a)
        self.assertTrue(self.pair(a, "--name", "actr_lab"))
        self.assertTrue(self.pair(a, "--network", "actr_net"))
        self.assertTrue(self.pair(a, "--user", f"{os.getuid()}:{os.getgid()}"))
        self.assertTrue(self.pair(a, "-e", "HOME=/tmp/actr_home"))
        self.assertTrue(self.pair(a, "-e", "OLLAMA_HOST=http://actr_ollama:11434"))
        self.assertIn("--add-host=host.docker.internal:host-gateway", a)
        self.assertIn("--shm-size=1g", a)
        self.assertIn("-i", a)                                   # stdin is not a terminal in the test
        self.assertNotIn("-it", a)
        i = a.index("actr_lab1:humble")
        self.assertEqual(a[i + 1:], ["ros2", "launch", "actr_lab1", "all.launch.py", "gui:=false", "planner:=ollama"])
        self.assertEqual(self.calls()[0][0], "info")             # asked first whether docker works
        # the folders for the results and the logs belong to the user, not to root
        self.assertTrue(os.path.isdir(os.path.join(self.repo, "results")))
        self.assertTrue(os.path.isdir(os.path.join(self.home, "actr_lab1_logs")))
        self.assertTrue(self.pair(a, "-v", f"{self.repo}/results:/ws/results"))
        self.assertTrue(self.pair(a, "-v", f"{self.home}/actr_lab1_logs:/tmp/actr_home/actr_lab1_logs"))

    def test_a_second_run_explains_that_the_lab_container_exists(self):
        r = self.script_run("run", "ros2", "topic", "list", FAKE_RUNNING="actr_lab")
        self.assertEqual(r.returncode, 1)
        self.assertIn("already running", r.stderr)
        self.assertIn("docker/run.sh exec", r.stderr)
        self.assertIn("docker rm -f actr_lab", r.stderr)
        self.assertFalse([c for c in self.calls() if c[0] == "run"])

    def test_a_stopped_leftover_container_with_the_same_name_is_removed_first(self):
        self.script_run("run", "true")
        names = [c[0] for c in self.calls()]
        self.assertLess(names.index("rm"), names.index("run"))
        self.assertEqual(self.call("rm"), ["rm", "-f", "actr_lab"])

    def test_shell_in_a_running_lab_opens_a_second_shell(self):
        r = self.script_run("shell", FAKE_RUNNING="actr_lab")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.call("exec"), ["exec", "-i", "actr_lab", "/ws/docker/entrypoint.sh", "bash"])
        self.assertIn("second shell", r.stderr)
        self.assertFalse([c for c in self.calls() if c[0] == "run"])

    def test_wslg_windows_11(self):
        self.script_run("shell", XDG_RUNTIME_DIR="/mnt/wslg/runtime-dir", WAYLAND_DISPLAY="wayland-0",
                        PULSE_SERVER="unix:/mnt/wslg/PulseServer")
        a = self.call("run")
        self.assertTrue(self.pair(a, "-v", f"{self.x11}:/tmp/.X11-unix"))
        self.assertTrue(self.pair(a, "-v", f"{self.wslg}:/mnt/wslg"))
        for env in ("DISPLAY=:0", "WAYLAND_DISPLAY=wayland-0", "XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir",
                    "PULSE_SERVER=unix:/mnt/wslg/PulseServer"):
            self.assertTrue(self.pair(a, "-e", env), env)
        self.assertEqual(a[-1], "bash")

    def test_wslg_defaults_when_the_variables_are_empty(self):
        self.script_run("shell", DISPLAY="")
        a = self.call("run")
        for env in ("DISPLAY=:0", "WAYLAND_DISPLAY=wayland-0", "XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir"):
            self.assertTrue(self.pair(a, "-e", env), env)

    def test_plain_linux_desktop(self):
        shutil.rmtree(self.wslg)
        self.script_run("shell", DISPLAY=":1")
        a = self.call("run")
        self.assertTrue(self.pair(a, "-v", f"{self.x11}:/tmp/.X11-unix"))
        self.assertTrue(self.pair(a, "-e", "DISPLAY=:1"))
        self.assertFalse(any(x.endswith(":/mnt/wslg") for x in a))

    def test_no_display_at_all(self):
        shutil.rmtree(self.wslg)
        self.script_run("shell", DISPLAY="")
        a = self.call("run")
        self.assertFalse(any("DISPLAY" in x for x in a))
        self.assertFalse(any("X11" in x for x in a))

    def test_default_command_is_a_shell(self):
        r = self.script_run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.call("run")[-1], "bash")

    def test_arguments_with_spaces_stay_in_one_piece(self):
        msg = "{data: 'Bring cube 0 to room A'}"
        self.script_run("run", "ros2", "topic", "pub", "--once", "/actr/mission", "std_msgs/msg/String", msg)
        a = self.call("run")
        self.assertEqual(a[-1], msg)
        self.assertEqual(a[-2], "std_msgs/msg/String")

    def test_ollama_host_can_be_changed(self):
        self.script_run("run", "true", OLLAMA_HOST="http://host.docker.internal:11434")
        self.assertTrue(self.pair(self.call("run"), "-e", "OLLAMA_HOST=http://host.docker.internal:11434"))

    def test_dev_mode_mounts_the_source(self):
        self.script_run("run", "true", DEV="1")
        self.assertTrue(self.pair(self.call("run"), "-v", f"{self.repo}/src:/ws/src"))
        self.script_run("run", "true")
        self.assertFalse(any(x.endswith(":/ws/src") for x in self.calls()[-1]))

    def test_network_is_created_when_missing(self):
        self.script_run("run", "true", FAKE_NET="0")
        self.assertEqual(self.call("network")[0:1], ["network"])
        created = [c for c in self.calls() if c[:2] == ["network", "create"]]
        self.assertEqual(created, [["network", "create", "actr_net"]])
        self.script_run("run", "true", FAKE_NET="1")
        self.assertEqual(len([c for c in self.calls() if c[:2] == ["network", "create"]]), 1)

    # ----------------------------------------------------- problems are explained
    def test_image_missing(self):
        r = self.script_run("run", "true", FAKE_IMAGE="0")
        self.assertEqual(r.returncode, 1)
        self.assertIn("docker/run.sh build", r.stderr)
        self.assertFalse([c for c in self.calls() if c[0] == "run"])

    def test_docker_not_running(self):
        r = self.script_run("run", "true", FAKE_DOCKER_DOWN="1")
        self.assertEqual(r.returncode, 1)
        self.assertIn("Start Docker Desktop", r.stderr)

    def test_docker_not_installed(self):
        r = self.script_run("run", "true", DOCKER="/nonexistent/docker")
        self.assertEqual(r.returncode, 1)
        self.assertIn("WSL integration", r.stderr)

    def test_unknown_command_and_missing_arguments(self):
        self.assertEqual(self.script_run("dance").returncode, 1)
        self.assertIn("unknown command", self.script_run("dance").stderr)
        self.assertIn("needs a command", self.script_run("run").stderr)
        self.assertIn("needs a command", self.script_run("exec").stderr)

    # ------------------------------------------------------------------ llm
    def test_llm_start(self):
        r = self.script_run("llm-start", FAKE_NET="0")
        self.assertEqual(r.returncode, 0, r.stderr)
        a = self.call("run")
        self.assertEqual(a[:2], ["run", "-d"])
        self.assertTrue(self.pair(a, "--name", "actr_ollama"))
        self.assertTrue(self.pair(a, "--network", "actr_net"))
        self.assertTrue(self.pair(a, "-p", "11434:11434"))
        self.assertTrue(self.pair(a, "-v", "actr_ollama_models:/root/.ollama"))
        self.assertEqual(a[-1], "ollama/ollama")
        self.assertNotIn("--gpus", a)
        self.assertIn("llm-pull", r.stdout)

    def test_llm_start_with_gpu_and_other_port(self):
        self.script_run("llm-start", GPU="1", OLLAMA_PORT="11500")
        a = self.call("run")
        self.assertTrue(self.pair(a, "--gpus", "all"))
        self.assertTrue(self.pair(a, "-p", "11500:11434"))

    def test_llm_start_when_it_is_already_running(self):
        r = self.script_run("llm-start", FAKE_RUNNING="actr_ollama")
        self.assertEqual(r.returncode, 0)
        self.assertIn("already running", r.stdout)
        self.assertFalse([c for c in self.calls() if c[0] == "run"])

    def test_llm_start_when_the_port_is_busy(self):
        r = self.script_run("llm-start", FAKE_RUN_FAILS="1")
        self.assertEqual(r.returncode, 1)
        self.assertIn("already run Ollama", r.stderr)

    def test_llm_pull(self):
        r = self.script_run("llm-pull", FAKE_RUNNING="actr_ollama")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.call("exec"), ["exec", "-it", "actr_ollama", "ollama", "pull", "qwen2.5:3b-instruct"])
        self.script_run("llm-pull", "llama3.2:3b", FAKE_RUNNING="actr_ollama")
        self.assertEqual(self.call("exec")[-1], "llama3.2:3b")

    def test_llm_pull_needs_a_running_server(self):
        r = self.script_run("llm-pull")
        self.assertEqual(r.returncode, 1)
        self.assertIn("llm-start", r.stderr)

    def test_llm_stop(self):
        r = self.script_run("llm-stop")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(self.call("rm"), ["rm", "-f", "actr_ollama"])

    # ------------------------------------------------------ exec, check, test
    def test_exec_in_the_running_container(self):
        r = self.script_run("exec", "ros2", "topic", "list", FAKE_RUNNING="actr_lab")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.call("exec"), ["exec", "-i", "actr_lab", "/ws/docker/entrypoint.sh", "ros2", "topic",
                                             "list"])

    def test_exec_without_a_container(self):
        r = self.script_run("exec", "ros2", "topic", "list")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no lab container", r.stderr)

    def test_check_looks_at_the_computer_in_a_new_container_without_a_name(self):
        r = self.script_run("check", "--no-llm", FAKE_RUNNING="actr_lab")      # works while the simulation runs
        self.assertEqual(r.returncode, 0, r.stderr)
        a = self.call("run")
        self.assertNotIn("--name", a)
        i = a.index("actr_lab1:humble")
        self.assertEqual(a[i + 1:], ["/ws/tools/preflight.sh", "--no-llm"])

    def test_check_runtime_goes_into_the_running_lab_container(self):
        """The ROS graph of the simulation lives there: a new container would see nothing."""
        r = self.script_run("check", "--runtime", FAKE_RUNNING="actr_lab")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.call("exec"), ["exec", "-i", "actr_lab", "/ws/docker/entrypoint.sh",
                                             "/ws/tools/preflight.sh", "--runtime"])
        self.assertFalse([c for c in self.calls() if c[0] == "run"])

    def test_check_runtime_without_a_simulation_says_how_to_start_one(self):
        r = self.script_run("check", "--runtime")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no lab container is running", r.stderr)
        self.assertIn("sim.launch.py", r.stderr)
        self.assertFalse([c for c in self.calls() if c[0] in ("run", "exec")])

    def test_unit_tests_inside(self):
        self.script_run("test", FAKE_RUNNING="actr_lab")          # also while the simulation runs
        a = self.call("run")
        self.assertNotIn("--name", a)
        self.assertTrue(self.pair(a, "-w", "/ws/src/actr_lab1"))
        self.assertEqual(a[a.index("actr_lab1:humble") + 1:], ["python3", "-m", "unittest", "discover", "-s", "test"])


class EntrypointTests(unittest.TestCase):
    def test_sources_ros_and_runs_the_command(self):
        with tempfile.TemporaryDirectory() as t:
            ros = os.path.join(t, "opt", "ros", "humble")
            os.makedirs(ros)
            write_exec(os.path.join(ros, "setup.bash"), "export FROM_ROS=1\n")
            # the script has fixed paths (/opt/ros, /ws/install): run a copy with them replaced
            text = read_text(ENTRYPOINT).replace('"/opt/ros/${ROS_DISTRO}/setup.bash"', f'"{ros}/setup.bash"') \
                .replace("/ws/install/setup.bash", os.path.join(t, "ws_setup.bash"))
            write_exec(os.path.join(t, "ws_setup.bash"), "export FROM_WS=1\n")
            script = os.path.join(t, "entry.sh")
            write_exec(script, text)
            env = {"PATH": os.environ["PATH"], "ROS_DISTRO": "humble"}
            r = run(["bash", script, "bash", "-c", 'echo "$FROM_ROS $FROM_WS $HOME"'], env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "1 1 /tmp/actr_home")
            r = run(["bash", script, "bash", "-c", "exit 7"], env)
            self.assertEqual(r.returncode, 7)               # the exit code of the command is kept


@unittest.skipUnless(shutil.which("bash") and shutil.which("curl"), "bash and curl are needed")
class PreflightTests(unittest.TestCase):
    FAKE_ROS2 = r'''#!/usr/bin/env bash
[ -n "${FAKE_ROS2_LOG:-}" ] && echo "$*" >> "$FAKE_ROS2_LOG"
# FAKE_SLOW_DAEMON=1: like a ROS 2 daemon that has just started, the first look at the system sees no
# services, actions or nodes (only the topics)
first_look() { [ "${FAKE_SLOW_DAEMON:-0}" = 1 ] && [ "$(grep -c '^topic list' "$FAKE_ROS2_LOG")" -le 1 ]; }
case "$1 $2" in
  "pkg prefix")
    for m in ${FAKE_MISSING_PKGS:-}; do [ "$m" = "$3" ] && exit 1; done
    echo "/opt/fake/$3"; exit 0 ;;
  "topic list") printf '%s\n' ${FAKE_TOPICS:-} ;;
  "topic info") printf 'Type: trajectory_msgs/msg/JointTrajectory\nPublisher count: 0\nSubscription count: %s\n' \
                  "${FAKE_LIFT_LISTENERS:-1}" ;;
  "service list") first_look && exit 0; printf '%s\n' ${FAKE_SERVICES:-} ;;
  "action list") first_look && exit 0; printf '%s\n' ${FAKE_ACTIONS:-} ;;
  "node list") first_look && exit 0; printf '%s\n' ${FAKE_NODES:-} ;;
  "topic echo")
    # Foxy has no `--once`: the real command stops with an error. The fake does the same.
    for a in "$@"; do
      if [ "$a" = --once ]; then echo "ros2 topic echo: error: unrecognized arguments: --once" >&2; exit 2; fi
    done
    # no message at all (the real command would wait until `timeout` stops it)
    for a in "$@"; do
      if [ "$a" = /amcl_pose ] && [ "${FAKE_NO_AMCL:-0}" = 1 ]; then exit 0; fi
    done
    echo "header:"; echo "  stamp:" ;;
  "run tf2_ros") echo "- Translation: [0.0, 0.0, 0.0]" ;;
esac
exit 0
'''
    TOPICS = "/clock /scan /odom /tf /cmd_vel /amcl_pose /map /rosout"
    SERVICES = "/gazebo/get_entity_state /gazebo/set_entity_state"
    ACTIONS = "/navigate_to_pose"
    NODES = "/amcl /bt_navigator /controller_server /planner_server /actr_agent"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.bin = os.path.join(t, "bin")
        os.makedirs(self.bin)
        write_exec(os.path.join(self.bin, "ros2"), self.FAKE_ROS2)
        write_exec(os.path.join(self.bin, "gzserver"), "#!/usr/bin/env bash\necho 'gazebo multi-robot simulator, version 11.10.2'\n")
        # a fake `rclpy` and a fake `ament_index_python` so that the python checks pass
        self.py = os.path.join(t, "py")
        os.makedirs(os.path.join(self.py, "rclpy"))
        write_exec(os.path.join(self.py, "rclpy", "__init__.py"), "")
        os.makedirs(os.path.join(self.py, "ament_index_python"))
        write_exec(os.path.join(self.py, "ament_index_python", "__init__.py"), "")
        write_exec(os.path.join(self.py, "ament_index_python", "packages.py"),
                   "import json, os\n\ndef get_package_share_directory(name):\n"
                   "    return json.loads(os.environ.get('FAKE_SHARE_JSON', '{}'))[name]\n")
        # a fake TurtleBot3 installation
        gz = os.path.join(t, "turtlebot3_gazebo")
        os.makedirs(os.path.join(gz, "urdf"))
        os.makedirs(os.path.join(gz, "models", "turtlebot3_burger"))
        write_exec(os.path.join(gz, "urdf", "turtlebot3_burger.urdf"), "<robot/>")
        write_exec(os.path.join(gz, "models", "turtlebot3_burger", "model.sdf"), "<sdf/>")
        nav = os.path.join(t, "turtlebot3_navigation2")
        os.makedirs(os.path.join(nav, "param"))
        write_exec(os.path.join(nav, "param", "burger.yaml"), "amcl: {}\n")
        self.share = {"turtlebot3_gazebo": gz, "turtlebot3_navigation2": nav}
        # what the script reads to know where it runs: a plain Linux computer unless a test says otherwise
        self.dockerenv = os.path.join(t, "dockerenv")                  # does not exist = not in Docker
        self.proc_version = os.path.join(t, "proc_version")
        write_exec(self.proc_version, "Linux version 6.1.0 (gcc)\n")
        self.ros2_log = os.path.join(t, "ros2.log")

    def tearDown(self):
        self.tmp.cleanup()

    def env(self, srv=None, **extra):
        env = {k: v for k, v in os.environ.items()
               if k not in ("ROS_DISTRO", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "OLLAMA_HOST", "OLLAMA_MODEL",
                            "TURTLEBOT3_MODEL", "DISPLAY", "WAYLAND_DISPLAY")}
        env.update(PATH=self.bin + os.pathsep + env["PATH"], PYTHONPATH=self.py, ROS_DISTRO="humble",
                   FAKE_SHARE_JSON=json.dumps(self.share), NO_COLOR="1", FAKE_TOPICS=self.TOPICS,
                   FAKE_SERVICES=self.SERVICES, FAKE_ACTIONS=self.ACTIONS, FAKE_NODES=self.NODES,
                   DOCKERENV_FILE=self.dockerenv, PROC_VERSION_FILE=self.proc_version, FAKE_ROS2_LOG=self.ros2_log,
                   PREFLIGHT_GRAPH_WAIT="0")
        if srv is not None:
            env["OLLAMA_HOST"] = srv.url
        env.update(extra)
        return env

    def preflight(self, *args, srv=None, **extra):
        return run(["bash", PREFLIGHT] + list(args), self.env(srv, **extra))

    # -------------------------------------------------------------------------
    def test_everything_fine(self):
        with FakeLLMServer() as srv:
            r = self.preflight(srv=srv)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("All checks passed", r.stdout)
        self.assertIn("[ OK ] Ollama is running, version 0.5.7", r.stdout)
        self.assertIn("[ OK ] model qwen2.5:3b-instruct is installed", r.stdout)
        self.assertIn("[ OK ] package nav2_bringup", r.stdout)
        self.assertIn("TurtleBot3 files found", r.stdout)

    def test_no_llm_option_skips_ollama(self):
        r = self.preflight("--no-llm")             # no server at all
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertNotIn("Ollama", r.stdout)

    def test_ros_not_loaded(self):
        r = self.preflight("--no-llm", ROS_DISTRO="")
        self.assertGreaterEqual(r.returncode, 1)
        self.assertIn("ROS 2 is not loaded", r.stdout)
        self.assertIn("source /opt/ros/", r.stdout)

    def test_missing_packages_are_listed_with_the_install_command(self):
        r = self.preflight("--no-llm", FAKE_MISSING_PKGS="nav2_bringup turtlebot3_gazebo")
        self.assertIn("package nav2_bringup is missing", r.stdout)
        self.assertIn("package turtlebot3_gazebo is missing", r.stdout)
        self.assertIn("sudo apt install ros-humble-turtlebot3-gazebo", r.stdout)
        self.assertEqual(r.returncode, 2)

    def test_foxy_and_unknown_distros(self):
        r = self.preflight("--no-llm", ROS_DISTRO="foxy")
        self.assertIn("ROS 2 foxy", r.stdout)
        r = self.preflight("--no-llm", ROS_DISTRO="jazzy")
        self.assertIn("[warn] ROS 2 jazzy", r.stdout)
        self.assertIn("Docker", r.stdout)

    def test_turtlebot_files_missing(self):
        self.share = {}
        r = self.preflight("--no-llm")
        self.assertGreaterEqual(r.returncode, 1)
        self.assertIn("turtlebot3_gazebo not found", r.stdout)

    def test_python_environment_is_a_warning_not_an_error(self):
        r = self.preflight("--no-llm", CONDA_DEFAULT_ENV="deeplearning")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("[warn] a Python environment is active (deeplearning)", r.stdout)
        self.assertIn("conda deactivate", r.stdout)

    def test_ollama_not_running(self):
        r = self.preflight(OLLAMA_HOST="http://127.0.0.1:9")
        self.assertGreaterEqual(r.returncode, 1)
        self.assertIn("Ollama does not answer", r.stdout)
        self.assertIn("ollama serve", r.stdout)

    def test_ollama_without_the_model(self):
        with FakeLLMServer() as srv:
            srv.models = ["llama3.2:latest"]
            r = self.preflight(srv=srv)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("model qwen2.5:3b-instruct is not installed (installed: llama3.2:latest)", r.stdout)
        self.assertIn("ollama pull qwen2.5:3b-instruct", r.stdout)

    def test_other_model_name_and_latest_tag(self):
        with FakeLLMServer() as srv:
            srv.models = ["llama3.2:latest"]
            r = self.preflight(srv=srv, OLLAMA_MODEL="llama3.2:latest")
            self.assertEqual(r.returncode, 0, r.stdout)
            r = self.preflight(srv=srv, OLLAMA_MODEL="llama3.2")
            self.assertEqual(r.returncode, 0, r.stdout)

    def test_old_ollama_cannot_do_json_schema(self):
        with FakeLLMServer() as srv:
            srv.version = "0.4.7"
            r = self.preflight(srv=srv)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("too old", r.stdout)
        with FakeLLMServer() as srv:
            srv.version = "0.12.3"
            self.assertEqual(self.preflight(srv=srv).returncode, 0)     # 0.12 is newer than 0.5 (not a text compare)
        with FakeLLMServer() as srv:
            srv.version = "0.5.0"
            self.assertEqual(self.preflight(srv=srv).returncode, 0)

    def test_ollama_address_without_scheme(self):
        with FakeLLMServer() as srv:
            r = self.preflight(OLLAMA_HOST=srv.url.replace("http://", ""))
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_runtime_all_fine(self):
        r = self.preflight("--no-llm", "--runtime")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("service /gazebo/set_entity_state (Gazebo state)", r.stdout)
        self.assertIn("action /navigate_to_pose", r.stdout)
        self.assertIn("AMCL publishes the robot pose", r.stdout)
        self.assertIn("TF map -> base_footprint exists", r.stdout)

    def test_runtime_missing_things(self):
        r = self.preflight("--no-llm", "--runtime", FAKE_SERVICES="/gazebo/get_entity_state", FAKE_ACTIONS="",
                           FAKE_NODES="/amcl")
        self.assertIn("service /gazebo/set_entity_state is missing", r.stdout)
        self.assertIn("libgazebo_ros_state.so", r.stdout)
        self.assertIn("action /navigate_to_pose is missing", r.stdout)
        self.assertIn("node /bt_navigator is missing", r.stdout)
        self.assertEqual(r.returncode, 1 + 1 + 3)           # service, action, three nodes

    def test_runtime_with_and_without_the_forklift(self):
        r = self.preflight("--no-llm", "--runtime")
        self.assertEqual(r.returncode, 0, r.stdout)                     # no forklift is only a warning
        self.assertIn("[warn] the robot has no forklift", r.stdout)
        r = self.preflight("--no-llm", "--runtime", FAKE_TOPICS=self.TOPICS + " /arm/set_joint_trajectory")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("[ OK ] forklift: lift (/arm/set_joint_trajectory)", r.stdout)
        # the topic is there only because the agent publishes to it: nothing in Gazebo listens
        r = self.preflight("--no-llm", "--runtime", FAKE_TOPICS=self.TOPICS + " /arm/set_joint_trajectory",
                           FAKE_LIFT_LISTENERS="0")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("[warn] the robot has no forklift (nothing listens", r.stdout)

    def test_runtime_asks_again_when_the_daemon_has_just_started(self):
        r = self.preflight("--no-llm", "--runtime", FAKE_SLOW_DAEMON="1")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("node /bt_navigator", r.stdout)
        self.assertEqual(read_text(self.ros2_log).splitlines().count("topic list"), 2)

    def test_runtime_gives_up_after_a_few_looks(self):
        r = self.preflight("--no-llm", "--runtime", FAKE_ACTIONS="")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertEqual(read_text(self.ros2_log).splitlines().count("topic list"), 5)

    def test_runtime_nothing_is_running(self):
        r = self.preflight("--no-llm", "--runtime", FAKE_TOPICS="")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("nothing is running", r.stdout)
        self.assertIn("ros2 launch actr_lab1 sim.launch.py", r.stdout)

    def test_runtime_no_localization_yet_is_only_a_warning(self):
        r = self.preflight("--no-llm", "--runtime", FAKE_NO_AMCL="1")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("[warn] no /amcl_pose message", r.stdout)

    def test_runtime_checks_work_with_foxy_which_has_no_echo_once(self):
        r = self.preflight("--no-llm", "--runtime")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("[ OK ] /clock is running", r.stdout)
        calls = read_text(self.ros2_log)
        self.assertNotIn("--once", calls)
        self.assertIn("topic echo /clock", calls)

    def test_amcl_pose_is_read_with_the_qos_of_the_publisher(self):
        """/amcl_pose is 'transient local': a reader with the default QoS would never see the saved pose."""
        self.preflight("--no-llm", "--runtime")
        amcl = [line for line in read_text(self.ros2_log).splitlines() if line.endswith("/amcl_pose")]
        self.assertEqual(len(amcl), 1)
        self.assertIn("--qos-durability transient_local", amcl[0])
        self.assertIn("--qos-reliability reliable", amcl[0])

    def test_no_clock_message_is_a_failure(self):
        r = self.preflight("--no-llm", "--runtime", FAKE_TOPICS=self.TOPICS)
        self.assertEqual(r.returncode, 0, r.stdout)
        # a ros2 whose `topic echo` prints nothing at all: /clock gives no messages
        write_exec(os.path.join(self.bin, "ros2"), self.FAKE_ROS2.replace('echo "header:"; echo "  stamp:" ;;', ';;'))
        r = self.preflight("--no-llm", "--runtime")
        self.assertIn("[FAIL] /clock gives no messages", r.stdout)
        self.assertIn("[warn] no /amcl_pose message", r.stdout)

    # ---------------------------------------------------- where the script runs
    def test_inside_docker_on_windows_with_wsl2(self):
        write_exec(self.dockerenv, "")
        write_exec(self.proc_version, "Linux version 5.15.167.4-microsoft-standard-WSL2 (gcc)\n")
        r = self.preflight("--no-llm")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("running inside a Docker container", r.stdout)
        self.assertIn("Windows with WSL2", r.stdout)
        self.assertIn("[warn] no graphical display", r.stdout)
        self.assertIn("docker/run.sh from a WSL terminal", r.stdout)
        self.assertNotIn("wsl --update", r.stdout)

    def test_inside_docker_with_a_display_there_is_no_display_warning(self):
        write_exec(self.dockerenv, "")
        r = self.preflight("--no-llm", DISPLAY=":0")
        self.assertIn("running inside a Docker container", r.stdout)
        self.assertNotIn("no graphical display", r.stdout)

    def test_wsl_without_docker(self):
        write_exec(self.proc_version, "Linux version 5.15.153.1-microsoft-standard-WSL2 (gcc)\n")
        r = self.preflight("--no-llm")
        self.assertIn("running in WSL", r.stdout)
        self.assertNotIn("Docker", r.stdout)
        self.assertIn("wsl --update", r.stdout)

    def test_plain_linux_has_no_wsl_or_docker_lines(self):
        r = self.preflight("--no-llm")
        self.assertNotIn("WSL", r.stdout)
        self.assertNotIn("Docker container", r.stdout)

    def test_inside_docker_ollama_is_looked_for_in_the_usual_places(self):
        write_exec(self.dockerenv, "")
        with FakeLLMServer() as srv:
            r = self.preflight(OLLAMA_HOST="http://127.0.0.1:9", PREFLIGHT_OLLAMA_CANDIDATES=srv.url, DISPLAY=":0")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn(f"[warn] Ollama answers at {srv.url}, not at http://127.0.0.1:9", r.stdout)
        self.assertIn("export OLLAMA_HOST=" + srv.url, r.stdout)
        self.assertIn("[ OK ] model qwen2.5:3b-instruct is installed", r.stdout)

    def test_inside_docker_a_missing_ollama_points_to_llm_start_not_to_serve(self):
        write_exec(self.dockerenv, "")
        r = self.preflight(OLLAMA_HOST="http://127.0.0.1:9", PREFLIGHT_OLLAMA_CANDIDATES="", DISPLAY=":0")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("Ollama does not answer", r.stdout)
        self.assertIn("docker/run.sh llm-start", r.stdout)
        self.assertIn("host.docker.internal", r.stdout)
        self.assertNotIn("ollama serve", r.stdout)

    def test_unknown_option(self):
        r = self.preflight("--bogus")
        self.assertEqual(r.returncode, 64)
        self.assertIn("unknown option", r.stdout)

    def test_the_script_changes_nothing(self):
        before = sorted(os.listdir(REPO))
        with FakeLLMServer() as srv:
            self.preflight("--runtime", srv=srv)
        self.assertEqual(sorted(os.listdir(REPO)), before)


if __name__ == "__main__":
    unittest.main()
