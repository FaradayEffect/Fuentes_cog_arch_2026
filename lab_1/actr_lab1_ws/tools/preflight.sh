#!/usr/bin/env bash
# Check list for the ROS 2 / Gazebo / Nav2 / Ollama part of the lab. It changes nothing.
#
#   tools/preflight.sh                # the computer: ROS 2, Gazebo, TurtleBot3, our package, Ollama
#   tools/preflight.sh --runtime      # also: is the simulation running? (topics, services, Nav2, TF)
#   tools/preflight.sh --no-llm       # skip the Ollama check
#
# Exit code = number of problems (0 = all fine). Warnings are not counted.
# Environment: OLLAMA_HOST (default http://localhost:11434), OLLAMA_MODEL (default qwen2.5:3b-instruct),
#              TURTLEBOT3_MODEL (default burger), NO_COLOR=1 for plain text.
# It also works inside the Docker container (docker/run.sh check): it knows that it is in a container.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_SRC="$REPO/src/actr_lab1"
RUNTIME=0
CHECK_LLM=1
for arg in "$@"; do
  case "$arg" in
    --runtime) RUNTIME=1 ;;
    --no-llm) CHECK_LLM=0 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)"; exit 64 ;;
  esac
done

PROBLEMS=0
WARNINGS=0
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; B=$'\033[1m'; N=$'\033[0m'
else
  G=""; R=""; Y=""; B=""; N=""
fi
ok()   { printf '  [%s OK %s] %s\n' "$G" "$N" "$*"; }
bad()  { printf '  [%sFAIL%s] %s\n' "$R" "$N" "$*"; PROBLEMS=$((PROBLEMS + 1)); }
warn() { printf '  [%swarn%s] %s\n' "$Y" "$N" "$*"; WARNINGS=$((WARNINGS + 1)); }
fix()  { printf '         fix: %s\n' "$*"; }
title() { printf '\n%s%s%s\n' "$B" "$*" "$N"; }
have() { command -v "$1" >/dev/null 2>&1; }
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }   # $1 >= $2

DISTRO="${ROS_DISTRO:-}"
MODEL="${TURTLEBOT3_MODEL:-burger}"
OLLAMA="${OLLAMA_HOST:-http://localhost:11434}"
case "$OLLAMA" in http://*|https://*) ;; *) OLLAMA="http://$OLLAMA" ;; esac
LLM_MODEL="${OLLAMA_MODEL:-qwen2.5:3b-instruct}"

# ---------------------------------------------------------------------------- system
title "1. System"
if [ -r /etc/os-release ]; then
  . /etc/os-release
  ok "operating system: ${PRETTY_NAME:-unknown}"
else
  warn "cannot read /etc/os-release"
fi
# (the two files can be changed with the variables DOCKERENV_FILE and PROC_VERSION_FILE: only for the tests)
IN_DOCKER=0
if [ -f "${DOCKERENV_FILE:-/.dockerenv}" ]; then
  IN_DOCKER=1
  ok "running inside a Docker container"
fi
IN_WSL=0
if grep -qi microsoft "${PROC_VERSION_FILE:-/proc/version}" 2>/dev/null; then
  IN_WSL=1
  if [ "$IN_DOCKER" = 1 ]; then
    ok "the computer is Windows with WSL2 (a container uses the kernel of WSL2)"
  else
    ok "running in WSL"
  fi
fi
if [ "$IN_WSL" = 1 ] || [ "$IN_DOCKER" = 1 ]; then
  if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    warn "no graphical display (DISPLAY is empty): the Gazebo window will not open"
    if [ "$IN_DOCKER" = 1 ]; then
      fix "start the container with docker/run.sh from a WSL terminal (it hands WSLg to the container), or use 'gui:=false'"
    else
      fix "use 'gui:=false' (no window), or update WSL ('wsl --update') so that WSLg works"
    fi
  fi
fi
if have python3; then
  PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
  ok "python3 $PYV"
else
  bad "python3 not found"
  PYV=""
fi
if [ -n "${CONDA_DEFAULT_ENV:-}" ] || [ -n "${VIRTUAL_ENV:-}" ]; then
  warn "a Python environment is active (${CONDA_DEFAULT_ENV:-$VIRTUAL_ENV}): ROS 2 needs the system Python"
  fix "conda deactivate   (or: deactivate) before you source ROS and start the launch files"
fi

# ---------------------------------------------------------------------------- ROS 2
title "2. ROS 2"
if [ -z "$DISTRO" ]; then
  bad "ROS 2 is not loaded (the variable ROS_DISTRO is empty)"
  fix "source /opt/ros/<humble|foxy>/setup.bash   (put it into ~/.bashrc)"
else
  case "$DISTRO" in
    humble) EXPECT_PY="3.10"; ok "ROS 2 $DISTRO (Ubuntu 22.04, Gazebo Classic 11): tested design" ;;
    foxy)   EXPECT_PY="3.8";  ok "ROS 2 $DISTRO (Ubuntu 20.04): the version in the course link" ;;
    *)      EXPECT_PY=""
            warn "ROS 2 $DISTRO: this lab was written for humble and foxy (Gazebo Classic). 'gazebo_ros' does not exist in jazzy."
            fix "use Ubuntu 22.04 with humble, or run it in Docker (docker/run.sh)" ;;
  esac
  if [ -n "$EXPECT_PY" ] && [ -n "$PYV" ] && [ "$PYV" != "$EXPECT_PY" ]; then
    warn "ROS 2 $DISTRO expects Python $EXPECT_PY, but python3 is $PYV"
  fi
fi

if have ros2; then
  ok "command ros2 found"
else
  bad "command ros2 not found"
  fix "source /opt/ros/$DISTRO/setup.bash"
fi

MISSING=""
for pkg in rclpy nav2_msgs nav2_bringup gazebo_ros gazebo_msgs lifecycle_msgs robot_state_publisher tf2_ros \
           turtlebot3_gazebo turtlebot3_navigation2 geometry_msgs; do
  if have ros2 && ros2 pkg prefix "$pkg" >/dev/null 2>&1; then
    ok "package $pkg"
  else
    bad "package $pkg is missing"
    MISSING="$MISSING $pkg"
  fi
done
if [ -n "$MISSING" ]; then
  fix "sudo apt install ros-${DISTRO:-\$ROS_DISTRO}-turtlebot3-gazebo ros-${DISTRO:-\$ROS_DISTRO}-turtlebot3-navigation2 ros-${DISTRO:-\$ROS_DISTRO}-navigation2 ros-${DISTRO:-\$ROS_DISTRO}-nav2-bringup ros-${DISTRO:-\$ROS_DISTRO}-gazebo-ros-pkgs"
fi

for mod in rclpy yaml; do
  if python3 -c "import $mod" >/dev/null 2>&1; then
    ok "python can import $mod"
  else
    bad "python cannot import $mod"
    if [ "$mod" = rclpy ]; then fix "source /opt/ros/$DISTRO/setup.bash in THIS terminal; no conda environment"; else fix "sudo apt install python3-yaml"; fi
  fi
done
for mod in matplotlib numpy; do
  if python3 -c "import $mod" >/dev/null 2>&1; then
    ok "python can import $mod (pictures of the experiments)"
  else
    warn "python cannot import $mod (only the pictures of the experiments need it)"
    fix "sudo apt install python3-matplotlib python3-numpy"
  fi
done

# ---------------------------------------------------------------------------- Gazebo + TurtleBot3
title "3. Gazebo and TurtleBot3 ($MODEL)"
if have gzserver; then
  ok "gzserver found: $(gzserver --version 2>/dev/null | head -n1)"
else
  bad "gzserver (Gazebo Classic) not found"
  fix "sudo apt install ros-${DISTRO:-\$ROS_DISTRO}-gazebo-ros-pkgs"
fi
if have python3; then
  FILES="$(PYTHONPATH="$PKG_SRC:${PYTHONPATH:-}" python3 - "$MODEL" <<'PY' 2>&1
import sys
try:
    from actr_lab1.ros.launch_utils import tb3_files
    files = tb3_files(sys.argv[1])
    print("OK " + ", ".join(k + "=" + v for k, v in sorted(files.items())))
except Exception as e:
    print("FAIL " + str(e).replace("\n", " "))
PY
)"
  case "$FILES" in
    "OK "*) ok "TurtleBot3 files found for the launch file"
            printf '         %s\n' "${FILES#OK }" ;;
    *)      bad "${FILES#FAIL }" ;;
  esac
fi
if [ "$MODEL" != "burger" ]; then
  warn "TURTLEBOT3_MODEL=$MODEL: only 'burger' is supported (the launch file sets it to burger)"
fi

# ---------------------------------------------------------------------------- our package
title "4. This package (actr_lab1)"
if have ros2 && ros2 pkg prefix actr_lab1 >/dev/null 2>&1; then
  ok "actr_lab1 is built and found by ROS: $(ros2 pkg prefix actr_lab1 2>/dev/null)"
else
  bad "actr_lab1 is not found by ROS (not built, or this terminal did not source the workspace)"
  fix "cd $REPO && colcon build --symlink-install && source install/setup.bash"
fi
if [ -f "$PKG_SRC/config/places.json" ] && [ -f "$PKG_SRC/worlds/rooms.world" ] && [ -f "$PKG_SRC/maps/rooms.yaml" ]; then
  ok "world, map and places are in the source folder"
else
  bad "world / map / places.json are missing in $PKG_SRC"
  fix "python3 $REPO/tools/generate_world.py"
fi

# ---------------------------------------------------------------------------- Ollama
if [ "$CHECK_LLM" = 1 ]; then
  title "5. Local LLM (Ollama at $OLLAMA)"
  http_get() {
    if have curl; then
      curl -fsS --max-time 4 --noproxy '*' "$1" 2>/dev/null
    else
      python3 - "$1" <<'PY' 2>/dev/null
import sys, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
print(opener.open(sys.argv[1], timeout=4).read().decode())
PY
    fi
  }
  VERSION_JSON="$(http_get "$OLLAMA/api/version")"
  if [ -z "$VERSION_JSON" ]; then
    # Maybe it runs somewhere else: look at the usual places and say where it is.
    CANDIDATES=""
    if [ "$IN_DOCKER" = 1 ]; then
      CANDIDATES="http://actr_ollama:11434 http://host.docker.internal:11434"
    elif [ "$IN_WSL" = 1 ] && have ip; then
      WIN_IP="$(ip route show default 2>/dev/null | awk '{print $3; exit}')"
      [ -n "$WIN_IP" ] && CANDIDATES="http://$WIN_IP:11434"
    fi
    if [ -n "${PREFLIGHT_OLLAMA_CANDIDATES+x}" ]; then CANDIDATES="$PREFLIGHT_OLLAMA_CANDIDATES"; fi   # (for the tests)
    for CAND in $CANDIDATES; do
      if [ "$CAND" != "$OLLAMA" ] && [ -n "$(http_get "$CAND/api/version")" ]; then
        warn "Ollama answers at $CAND, not at $OLLAMA"
        fix "export OLLAMA_HOST=$CAND   and pass llm_host:=$CAND   (docker/run.sh passes OLLAMA_HOST to the container)"
        OLLAMA="$CAND"
        VERSION_JSON="$(http_get "$OLLAMA/api/version")"
        break
      fi
    done
  fi
  if [ -z "$VERSION_JSON" ]; then
    bad "Ollama does not answer at $OLLAMA"
    if [ "$IN_DOCKER" = 1 ]; then
      fix "on the computer (not in the container):  docker/run.sh llm-start   and then   docker/run.sh llm-pull"
      fix "Ollama as a Windows app? start docker/run.sh with OLLAMA_HOST=http://host.docker.internal:11434 (and set OLLAMA_HOST=0.0.0.0 in Windows, then restart Ollama)"
    else
      fix "start it:  ollama serve     (install: https://ollama.com/download)"
      if [ "$IN_WSL" = 1 ]; then
        fix "Ollama on Windows? set OLLAMA_HOST=0.0.0.0 on Windows and restart it, then use the Windows IP"
      fi
    fi
  else
    VERSION="$(printf '%s' "$VERSION_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null)"
    ok "Ollama is running, version ${VERSION:-unknown}"
    if [ -n "$VERSION" ] && ! version_ge "$VERSION" "0.5.0"; then
      bad "Ollama $VERSION is too old for answers with a JSON schema (needs 0.5.0 or newer)"
      fix "update Ollama (https://ollama.com/download)"
    fi
    TAGS="$(http_get "$OLLAMA/api/tags" | python3 -c 'import json,sys; print(" ".join(m["name"] for m in json.load(sys.stdin).get("models", [])))' 2>/dev/null)"
    FOUND=0
    for name in $TAGS; do
      if [ "$name" = "$LLM_MODEL" ] || [ "$name" = "$LLM_MODEL:latest" ]; then FOUND=1; fi
    done
    if [ "$FOUND" = 1 ]; then
      ok "model $LLM_MODEL is installed"
    else
      bad "model $LLM_MODEL is not installed (installed: ${TAGS:-nothing})"
      fix "ollama pull $LLM_MODEL"
    fi
  fi
fi

# ---------------------------------------------------------------------------- runtime
if [ "$RUNTIME" = 1 ]; then
  title "6. The simulation that is running now"
  if ! have ros2; then
    bad "ros2 not found: cannot look at the running system"
  else
    NEED_TOPICS="/clock /scan /odom /tf /cmd_vel /amcl_pose /map"
    NEED_SERVICES="/gazebo/get_entity_state /gazebo/set_entity_state"
    NEED_NODES="/amcl /bt_navigator /controller_server /planner_server"
    read_graph() {
      TOPICS="$(ros2 topic list 2>/dev/null)"
      SERVICES="$(ros2 service list 2>/dev/null)"
      ACTIONS="$(ros2 action list 2>/dev/null)"
      NODES="$(ros2 node list 2>/dev/null)"
    }
    graph_complete() {
      local item
      for item in $NEED_TOPICS; do printf '%s\n' "$TOPICS" | grep -qx "$item" || return 1; done
      for item in $NEED_SERVICES; do printf '%s\n' "$SERVICES" | grep -qx "$item" || return 1; done
      printf '%s\n' "$ACTIONS" | grep -qx "/navigate_to_pose" || return 1
      for item in $NEED_NODES; do printf '%s\n' "$NODES" | grep -qx "$item" || return 1; done
      return 0
    }
    # The ROS 2 daemon that the first `ros2 ... list` starts needs a few seconds to see the whole system:
    # just after it starts, the lists can be empty or incomplete. So we ask again before we say "missing".
    read_graph
    tries=1
    while [ -n "$TOPICS" ] && ! graph_complete && [ "$tries" -lt "${PREFLIGHT_GRAPH_TRIES:-5}" ]; do
      sleep "${PREFLIGHT_GRAPH_WAIT:-3}"
      read_graph
      tries=$((tries + 1))
    done
    if [ -z "$TOPICS" ]; then
      bad "no ROS topics at all: nothing is running in this terminal's ROS domain"
      fix "start the simulation first:  ros2 launch actr_lab1 sim.launch.py   (and check ROS_DOMAIN_ID)"
    else
      for t in $NEED_TOPICS; do
        if printf '%s\n' "$TOPICS" | grep -qx "$t"; then ok "topic $t"; else bad "topic $t is missing"; fi
      done
      for s in $NEED_SERVICES; do
        if printf '%s\n' "$SERVICES" | grep -qx "$s"; then ok "service $s (Gazebo state)"; else
          bad "service $s is missing"
          fix "the world file must load libgazebo_ros_state.so with namespace 'gazebo' (our worlds/rooms.world does)"
        fi
      done
      if printf '%s\n' "$ACTIONS" | grep -qx "/navigate_to_pose"; then ok "action /navigate_to_pose (Nav2)"; else
        bad "action /navigate_to_pose is missing (Nav2 is not running)"
        fix "ros2 launch actr_lab1 sim.launch.py nav2:=true"
      fi
      for n in $NEED_NODES; do
        if printf '%s\n' "$NODES" | grep -qx "$n"; then ok "node $n"; else bad "node $n is missing"; fi
      done
      # the forklift is optional: without it the agent uses the virtual gripper. Its lift plugin listens to
      # /arm/set_joint_trajectory (the agent only publishes there: a topic that nobody listens to is no forklift)
      LIFT_TOPIC=/arm/set_joint_trajectory
      LISTENERS=0
      if printf '%s\n' "$TOPICS" | grep -qx "$LIFT_TOPIC"; then
        LISTENERS="$(ros2 topic info "$LIFT_TOPIC" 2>/dev/null | sed -n 's/^Subscription count: *\([0-9]*\).*/\1/p' | head -n 1)"
        [ -n "$LISTENERS" ] || LISTENERS=1        # the count cannot be read: trust the topic list
      fi
      if [ "$LISTENERS" -gt 0 ]; then
        ok "forklift: lift ($LIFT_TOPIC)"
      else
        warn "the robot has no forklift (nothing listens to $LIFT_TOPIC): the agent will use the virtual gripper"
        fix "start the simulation with the forklift: ros2 launch actr_lab1 sim.launch.py arm:=true"
      fi
      if have timeout; then
        # `ros2 topic echo --once` does not exist in Foxy: read the first line of the output instead.
        # PYTHONUNBUFFERED: without it the text stays in the buffer of Python until `timeout` kills the program.
        first_line() {   # first_line SECONDS <arguments of ros2 topic echo>   ->  the first output line (or nothing)
          local seconds="$1"
          shift
          PYTHONUNBUFFERED=1 timeout "$seconds" ros2 topic echo "$@" 2>/dev/null | head -n 1
        }
        if [ -n "$(first_line 8 /clock)" ]; then ok "/clock is running"; else bad "/clock gives no messages"; fi
        # /amcl_pose is "transient local": the last pose is kept for late readers. Ask for it with the same QoS.
        if [ -n "$(first_line 10 --qos-durability transient_local --qos-reliability reliable /amcl_pose)" ]; then
          ok "AMCL publishes the robot pose"
        else
          warn "no /amcl_pose message in 10 s (the localization may still be starting)"
          fix "wait a little; or in RViz use '2D Pose Estimate'; the agent publishes /initialpose itself if TF is missing"
        fi
        STDBUF=""
        if have stdbuf; then STDBUF="stdbuf -oL"; fi
        if timeout 10 $STDBUF ros2 run tf2_ros tf2_echo map base_footprint 2>/dev/null | grep -q "Translation"; then
          ok "TF map -> base_footprint exists"
        else
          warn "no TF map -> base_footprint in 10 s"
        fi
      else
        warn "'timeout' is not installed: skipping the message checks"
      fi
    fi
  fi
fi

# ---------------------------------------------------------------------------- end
printf '\n'
if [ "$PROBLEMS" -eq 0 ]; then
  printf '%sAll checks passed%s (%d warning(s)).\n' "$G" "$N" "$WARNINGS"
else
  printf '%s%d problem(s)%s and %d warning(s). Fix the lines marked FAIL, then run this script again.\n' "$R" "$PROBLEMS" "$N" "$WARNINGS"
fi
exit "$PROBLEMS"
