#!/usr/bin/env bash
# Runs the lab in Docker: Docker Desktop on Windows 11 + WSL2 (start this script in a WSL terminal,
# not in PowerShell), or Docker on Linux.
#
#   docker/run.sh build                  build the image (the first time: 10-30 minutes, several GB)
#   docker/run.sh llm-start              start Ollama (the local LLM) in its own container
#   docker/run.sh llm-pull [model]       download a model (default: qwen2.5:3b-instruct)
#   docker/run.sh llm-stop               stop Ollama (the downloaded models stay)
#   docker/run.sh check                  tools/preflight.sh in a new container: is everything installed?
#   docker/run.sh check --runtime        tools/preflight.sh INSIDE the running lab container: does the
#                                        simulation work? (start the simulation first, in another terminal)
#   docker/run.sh shell                  a shell inside the lab container (this is the default); if the lab
#                                        container is already running: a second shell in the same container
#   docker/run.sh run <command...>       any command in a new lab container, for example
#        docker/run.sh run ros2 launch actr_lab1 all.launch.py gui:=false planner:=mock
#   docker/run.sh exec <command...>      a command in the lab container that is already running
#        (a second terminal:  docker/run.sh exec ros2 topic list)
#   docker/run.sh test                   the unit tests of the package
#
# Variables:
#   LAB_ROS_DISTRO  humble (default) or foxy. (The variable ROS_DISTRO of this computer is NOT used.)
#   OLLAMA_HOST  where the LLM is. Default: the container started by llm-start (http://actr_ollama:11434).
#                Ollama as a Windows app: http://host.docker.internal:11434
#   OLLAMA_PORT  port published by llm-start on this computer (default 11434)
#   GPU=1        llm-start: give the NVIDIA GPU to Ollama
#   DEV=1        use the source folder of this computer inside the container (edit code without a new build)
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER="${DOCKER:-docker}"
LAB_ROS_DISTRO="${LAB_ROS_DISTRO:-humble}"      # not ROS_DISTRO: if ROS is installed here, that one is "jazzy" or something else
IMAGE="actr_lab1:${LAB_ROS_DISTRO}"
NET="actr_net"
LAB="actr_lab"
LLM="actr_ollama"
LLM_PORT="${OLLAMA_PORT:-11434}"
LLM_MODEL_DEFAULT="qwen2.5:3b-instruct"
WSLG_DIR="${WSLG_DIR:-/mnt/wslg}"
X11_DIR="${X11_DIR:-/tmp/.X11-unix}"

die() { echo "error: $*" >&2; exit 1; }

need_docker() {
  command -v "$DOCKER" >/dev/null 2>&1 || die "docker not found. Install Docker Desktop and turn on 'WSL integration' for this distro (Settings > Resources > WSL integration)."
  "$DOCKER" info >/dev/null 2>&1 || die "docker does not answer. Start Docker Desktop (or the docker service) and try again."
}

need_image() {
  "$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1 || die "the image $IMAGE does not exist yet. Run:  docker/run.sh build"
}

ensure_network() {
  "$DOCKER" network inspect "$NET" >/dev/null 2>&1 || "$DOCKER" network create "$NET" >/dev/null || die "cannot create the network $NET"
}

is_running() { "$DOCKER" ps --format '{{.Names}}' | grep -qx "$1"; }

# The name "actr_lab" can be used by one container at a time. Docker's own message ("name already in use")
# is hard to understand, so we explain what to do.
free_lab_name() {
  if is_running "$LAB"; then
    die "the lab container ($LAB) is already running (in another terminal?). Use it:  docker/run.sh exec <command>   (a second shell:  docker/run.sh shell). To start from zero:  docker rm -f $LAB"
  fi
  "$DOCKER" rm -f "$LAB" >/dev/null 2>&1      # a stopped leftover with the same name (after a crash)
}

# -i or -it, as the terminal allows
tty_flag() { if [ -t 0 ]; then echo -it; else echo -i; fi; }

# arguments for `docker run` of a lab container (built in an array, so spaces are safe).
#   lab_args named       the container is called actr_lab (the simulation; other terminals can exec into it)
#   lab_args anonymous   no name (tests and checks: they must not collide with the simulation)
lab_args() {
  ARGS=(--rm --network "$NET" --shm-size=1g --add-host=host.docker.internal:host-gateway
        --user "$(id -u):$(id -g)" -e HOME=/tmp/actr_home
        -e "OLLAMA_HOST=${OLLAMA_HOST:-http://${LLM}:11434}"
        -v "$REPO/results:/ws/results" -v "$HOME/actr_lab1_logs:/tmp/actr_home/actr_lab1_logs")
  if [ "${1:-named}" = named ]; then ARGS+=(--name "$LAB"); fi
  ARGS+=("$(tty_flag)")
  if [ -d "$WSLG_DIR" ]; then              # WSLg (Windows 11): windows of Linux programs appear on the Windows desktop
    ARGS+=(-v "$X11_DIR:/tmp/.X11-unix" -v "$WSLG_DIR:/mnt/wslg"
           -e "DISPLAY=${DISPLAY:-:0}" -e "WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}"
           -e "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
           -e "PULSE_SERVER=${PULSE_SERVER:-unix:/mnt/wslg/PulseServer}")
  elif [ -n "${DISPLAY:-}" ]; then          # a normal Linux desktop (you may need: xhost +local:)
    ARGS+=(-v "$X11_DIR:/tmp/.X11-unix" -e "DISPLAY=$DISPLAY")
  fi
  if [ "${DEV:-0}" = 1 ]; then ARGS+=(-v "$REPO/src:/ws/src"); fi
}

prepare_folders() {
  mkdir -p "$REPO/results" "$HOME/actr_lab1_logs"
}

cmd="${1:-shell}"
[ $# -gt 0 ] && shift

case "$cmd" in
  build)
    need_docker
    cd "$REPO" || die "cannot enter $REPO"
    exec "$DOCKER" build -f docker/Dockerfile --build-arg "ROS_DISTRO=$LAB_ROS_DISTRO" -t "$IMAGE" .
    ;;

  llm-start)
    need_docker
    ensure_network
    if "$DOCKER" ps --format '{{.Names}}' | grep -qx "$LLM"; then
      echo "Ollama is already running (container $LLM)."
    else
      "$DOCKER" rm -f "$LLM" >/dev/null 2>&1
      gpu=()
      if [ "${GPU:-0}" = 1 ]; then gpu=(--gpus all); fi
      "$DOCKER" run -d --name "$LLM" --network "$NET" -p "${LLM_PORT}:11434" -v actr_ollama_models:/root/.ollama \
        ${gpu[@]+"${gpu[@]}"} ollama/ollama >/dev/null \
        || die "could not start Ollama. If the port ${LLM_PORT} is busy, you already run Ollama on this computer: then you do not need llm-start (set OLLAMA_HOST, see the guide), or use OLLAMA_PORT=11500."
      echo "Ollama started. On this computer: http://localhost:${LLM_PORT}  Inside the lab container: http://${LLM}:11434"
    fi
    echo "Next: docker/run.sh llm-pull    (downloads $LLM_MODEL_DEFAULT, about 2 GB)"
    ;;

  llm-pull)
    need_docker
    model="${1:-$LLM_MODEL_DEFAULT}"
    "$DOCKER" ps --format '{{.Names}}' | grep -qx "$LLM" || die "Ollama is not running. Run:  docker/run.sh llm-start"
    exec "$DOCKER" exec -it "$LLM" ollama pull "$model"
    ;;

  llm-stop)
    need_docker
    "$DOCKER" rm -f "$LLM" >/dev/null 2>&1
    echo "Ollama stopped (the models are kept in the volume actr_ollama_models)."
    ;;

  check)
    need_docker
    runtime=0
    for a in "$@"; do [ "$a" = --runtime ] && runtime=1; done
    if [ "$runtime" = 1 ]; then
      # The ROS graph of the simulation lives in the lab container: look at it from the inside.
      is_running "$LAB" || die "--runtime looks at the simulation that is running, but no lab container is running. Start it first (in another terminal):  docker/run.sh run ros2 launch actr_lab1 sim.launch.py gui:=false"
      exec "$DOCKER" exec "$(tty_flag)" "$LAB" /ws/docker/entrypoint.sh /ws/tools/preflight.sh "$@"
    fi
    need_image; ensure_network; prepare_folders
    lab_args anonymous
    exec "$DOCKER" run "${ARGS[@]}" "$IMAGE" /ws/tools/preflight.sh "$@"
    ;;

  shell)
    need_docker
    if is_running "$LAB"; then
      echo "The lab container ($LAB) is already running: opening a second shell in it." >&2
      exec "$DOCKER" exec "$(tty_flag)" "$LAB" /ws/docker/entrypoint.sh bash
    fi
    need_image; ensure_network; prepare_folders; free_lab_name
    lab_args named
    exec "$DOCKER" run "${ARGS[@]}" "$IMAGE" bash
    ;;

  run)
    [ $# -gt 0 ] || die "run needs a command, for example: docker/run.sh run ros2 launch actr_lab1 all.launch.py"
    need_docker; need_image; ensure_network; prepare_folders; free_lab_name
    lab_args named
    exec "$DOCKER" run "${ARGS[@]}" "$IMAGE" "$@"
    ;;

  exec)
    [ $# -gt 0 ] || die "exec needs a command, for example: docker/run.sh exec ros2 topic list"
    need_docker
    is_running "$LAB" || die "no lab container is running. Start one with: docker/run.sh run ... (or shell)"
    exec "$DOCKER" exec "$(tty_flag)" "$LAB" /ws/docker/entrypoint.sh "$@"
    ;;

  test)
    need_docker; need_image; prepare_folders
    lab_args anonymous
    exec "$DOCKER" run "${ARGS[@]}" -w /ws/src/actr_lab1 "$IMAGE" python3 -m unittest discover -s test
    ;;

  -h|--help|help)
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    ;;

  *)
    die "unknown command '$cmd' (see: docker/run.sh help)"
    ;;
esac
