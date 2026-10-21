#!/usr/bin/env bash
# Loads ROS 2 and the built workspace, then runs the command that was given.
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /ws/install/setup.bash
# the container runs as the user of the host: that user has no home folder here
export HOME="${HOME:-/tmp/actr_home}"
mkdir -p "$HOME" 2>/dev/null
exec "$@"
