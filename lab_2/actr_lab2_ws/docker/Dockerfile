# ROS 2 + Gazebo Classic + Nav2 + TurtleBot3 + the code of this lab.
# Build it with:  docker/run.sh build        (the first time it takes a while and uses several GB)
#
# ROS_DISTRO can be humble (Ubuntu 22.04, the default) or foxy (Ubuntu 20.04, the version in the course link).
ARG ROS_DISTRO=humble
FROM osrf/ros:${ROS_DISTRO}-desktop
ARG ROS_DISTRO

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=${ROS_DISTRO} \
    TURTLEBOT3_MODEL=burger \
    PYTHONUNBUFFERED=1 \
    LIBGL_ALWAYS_SOFTWARE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-navigation2 \
        ros-${ROS_DISTRO}-nav2-bringup \
        ros-${ROS_DISTRO}-turtlebot3-gazebo \
        ros-${ROS_DISTRO}-turtlebot3-navigation2 \
        ros-${ROS_DISTRO}-gazebo-ros-pkgs \
        python3-colcon-common-extensions \
        python3-yaml python3-numpy python3-matplotlib \
        curl ca-certificates iproute2 mesa-utils x11-apps \
    && rm -rf /var/lib/apt/lists/*

# The container runs as the user of the host computer (docker/run.sh passes --user and HOME=/tmp/actr_home).
# This folder must exist and be writable for any user: ROS, Gazebo and our logs write in the home folder, and
# the bind-mounted log folder (docker -v ...:/tmp/actr_home/actr_lab1_logs) would otherwise create it as root.
RUN install -d -m 1777 /tmp/actr_home

WORKDIR /ws
COPY src /ws/src
COPY tools /ws/tools
COPY docker/entrypoint.sh /ws/docker/entrypoint.sh
RUN chmod +x /ws/docker/entrypoint.sh /ws/tools/preflight.sh \
 && . /opt/ros/${ROS_DISTRO}/setup.sh \
 && colcon build --symlink-install \
 && chmod -R a+rX /ws

ENTRYPOINT ["/ws/docker/entrypoint.sh"]
CMD ["bash"]
