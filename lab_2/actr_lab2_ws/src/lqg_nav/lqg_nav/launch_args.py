"""Launch arguments of Lab 2 (plain Python, so that the tests can read them without ROS)."""

# (name, default, type, help)
LQG_ARGS = [
    ("steering", "lqg", str, "steering law: lqg (LQR + Kalman filter), lqr (no filter) or pure_pursuit"),
    ("pose_noise", "0.0", float, "experiment: noise in metres added to the measured x and y of the robot (0 = off)"),
    ("yaw_noise", "0.0", float, "experiment: noise in radians added to the measured yaw (0 = off)"),
]

# who drives the robot -> the action that the agent calls
CONTROLLERS = {"lqg": "lqg/navigate_to_pose", "nav2": "navigate_to_pose"}
CONTROLLER_ARG = ("controller", "lqg", str, "who drives the robot: lqg (the controller of Lab 2) or nav2 "
                                            "(the standard Nav2 controller, to compare)")


def nav_action(controller: str) -> str:
    name = controller.strip().lower()
    if name not in CONTROLLERS:
        raise ValueError(f"unknown controller '{controller}': use " + " or ".join(CONTROLLERS))
    return CONTROLLERS[name]
