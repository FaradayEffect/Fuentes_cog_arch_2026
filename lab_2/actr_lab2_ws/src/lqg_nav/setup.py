import os
from glob import glob

from setuptools import find_packages, setup

package_name = "lqg_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pedro",
    maintainer_email="pfuentessantander@gmail.com",
    description="Lab 2: an LQG controller (LQR + Kalman filter) that drives the TurtleBot3 of Lab 1 instead of the "
                "standard Nav2 controller",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lqg_navigator = lqg_nav.navigator_node:main",
            "track_monitor = lqg_nav.monitor_node:main",
            "lqg_experiment = lqg_nav.experiment:main",
        ],
    },
)
