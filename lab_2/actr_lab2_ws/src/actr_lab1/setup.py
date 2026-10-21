import os
from glob import glob

from setuptools import find_packages, setup

package_name = "actr_lab1"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*")),
        (os.path.join("share", package_name, "maps"), glob("maps/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pedro",
    maintainer_email="pfuentessantander@gmail.com",
    description="ACT-R style agent with a local LLM planner for a TurtleBot3 in Gazebo (Nav2)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "agent_node = actr_lab1.ros.agent_node:main",
            "actr_demo = actr_lab1.demo:main",
            "actr_ablation = actr_lab1.experiments.ablation:main",
            "actr_compare = actr_lab1.experiments.compare_planners:main",
            "actr_learning = actr_lab1.experiments.learning:main",
        ],
    },
)
