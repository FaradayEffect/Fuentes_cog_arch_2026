"""A fake of the parts of `launch`, `launch_ros` and `ament_index_python` that our launch files use.

The launch files run for real (their Python code is executed); only the classes they create are
simple recorders. So a typo, a wrong argument name or a missing file shows up in the tests.
It is NOT the real launch system: the real one may still dislike something.
"""
import contextlib
import importlib.util
import os
import sys
import types
from typing import Dict, List


class PackageNotFoundError(Exception):
    pass


class FakeContext:
    def __init__(self, configs=None):
        self.launch_configurations = dict(configs or {})


class LaunchDescription:
    def __init__(self, entities=None):
        self.entities = list(entities or [])


class DeclareLaunchArgument:
    def __init__(self, name, default_value=None, description=""):
        self.name, self.default_value, self.description = name, default_value, description


class LaunchConfiguration:
    def __init__(self, name):
        self.name = name

    def perform(self, context):
        return context.launch_configurations[self.name]


class PythonLaunchDescriptionSource:
    def __init__(self, path):
        self.path = path


class IncludeLaunchDescription:
    def __init__(self, launch_description_source, launch_arguments=None):
        self.source = launch_description_source
        self.arguments = dict(launch_arguments or {})


class OpaqueFunction:
    def __init__(self, function=None):
        self.function = function


class SetEnvironmentVariable:
    def __init__(self, name, value):
        self.name, self.value = name, value


class LogInfo:
    def __init__(self, msg=""):
        self.msg = msg


class ProcessExited:
    """What the real launch system gives to the handler when a process ends (we only need the exit code)."""
    def __init__(self, returncode=0):
        self.returncode = returncode


class OnProcessExit:
    """`on_exit` is a list of actions (like in the real class) or a function (event, context) -> actions."""
    def __init__(self, target_action=None, on_exit=None):
        self.target_action = target_action
        self.on_exit = on_exit if callable(on_exit) else list(on_exit or [])

    def fire(self, returncode=0):
        """Pretend that the target process ended with this exit code. Returns the actions that would start."""
        if callable(self.on_exit):
            return list(self.on_exit(ProcessExited(returncode), FakeContext()) or [])
        return list(self.on_exit)


class RegisterEventHandler:
    def __init__(self, event_handler):
        self.event_handler = event_handler


class Shutdown:
    """launch.events.Shutdown (keyword-only arguments like the real one)."""
    def __init__(self, *, reason="reason not given", due_to_sigint=False):
        self.reason, self.due_to_sigint = reason, due_to_sigint


class EmitEvent:
    def __init__(self, *, event, **kwargs):
        self.event = event


class IfCondition:
    def __init__(self, predicate):
        self.predicate = predicate


class Node:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.package = kwargs.get("package")
        self.executable = kwargs.get("executable")
        self.arguments = kwargs.get("arguments", [])
        self.parameters = kwargs.get("parameters", [])


class ParameterValue:
    def __init__(self, value, value_type=None):
        self.value, self.value_type = value, value_type


class FakeShare:
    """Maps package names to folders (set by the test)."""
    folders: Dict[str, str] = {}


def get_package_share_directory(name):
    if name not in FakeShare.folders:
        raise PackageNotFoundError(name)
    return FakeShare.folders[name]


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def build_modules():
    return {
        "launch": _module("launch", LaunchDescription=LaunchDescription, LaunchContext=FakeContext),
        "launch.actions": _module("launch.actions", DeclareLaunchArgument=DeclareLaunchArgument,
                                  EmitEvent=EmitEvent, IncludeLaunchDescription=IncludeLaunchDescription,
                                  LogInfo=LogInfo, OpaqueFunction=OpaqueFunction,
                                  RegisterEventHandler=RegisterEventHandler,
                                  SetEnvironmentVariable=SetEnvironmentVariable),
        "launch.conditions": _module("launch.conditions", IfCondition=IfCondition),
        "launch.event_handlers": _module("launch.event_handlers", OnProcessExit=OnProcessExit),
        "launch.events": _module("launch.events", Shutdown=Shutdown),
        "launch.launch_description_sources": _module("launch.launch_description_sources",
                                                     PythonLaunchDescriptionSource=PythonLaunchDescriptionSource),
        "launch.substitutions": _module("launch.substitutions", LaunchConfiguration=LaunchConfiguration),
        "launch_ros": _module("launch_ros"),
        "launch_ros.actions": _module("launch_ros.actions", Node=Node),
        "launch_ros.parameter_descriptions": _module("launch_ros.parameter_descriptions",
                                                     ParameterValue=ParameterValue),
        "ament_index_python": _module("ament_index_python"),
        "ament_index_python.packages": _module("ament_index_python.packages",
                                               get_package_share_directory=get_package_share_directory),
    }


@contextlib.contextmanager
def install(folders: Dict[str, str]):
    FakeShare.folders = dict(folders)
    fake = build_modules()
    saved = {k: sys.modules.get(k) for k in fake}
    sys.modules.update(fake)
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        FakeShare.folders = {}


def load_launch_file(path: str):
    """Execute a launch file (a Python file) and return its module."""
    spec = importlib.util.spec_from_file_location("launch_under_test_" + os.path.basename(path).split(".")[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten(entities) -> List:
    return list(entities)
