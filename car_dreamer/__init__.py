"""
`CarDreamer` is a platform designed for world model based autonomous driving,
featuring a set of urban driving tasks within the realistic CARLA simulator.
Tasks range from basic maneuvers, such as lane following, to complex navigation in varied road conditions.
All tasks are integrated with OpenAI Gym interfaces, enabling straightforward evaluation of algorithms.
The platform includes decoupled data handlers and an observer to manage multi-modal observations,
allowing users to easily customize modality and observability.
The Development Suite aims at facilitating creation of new urban driving tasks.
"""

from .carla_base_env import CarlaBaseEnv
from .carla_four_lane_env import CarlaFourLaneEnv
from .carla_lane_merge_env import CarlaLaneMergeEnv
from .carla_left_turn_env import CarlaLeftTurnEnv
from .carla_navigation_env import CarlaNavigationEnv
from .carla_overtake_env import CarlaOvertakeEnv
from .carla_right_turn_env import CarlaRightTurnEnv
from .carla_right_turn_random_env import CarlaRightTurnRandomEnv
from .carla_roundabout_env import CarlaRoundaboutEnv
from .carla_stop_sign_env import CarlaStopSignEnv
from .carla_traffic_lights_env import CarlaTrafficLightsEnv
from .carla_wpt_env import CarlaWptEnv
from .carla_wpt_fixed_env import CarlaWptFixedEnv

__version__ = "0.2.0"

from . import toolkit


def load_task_configs(task_name: str):
    """
    Load the task configs for the specified task name.
    The task name should be one of the keys in the ``tasks.yaml`` file.

    :param task_name: str, the name of the task

    :return: the task configs
    """
    import os

    import yaml

    dir = os.path.join(os.path.dirname(__file__), "configs")
    with open(os.path.join(dir, "common.yaml")) as f:
        config = yaml.safe_load(f)
        config = toolkit.Config(config)
    with open(os.path.join(dir, "tasks.yaml")) as f:
        task_config = yaml.safe_load(f)
        config = config.update(task_config[task_name])
    return config


def create_task(task_name: str, argv=None):
    """
    Create a driving task with the specified task name.
    The task name should be one of the keys in the ``tasks.yaml`` file.

    :param task_name: str, the name of the task
    :param argv: list, the command line arguments, unrecognized arguments will be omitted

    :return: a tuple of the created environment and the configs
    """
    import gym

    config = load_task_configs(task_name)
    config, _ = toolkit.Flags(config).parse_known(argv)
    return gym.make(config.env.name, config=config.env), config


def _register_envs():
    import os
    from re import sub

    from gym.envs.registration import register

    def toClassName(s):
        return sub(r"(_|-)+", " ", s).title().replace(" ", "")

    for file in os.listdir(os.path.dirname(__file__)):
        if file.endswith("env.py") and file != "__init__.py":
            file_name = file[:-3]
            class_name = toClassName(file_name)
            exec(f"register(id='{class_name}-v0', entry_point='car_dreamer.{file_name}:{class_name}')")


_register_envs()
