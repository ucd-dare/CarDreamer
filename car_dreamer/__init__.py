"""
Package ``car_dreamer`` is a collection of tasks aimed at facilitating RL algorithm
designing, especially world model based ones. Each task is a certain driving
environment in Carla simulator, varying from a single skill such as lane following or left
turning, to random roaming in mixed road conditions which may encounter crossroads,
roundabouts, etc. They expose the same gym interface for backbone RL algorithm use.

Furthermore, ``car_dreamer`` includes a task development suite for those who want to customize
their own tasks. It provides a number of API calls to minimize users' efforts in spawning the
vehicles, planning the routes, and obtaining observation data as RL algorithm inputs. And an
integrated traning visualization server automatically grasps the observation data, displaying the
videos and plotting the statistics through an HTTP server. This eases reward engineering, algorithm designing and
hyper-parameter tuning.
"""

from .carla_base_env import CarlaBaseEnv
from .carla_roundabout_env import CarlaRoundaboutEnv
from .carla_right_turn_env import CarlaRightTurnEnv
from .carla_overtake_env import CarlaOvertakeEnv
from .carla_navigation_env import CarlaNavigationEnv
from .carla_left_turn_env import CarlaLeftTurnEnv
from .carla_lane_merge_env import CarlaLaneMergeEnv
from .carla_four_lane_env import CarlaFourLaneEnv
from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .carla_wpt_env import CarlaWptEnv
__version__ = '0.1.0'

from . import toolkit


def load_task_configs(task_name: str):
    """
    Load the task configs for the specified task name.
    The task name should be one of the keys in the ``tasks.yaml`` file.

    :param task_name: str, the name of the task

    :return: the task configs
    """
    import yaml
    import os
    dir = os.path.dirname(__file__) + '/configs/'
    with open(dir + 'common.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config = toolkit.Config(config)
    with open(dir + 'tasks.yaml', 'r') as f:
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
    from gym.envs.registration import register
    from re import sub

    def toClassName(s):
        return sub(r"(_|-)+", " ", s).title().replace(" ", "")
    for file in os.listdir(os.path.dirname(__file__)):
        if file.endswith('env.py') and file != '__init__.py':
            file_name = file[:-3]
            class_name = toClassName(file_name)
            exec(f"register(id='{class_name}-v0', entry_point='car_dreamer.{file_name}:{class_name}')")


_register_envs()
