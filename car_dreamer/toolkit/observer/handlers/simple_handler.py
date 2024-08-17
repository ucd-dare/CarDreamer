from typing import Callable, Dict, Tuple

import carla
from gym import spaces

from ...carla_manager import WorldManager
from .base_handler import BaseHandler


class SimpleHandler(BaseHandler):
    """
    A simplified handler for registering and handling simple observations.
    """

    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        self._observation_functions: Dict[str, Callable[..., Dict]] = {}
        self._observation_spaces: Dict[str, spaces.Space] = {}

    def register_observation(
        self,
        name: str,
        observation_fn: Callable[[], Dict],
        observation_space: spaces.Space,
    ):
        """
        Register a simple observation function and its corresponding observation space.
        """
        self._observation_functions[name] = observation_fn
        self._observation_spaces[name] = observation_space

    def get_observation_space(self) -> Dict:
        return self._observation_spaces

    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        obs = {name: obs_fn(env_state) for name, obs_fn in self._observation_functions.items()}
        return obs, {}

    def destroy(self) -> None:
        pass

    def reset(self, ego: carla.Actor) -> None:
        pass
