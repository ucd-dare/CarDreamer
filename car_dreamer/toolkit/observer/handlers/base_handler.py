from abc import ABC, abstractmethod
from typing import Dict, Tuple

import carla

from ...carla_manager import WorldManager


class BaseHandler(ABC):
    """
    Base class for data endpoint handlers.
    """

    def __init__(self, world: WorldManager, config):
        self._world = world
        self._config = config

    @abstractmethod
    def get_observation_space(self) -> Dict:
        """Get the observation space for this data endpoint."""
        pass

    @abstractmethod
    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        """
        Fetch observations from the environment.

        :param env_state: The state returned by :py:meth:`carla_manager.CarlaBaseEnv.get_state`.
        """
        pass

    @abstractmethod
    def destroy(self) -> None:
        """Clean up resources"""
        pass

    @abstractmethod
    def reset(self, ego: carla.Actor) -> None:
        """Reset the handler"""
        pass
