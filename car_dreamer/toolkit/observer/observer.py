from typing import Callable, Dict, Tuple

import carla
from gym import spaces

from ..carla_manager import WorldManager
from .handlers import BaseHandler, SimpleHandler
from .utils import HANDLER_DICT, HandlerType

SIMPLE_HANDLER_NAME = "simple"


class Observer:
    """
    An observer is a collection of data handlers, each providing some observation data.

    The output of the Observer can be configured through ``env.observation.enabled``, which is passed to the constructor in ``obs_config``.

    In addtion, :py:meth:`register_simple_handler` provides a flexible way to supplement the observation data with a callback.
    """

    def __init__(self, world: WorldManager, obs_config: dict):
        self._world = world
        self._obs_config = obs_config
        self._data_handlers = self._init_data_handlers()

    def register_simple_handler(
        self,
        key: str,
        observation_fn: Callable[[], Dict],
        observation_space: spaces.Space,
    ) -> None:
        """
        Register a simple observation function that may optionally use environment state.

        :param key: str, the key in observation space
        :param observation_fn: Callable[[], Dict], the callback function which returns the observation data
        :param observation_space: spaces.Space, the observation space
        """
        if SIMPLE_HANDLER_NAME not in self._data_handlers:
            self._data_handlers[SIMPLE_HANDLER_NAME] = SimpleHandler(self._world, {})
        self._data_handlers[SIMPLE_HANDLER_NAME].register_observation(key, observation_fn, observation_space)

    def destroy(self) -> None:
        """Destroy all the registered handlers."""
        for handler in self._data_handlers.values():
            handler.destroy()

    def reset(self, ego: carla.Actor) -> None:
        """Reset all the registered handlers with the given ego vehicle."""
        for handler in self._data_handlers.values():
            handler.reset(ego)

    def _init_data_handlers(self) -> Dict[str, BaseHandler]:
        """Initialize the EndpointHandlers based on the observation configuration."""
        handlers = {}
        for name in self._obs_config.enabled:
            config = self._obs_config[name]
            handler_class = HANDLER_DICT.get(HandlerType(config.handler))
            handler = handler_class(self._world, config)
            handlers[name] = handler
        return handlers

    def get_observation_space(self) -> spaces.Space:
        """Get the combined observation space from all the registered handlers."""
        obs_spaces = {}
        for handler in self._data_handlers.values():
            obs_spaces.update(handler.get_observation_space())
        return spaces.Dict(obs_spaces)

    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        """Get the current observation data from all the registered handlers."""
        obs = {}
        info = {}
        for handler in self._data_handlers.values():
            obs_data, info_data = handler.get_observation(env_state)
            obs.update(obs_data)
            info.update(info_data)
        return obs, info
