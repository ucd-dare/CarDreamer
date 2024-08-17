from typing import Dict, Tuple

import carla
import numpy as np
from gym import spaces

from ...carla_manager import Command, WorldManager
from .base_handler import BaseHandler
from .utils import get_neighbors, get_visibility


class MessageHandler(BaseHandler):
    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        self._world = world
        self._command_to_label = {
            Command.LaneFollow: 0,
            Command.LaneChangeLeft: 1,
            Command.LaneChangeRight: 2,
        }

    def get_observation_space(self) -> Dict:
        return {
            "message": spaces.Box(
                low=0,
                high=1,
                shape=(self._config.neighbor_num * self._config.n_commands,),
                dtype=np.float32,
            ),
            "dest": spaces.Box(low=0, high=1, shape=(self._config.dest_num,), dtype=np.float32),
        }

    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        dest = np.zeros((self._config.dest_num,), dtype=np.float32)
        dest[env_state["dest_lane_idx"]] = 1

        is_fov_visible, _ = get_visibility(
            self._ego,
            self._world.actor_transforms,
            self._world.actor_polygons,
            self._config.camera_fov,
        )
        neighbors = get_neighbors(self._ego, self._world.actor_transforms, is_fov_visible)
        actor_actions = self._world.actor_actions
        text = np.zeros((self._config.neighbor_num, self._config.n_commands), dtype=np.float32)
        for i, neighbor in enumerate(neighbors):
            if neighbor is not None and neighbor in actor_actions:
                actions = actor_actions[neighbor]
                if len(actions) == 0:
                    continue
                label = actions[0][0]
                label = self._command_to_label.get(label, 0)
                text[i][label] = 1
        text = text.flatten()
        return {"message": text, "dest": dest}, {}

    def destroy(self) -> None:
        pass

    def reset(self, ego: carla.Actor) -> None:
        self._ego = ego
