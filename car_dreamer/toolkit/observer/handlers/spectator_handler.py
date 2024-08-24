from typing import Dict, Tuple

import carla
import numpy as np
from gym import spaces

from ...carla_manager import WorldManager
from .base_handler import BaseHandler


class SpectatorHandler(BaseHandler):
    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        self._spectator_camera = None
        self._ego = None
        self._data = None

    def get_observation_space(self) -> Dict:
        return {self._config.key: spaces.Box(low=0, high=255, shape=self._config.shape, dtype=np.uint8)}

    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        if self._spectator_camera is None or self._ego is None or self._data is None:
            return {self._config.key: np.zeros(self._config.shape, dtype=np.uint8)}, {}

        # Update spectator location to follow the ego vehicle
        ego_transform = self._ego.get_transform()
        spectator_transform = carla.Transform(
            ego_transform.location + carla.Location(z=self._config.height),
            carla.Rotation(pitch=self._config.pitch),
        )
        spectator = self._world._world.get_spectator()
        spectator.set_transform(spectator_transform)
        self._spectator_camera.set_transform(spectator_transform)

        obs = {self._config.key: self._data}
        info = {}

        return obs, info

    def destroy(self) -> None:
        if self._spectator_camera is not None:
            self._spectator_camera.destroy()
            self._spectator_camera = None

    def reset(self, ego: carla.Actor) -> None:
        self._ego = ego

        if self._spectator_camera is None:
            spectator_bp = self._world.get_blueprint("sensor.camera.rgb")
            spectator_bp.set_attribute("image_size_x", str(self._config.shape[1]))
            spectator_bp.set_attribute("image_size_y", str(self._config.shape[0]))
            spectator_bp.set_attribute("sensor_tick", str(self._config.sensor_tick))
            spectator_bp.set_attribute("fov", str(self._config.fov))
            self._spectator_camera = self._world.spawn_unmanaged_actor(carla.Transform(), spectator_bp)
            self._spectator_camera.listen(self._update_data)

    def _update_data(self, data) -> None:
        camera_data = np.frombuffer(data.raw_data, dtype=np.uint8)
        camera_data = np.reshape(camera_data, (data.height, data.width, 4))
        camera_data = camera_data[:, :, :3]
        camera_data = camera_data[:, :, ::-1]
        self._data = camera_data
