from typing import Dict, Tuple

import carla
import cv2
import numpy as np
from gym import spaces

from ...carla_manager import WorldManager
from .base_handler import BaseHandler
from .renderer.birdeye_renderer import BirdeyeRenderer
from .renderer.constants import BirdeyeEntity, Color
from .utils import Observability, WaypointObservability, get_neighbors, get_visibility


class BirdeyeHandler(BaseHandler):
    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)

        self._display_size = max(512, config.shape[0])
        pixels_per_meter = self._display_size / config.obs_range
        pixels_ahead_vehicle = (config.obs_range / 2 - config.ego_offset) * pixels_per_meter

        self._birdeye_render = BirdeyeRenderer(
            world,
            pixels_per_meter,
            self._display_size,
            pixels_ahead_vehicle,
            config.camera_fov,
        )
        self.surface = np.zeros((self._display_size, self._display_size, 3), dtype=np.uint8)

    def get_observation_space(self) -> Dict:
        return {self._config.key: spaces.Box(low=0, high=255, shape=self._config.shape, dtype=np.uint8)}

    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        # Append actors polygon list
        entities = [BirdeyeEntity(c) for c in self._config.entities]

        # Birdeye view
        actor_transforms = self._world.actor_transforms
        actor_polygons = self._world.actor_polygons
        is_fov_visible, is_recursive_visible = get_visibility(self._ego, actor_transforms, actor_polygons, self._config.camera_fov)
        neighbors = get_neighbors(self._ego, actor_transforms, is_fov_visible)
        observability = Observability(self._config.observability)
        if observability == Observability.FOV:
            visible = is_fov_visible
        elif observability == Observability.RECURSIVE_FOV:
            visible = {id: is_fov_visible[id] or is_recursive_visible[id] for id in self._world.actor_ids}
        else:
            visible = {id: True for id in self._world.actor_ids}

        background_vehicles_color = self._get_background_vehicles_color(self._world.actor_ids, visible, is_fov_visible, is_recursive_visible)
        background_waypoints_color = self._get_background_waypoints_color(self._world.actor_ids, visible, neighbors)
        messages_color = self._get_messages_color(self._world.actor_ids, neighbors)

        env_state = {
            **env_state,
            "background_vehicles_color": background_vehicles_color,
            "background_waypoints_color": background_waypoints_color,
            "messages_color": messages_color,
            "extend_waypoints": self._config.extend_wpt,
        }
        if hasattr(self._config, "error_rate"):
            env_state["error_rate"] = self._config.error_rate
        self._birdeye_render.render(self.surface, entities, env_state)
        birdeye = self.surface

        # Reshape the birdeye display to the configured size
        birdeye_resized = cv2.resize(
            birdeye,
            (self._config.shape[0], self._config.shape[1]),
            interpolation=cv2.INTER_AREA,
        )

        obs = {self._config.key: birdeye_resized.astype(np.uint8)}
        info = {}

        return obs, info

    def destroy(self) -> None:
        pass

    def reset(self, ego: carla.Actor) -> None:
        self._ego = ego
        self._birdeye_render.set_ego(ego)

    def _get_background_vehicles_color(self, actor_ids, visible, is_fov_visible, is_recursive_visible):
        if self._config.color_by_obs:
            background_vehicles_color = {}
            for id in actor_ids:
                if visible[id]:
                    if is_fov_visible[id]:
                        background_vehicles_color[id] = Color.GREEN
                    elif is_recursive_visible[id]:
                        background_vehicles_color[id] = Color.BUTTER_0
                    else:
                        background_vehicles_color[id] = Color.ALUMINIUM_0
                else:
                    background_vehicles_color[id] = None
        else:
            background_vehicles_color = {id: Color.GREEN if visible[id] else None for id in actor_ids}
        return background_vehicles_color

    def _get_background_waypoints_color(self, actor_ids, visible, neighbors):
        waypoint_obs = WaypointObservability(self._config.waypoint_obs)
        if waypoint_obs == WaypointObservability.ALL:
            background_waypoints_color = {id: Color.ORANGE_0 for id in actor_ids}
        elif waypoint_obs == WaypointObservability.VISIBLE:
            background_waypoints_color = {id: Color.ORANGE_0 if visible[id] else None for id in actor_ids}
        else:
            background_waypoints_color = {id: Color.ORANGE_0 if id in neighbors else None for id in actor_ids}
        return background_waypoints_color

    def _get_messages_color(self, actor_ids, neighbors):
        messages_color = {id: Color.WHITE if id in neighbors else None for id in actor_ids}
        return messages_color
