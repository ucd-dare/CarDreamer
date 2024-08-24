from collections import deque

import carla
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedPathPlanner, get_location_distance, get_vehicle_pos


class CarlaWptFixedEnv(CarlaWptEnv):
    """
    This is the base env for all waypoint following tasks with a fixed route and car flow.
    **DO NOT** instantiate this class directly.

    All envs that inherit from this class also inherits the following config parameters:

    * ``lane_start_point``: The starting point of the ego vehicle in ``[x, y, z, yaw]``
    * ``ego_path``: The fixed path for the ego vehicle in array of ``[x, y, z]``
    * ``use_road_waypoints``: For each segment, whether to adapt the path according to road or use straight line
    * ``flow_spawn_point``: The spawn point of the car flow in ``[x, y, z, yaw]``
    * ``min_flow_dist``: Minimum distance between two cars in the flow, if ``None``, no cars will be spawned
    * ``max_flow_dist``: Maximum distance between two cars in the flow

    """

    def on_reset(self) -> None:
        self.ego_src = self._config.lane_start_point
        ego_transform = carla.Transform(carla.Location(*self.ego_src[:3]), carla.Rotation(yaw=self.ego_src[3]))
        self.ego = self._world.spawn_actor(transform=ego_transform)
        self.ego_path = self._config.ego_path
        self.use_road_waypoints = self._config.use_road_waypoints
        self.ego_planner = FixedPathPlanner(
            vehicle=self.ego,
            vehicle_path=self.ego_path,
            use_road_waypoints=self.use_road_waypoints,
        )
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

        # Initialize car flow
        self.actor_flow = deque()
        flow_spawn_point = self._config.flow_spawn_point
        self.flow_transform = carla.Transform(
            carla.Location(*flow_spawn_point[:3]),
            carla.Rotation(yaw=flow_spawn_point[3]),
        )

    def on_step(self) -> None:
        super().on_step()

        # Generate and sink car flow
        spawn = False
        if "min_flow_dist" in self._config:
            if len(self.actor_flow) == 0:
                spawn = True
            else:
                spawn_location = np.array(self._config.flow_spawn_point[:2])
                nearest_car_location = np.array(get_vehicle_pos(self.actor_flow[-1]))
                flow_dist = np.random.uniform(self._config.min_flow_dist, self._config.max_flow_dist)
                if get_location_distance(spawn_location, nearest_car_location) >= flow_dist:
                    spawn = True
        if spawn:
            vehicle = self._world.try_spawn_aggresive_actor(self.flow_transform)
            if vehicle is not None:
                self.actor_flow.append(vehicle)
