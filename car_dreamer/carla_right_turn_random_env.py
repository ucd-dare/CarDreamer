import random
import time
from collections import deque

import carla

from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import FixedPathPlanner, get_vehicle_pos


class CarlaRightTurnRandomEnv(CarlaWptFixedEnv):
    """
    Vehicle passes the crossing (random turn right) and avoid collision.

    **Provided Tasks**: ``carla_right_turn_random``
    """

    def __init__(self, config):
        super().__init__(config)

    def on_reset(self) -> None:
        random.seed(time.time())
        random_index = random.randint(0, len(self._config.lane_start_point) - 1)
        self.ego_src = self._config.lane_start_point[random_index]
        ego_transform = carla.Transform(carla.Location(*self.ego_src[:3]), carla.Rotation(yaw=self.ego_src[3]))
        self.ego = self._world.spawn_actor(transform=ego_transform)
        self.ego_path = self._config.ego_path[random_index]
        self.use_road_waypoints = self._config.use_road_waypoints
        self.ego_planner = FixedPathPlanner(vehicle=self.ego, vehicle_path=self.ego_path, use_road_waypoints=self.use_road_waypoints)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]
