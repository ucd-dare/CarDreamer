import time

import carla
import numpy as np

from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import FixedPathPlanner, get_vehicle_pos, get_vehicle_velocity


class CarlaStopSignEnv(CarlaWptFixedEnv):
    """
    Vehicle follows the stop sign when passing the intersection.

    **Provided Tasks**: ``carla_stop_sign``
    """

    def __init__(self, config):
        super().__init__(config)

    def on_reset(self) -> None:
        np.random.seed(time.time())
        random_index = np.random.randint(0, len(self._config.lane_start_point) - 1)
        self.ego_src = self._config.lane_start_point[random_index]
        ego_transform = carla.Transform(carla.Location(*self.ego_src[:3]), carla.Rotation(yaw=self.ego_src[3]))
        self.ego = self._world.spawn_actor(transform=ego_transform)
        self.ego_path = self._config.ego_path[random_index]
        self.use_road_waypoints = self._config.use_road_waypoints
        self.ego_planner = FixedPathPlanner(
            vehicle=self.ego,
            vehicle_path=self.ego_path,
            use_road_waypoints=self.use_road_waypoints,
        )
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]
        self._stop_time = 0
        self._entered = 0  # 0 default, 1 enter, 2 leave
        self._stop_sign_state = {}

    def get_state(self):
        return {
            "ego_waypoints": self.waypoints,
            "timesteps": self._time_step,
            "stop_sign_state": self._stop_sign_state,
        }

    def reward(self):
        reward_scales = self._config.reward.scales
        total_reward, info = super().reward()

        p_violate_stop_sign = 0
        p_violate_stop_sign = self.calculate_traffic_light_violation_penalty() * reward_scales["stop_sign"]

        total_reward += p_violate_stop_sign
        info["r_stop"] = p_violate_stop_sign

        return total_reward, info

    def on_step(self) -> None:
        self.handle_stop_sign()
        return super().on_step()

    def get_terminal_conditions(self):
        conds = super().get_terminal_conditions()
        conds["violate_stop_sign"] = self.violate_traffic_light()
        return conds

    def calculate_traffic_light_violation_penalty(self):
        if self.violate_traffic_light():
            return -1.0
        else:
            return 0.0

    def violate_traffic_light(self):
        if self.is_within_stop_sign_proximity(self._config.traffic_locations):
            self._stop_time += 1
            self._entered = 1
        elif self._entered == 1:  # Mark the leaving
            self._entered = 2

        if self._entered == 2 and self._stop_time < self._config.stopping_time:
            return True
        return False

    def is_within_stop_sign_proximity(self, sign_location):
        """
        Check if the ego vehicle is near the stop sign.
        """
        ego_location = np.array([*get_vehicle_pos(self.ego), 0.1])
        distance = np.linalg.norm(ego_location - sign_location)

        return distance <= self._config.stop_sign_proximity_threshold

    def _is_ego_near_stop_sign(self, stop_sign: carla.Actor) -> bool:
        """Check if the ego vehicle is within the proximity threshold of the stop sign."""
        ego_location = self.ego.get_location()
        stop_sign_location = stop_sign.get_location()
        distance = ego_location.distance(stop_sign_location)
        return distance < self._config.stop_sign_proximity_threshold

    def handle_stop_sign(self):
        stop_signs = self._world._get_world().get_actors().filter("traffic.stop")
        for stop_sign in stop_signs:
            if stop_sign.id not in self._stop_sign_state:
                if self._is_ego_near_stop_sign(stop_sign):
                    self._stop_sign_state[stop_sign.id] = {
                        "first_seen": self._time_step,
                        "turned_red": True,
                    }
                    color = 0  # Initially turn red
                    self._stop_sign_state[stop_sign.id]["color"] = color
            else:
                stop_sign_info = self._stop_sign_state[stop_sign.id]
                if self._time_step - stop_sign_info["first_seen"] < self._config.stopping_time:
                    color = 0  # Remain red during the countdown
                else:
                    color = 1  # Turn green after the countdown
                self._stop_sign_state[stop_sign.id]["color"] = color
