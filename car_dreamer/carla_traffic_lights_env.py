import time

import carla
import numpy as np

from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import FixedPathPlanner


class CarlaTrafficLightsEnv(CarlaWptFixedEnv):
    """
    Vehicle follows the traffic lights when passing the intersection.

    **Provided Tasks**: ``carla_traffic_lights``
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

        traffic_location = carla.Location(*self._config.traffic_locations)
        self.traffic_light = self.find_traffic_light_by_location(traffic_location)
        self.traffic_light.set_state(carla.TrafficLightState.Red)
        self.red_duration = np.random.randint(self._config.red_duration[0], self._config.red_duration[1])
        self.green_duration = np.random.randint(self._config.green_duration[0], self._config.green_duration[1])
        self.yellow_duration = 10
        self.total_duration = self.red_duration + self.green_duration + self.yellow_duration

    def reward(self):
        reward_scales = self._config.reward.scales
        total_reward, info = super().reward()

        p_traffic_light_violation = self.calculate_traffic_light_violation_penalty(reward_scales["traffic_light_violate"])

        total_reward += p_traffic_light_violation
        info["r_traffic_light_violation"] = p_traffic_light_violation

        return total_reward, info

    def get_terminal_conditions(self):
        conds = super().get_terminal_conditions()
        conds["violate_traffic_lights"] = is_violating_traffic_light(self.ego)
        return conds

    def calculate_traffic_light_violation_penalty(self, violate_scale):
        if is_violating_traffic_light(self.ego):
            return -violate_scale
        return 0.0

    def find_traffic_light_by_location(self, location, radius=10.0):
        traffic_lights = self._world._get_world().get_actors().filter("traffic.traffic_light")
        for traffic_light in traffic_lights:
            if traffic_light.get_transform().location.distance(location) < radius:
                return traffic_light
        return None

    def step(self, action):
        result = super().step(action)
        if self._time_step % self.total_duration == self.red_duration:
            self.traffic_light.set_state(carla.TrafficLightState.Green)
        elif self._time_step % self.total_duration == self.total_duration - self.yellow_duration:
            self.traffic_light.set_state(carla.TrafficLightState.Yellow)
        elif self._time_step % self.total_duration == 0:
            self.traffic_light.set_state(carla.TrafficLightState.Red)

        return result


def is_violating_traffic_light(vehicle):
    traffic_light = vehicle.get_traffic_light()
    if traffic_light is not None:
        traffic_light_state = traffic_light.get_state()
        vehicle_velocity = vehicle.get_velocity()
        speed = vehicle_velocity.length()
        if traffic_light_state == carla.TrafficLightState.Red and speed > 0.1:
            return True
    return False
