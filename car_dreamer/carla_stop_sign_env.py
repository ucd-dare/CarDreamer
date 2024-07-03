import carla
import random
import time
import numpy as np
from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import FixedPathPlanner, get_vehicle_velocity

class CarlaStopSignEnv(CarlaWptFixedEnv):
    """
    Vehicle follows the stop sign when passing the intersection.

    **Provided Tasks**: ``carla_stop_sign``
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
        self.num_completed = self.planner_stats['num_completed']
        self._stop_time = 0
        self._has_stopped = False

        traffic_location = carla.Location(*self._config.traffic_locations)
        self.stop_sign = self.find_stop_sign_by_location(traffic_location)

    def reward(self):
        reward_scales = self._config.reward.scales
        total_reward, info = super().reward()

        p_traffic_light_violation = self.calculate_traffic_light_violation_penalty(reward_scales['traffic_light_violate'])

        # r_speed_before_stop = 0
        # # Enter stop sign range while have not stopped
        # if hasattr(self, '_first_enter_time') and self._stop_time == 0:
        #     r_speed_before_stop = 1 / (0.1 + np.abs(np.linalg.norm(np.array([*get_vehicle_velocity(self.ego)])))) * reward_scales['speed_before_stop']

        r_stop = 0  # Encourage vehicle stops and moves
        if 0 <= self._stop_time <= self._config.stopping_time:
            r_stop = reward_scales['stop'] * self._stop_time
        elif self._stop_time > self._config.stopping_time:
            r_stop = - reward_scales['stop'] * (self._stop_time - self._config.stopping_time)

        total_reward += p_traffic_light_violation + r_stop
        info['r_traffic_light_violation'] = p_traffic_light_violation
        info['r_stop'] = r_stop
        # info['r_speed_before_stop'] = r_speed_before_stop

        return total_reward, info
    
    def get_terminal_conditions(self):
        conds = super().get_terminal_conditions()
        conds['violate_stop_signs'] = self.is_violating_traffic_sign(self.ego)
        return conds

    def calculate_traffic_light_violation_penalty(self, violate_scale):
        if self.is_violating_traffic_sign(self.ego):
            return -violate_scale
        return 0.0
    
    def find_stop_sign_by_location(self, location, radius=10.0):
        stop_signs = self._world._get_world().get_actors().filter('traffic.stop')
        for stop_sign in stop_signs:
            if stop_sign.get_transform().location.distance(location) < radius:
                return stop_sign
        return None

    def is_violating_traffic_sign(self, vehicle):
        traffic_light = vehicle.get_traffic_light()
        traffic_state = vehicle.get_traffic_light_state()
        if traffic_light is None and traffic_state == carla.TrafficLightState.Red:    # Have stop sign nearby
            if not hasattr(self, '_first_enter_time'):
                self._first_enter_time = self._time_step

            time_in_range = self._time_step - self._first_enter_time
            if vehicle.get_velocity().length() < 0.1 and self._has_stopped is False:
                self._stop_time += 1
            elif vehicle.get_velocity().length() >= 0.1 and self._stop_time > 0:
                self._has_stopped = True
                self._stop_time = 0
            else:
                self._stop_time = 0

            # if self._stop_time > self._config.stopping_time:
            #     traffic_state = carla.TrafficLightState.Green

            # # print(f"time range: {time_in_range}, stop time:{self._stop_time}")
            # print(f"light state: {vehicle.get_traffic_light_state()}, stop time: {self._stop_time}, has stopped: {self._has_stopped}")
            
            if time_in_range < self._config.stop_start_time:
                return False
            elif self._stop_time == 0:
                return True
            elif 0 < self._stop_time <= self._config.stopping_time and vehicle.get_velocity().length() > 0.1:
                return True
            elif self._stop_time > self._config.stopping_time:
                return False

            return False


