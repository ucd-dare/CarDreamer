import carla
import random
import time
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
        self._first_stop = True

        # self.traffic_location = carla.Location(*self._config.traffic_locations)
        # self.stop_sign = self.find_stop_sign_by_location(traffic_location)

    def reward(self):
        reward_scales = self._config.reward.scales
        ego_velocity = np.array([*get_vehicle_velocity(self.ego)])
        total_reward, info = super().reward()

        p_speed_before_stop = 0
        # Enter stop sign range while have not stopped
        if self._stop_time == 0:
            p_speed_before_stop = np.abs(np.linalg.norm(np.array([*get_vehicle_velocity(self.ego)]))) * reward_scales['speed_before_stop']

        r_stop = 0  # Encourage vehicle stops and moves
        if 0 <= self._stop_time <= self._config.stopping_time and np.linalg.norm(ego_velocity) < 0.1:
            r_stop = reward_scales['stop'] * self._stop_time
        elif self._stop_time > self._config.stopping_time and np.linalg.norm(ego_velocity) < 0.1:
            r_stop = - reward_scales['stop'] * (self._stop_time - self._config.stopping_time)

        total_reward += r_stop + p_speed_before_stop
        info['r_stop'] = r_stop
        info['p_speed_before_stop'] = p_speed_before_stop

        return total_reward, info
    
    def on_step(self) -> None:
        super().on_step()
        self.update_stop_time(np.array(self._config.traffic_locations))
    
    def get_terminal_conditions(self):
        conds = super().get_terminal_conditions()
        conds['violate_stop_signs'] = (self._stop_time < self._config.stopping_time and self._first_stop is False) or (self._stop_time == 0 and hasattr(self, '_entered') and getattr(self, '_entered') is False)
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
        
    def is_near_stop_sign(self, sign_location, threshold=10.0):
        """
        Check if the ego vehicle is near the stop sign.
        """
        ego_location = np.array([*get_vehicle_pos(self.ego), 0.1])
        distance = np.linalg.norm(ego_location - sign_location)
        if distance <= threshold and not hasattr(self, '_entered'): # First time enter
            self._entered = True
        if distance > threshold and hasattr(self, '_entered'):  # First time left
            self._entered = False
        return distance <= threshold
    
    def update_stop_time(self, sign_location):
        ego_velocity = np.array([*get_vehicle_velocity(self.ego)])

        if not hasattr(self, '_previous_stop_time'):
            self._previous_stop_time = None

        if not hasattr(self, '_stop_time'):
            self._stop_time = 0

        if not hasattr(self, '_first_stop'):
            self._first_stop = True
        
        if self.is_near_stop_sign(sign_location) and np.linalg.norm(ego_velocity) < 0.1:
            if self._first_stop:
                self._stop_time += 1
        
        if self._stop_time > 0 and np.linalg.norm(ego_velocity) >= 0.1:
            self._first_stop = False
        
        print(f"stopping time:{self._stop_time}, first stop:{self._first_stop}")


