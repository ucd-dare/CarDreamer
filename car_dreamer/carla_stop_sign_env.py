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

    def reward(self):
        reward_scales = self._config.reward.scales
        total_reward, info = super().reward()

        p_violate_stop_sign = 0
        p_violate_stop_sign = self.calculate_traffic_light_violation_penalty() * reward_scales['stop_sign']

        total_reward += p_violate_stop_sign
        info['r_stop'] = p_violate_stop_sign

        return total_reward, info
    
    def get_terminal_conditions(self):
        conds = super().get_terminal_conditions()
        conds['violate_stop_sign'] = self.violate_traffic_light()
        return conds
    
    def calculate_traffic_light_violation_penalty(self):
        if self.violate_traffic_light():
            return -1.0
        else:
            return 0.0
    
    def violate_traffic_light(self):
        if self.is_near_stop_sign(self._config.traffic_locations):
            self._stop_time += 1
            self._entered = True
        elif hasattr(self, '_entered') and self._entered is True: # Mark the leaving
            self._entered = False
        
        if hasattr(self, '_entered') and self._entered is False and self._stop_time < self._config.stopping_time:
            return True
        return False
        
    def is_near_stop_sign(self, sign_location, threshold=3.0):
        """
        Check if the ego vehicle is near the stop sign.
        """
        ego_location = np.array([*get_vehicle_pos(self.ego), 0.1])
        distance = np.linalg.norm(ego_location - sign_location)

        return distance <= threshold
    
