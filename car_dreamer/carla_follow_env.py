import carla
import math
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_vehicle_pos

class CarlaFollowEnv(CarlaWptEnv):
    def on_reset(self) -> None:
        self.nonego_spawn_point = self._config.nonego_spawn_points
        # print(self.nonego_spawn_point[0])
        nonego_transform = carla.Transform(carla.Location(*self.nonego_spawn_point[:3]), carla.Rotation(*self.nonego_spawn_point[-3:]))
        self.nonego = self._world.spawn_actor(transform=nonego_transform)
        self.ego_src = self._config.lane_start_points
        ego_transform = carla.Transform(carla.Location(x=self.ego_src[0], y=self.ego_src[1], z=self.ego_src[2]), carla.Rotation(yaw = 90.0))
        self.ego = self._world.spawn_actor(transform=ego_transform)

        # Path planning
        ego_dest = self._config.lane_end_points
        dest_location = carla.Location(x=ego_dest[0], y=ego_dest[1], z=ego_dest[2])
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats['num_completed']
        super().on_step()
    
    def apply_control(self, action) -> None:
        control = self.get_vehicle_control(action)
        nonego_control = self.get_nonego_vehicle_control()
        self.ego.apply_control(control)
        self.nonego.apply_control(nonego_control)
    
    def get_nonego_vehicle_control(self):
        '''
        Non-ego vehicle control is designed for the scenario.
        '''
        ego_loc = self.ego.get_transform().location
        nonego_loc = self.nonego.get_transform().location

        # Keep constant speed
        if abs(self.nonego.get_velocity().y) < 2:
            acc = 2
        else:
            acc = 0

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc/3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc/3, 0, 1)

        return carla.VehicleControl(throttle=float(throttle), brake=float(brake))

    
    def reward(self):
        total_reward, info = super().reward()
        total_reward -= info['r_out_of_lane']
        del info['r_out_of_lane']
        # total_reward -= info['waypoint']
        # del info['waypoint']

        reward_scales = self._config.reward.scales
        ego_x, ego_y = get_vehicle_pos(self.ego)
        nonego_loc = self.nonego.get_transform().location
        dist = math.sqrt((ego_x - nonego_loc.x) ** 2 + (ego_y - nonego_loc.y) ** 2)

        if ego_x < nonego_loc.x + 0.1 and ego_x > nonego_loc.x - 0.1:
            p_stay_in_lane = reward_scales['stay_in_lane']
        else:
            p_stay_in_lane = -1.5 * reward_scales['stay_in_lane']

        if dist < 5 and nonego_loc.y > ego_y + 12:
            p_dist = reward_scales["distance"]
        else:
            p_dist = -1.5 * reward_scales["distance"]

        total_reward += p_dist + p_stay_in_lane
        return total_reward, info

    def get_terminal_conditions(self):
        terminal_config = self._config.terminal
        info = super().get_terminal_conditions()

        ego_x, ego_y = get_vehicle_pos(self.ego)
        nonego_loc = self.nonego.get_transform().location
        dist = math.sqrt((ego_x - nonego_loc.x) ** 2 + (ego_y - nonego_loc.y) ** 2)
        if dist > terminal_config.terminal_dist:
            info['terminal_dist'] = True

        return info