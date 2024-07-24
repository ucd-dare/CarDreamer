import carla
import math
import numpy as np
import random

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_vehicle_pos, get_location_distance, get_vehicle_velocity, get_vehicle_orientation

class CarlaFollowEnv(CarlaWptEnv):
    def on_reset(self) -> None:
        random_num = self._config.direction

        # print(random_num)
        self.nonego_spawn_point = self._config.nonego_spawn_points[random_num]
        # self.nonego_spawn_point = self._config.nonego_spawn_points
        nonego_transform = carla.Transform(carla.Location(*self.nonego_spawn_point[:3]), carla.Rotation(yaw = self.nonego_spawn_point[4]))
        self.nonego = self._world.spawn_actor(transform=nonego_transform)
        # print(self.nonego_spawn_point)
        self.ego_src = self._config.lane_start_points[random_num]
        # self.ego_src = self._config.lane_start_points
        ego_transform = carla.Transform(carla.Location(*self.ego_src[:3]), carla.Rotation(yaw = self.nonego_spawn_point[4]))
        self.ego = self._world.spawn_actor(transform=ego_transform)
        # print(get_vehicle_pos(self.ego))

        # Path planning
        # ego_dest = self._config.lane_end_points
        # dest_location = carla.Location(x=ego_dest[0], y=ego_dest[1], z=ego_dest[2])
        # self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        # self.waypoints, self.planner_stats = self.ego_planner.run_step()
        # self.num_completed = self.planner_stats['num_completed']
        self.on_step()
    
    def on_step(self) -> None:
        random_num = self._config.direction

        nonego_x, nonego_y = get_vehicle_pos(self.nonego)
        dest_location = carla.Location(x = nonego_x, y = nonego_y, z=self._config.lane_end_points[random_num][2])
        # print(dest_location)
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
        total_reward -= info['r_out_of_lane'] + info['r_speed'] + info['r_waypoints']
        del info['r_out_of_lane']
        del info['r_speed']
        del info['r_waypoints']
        # total_reward -= info['waypoint']
        # del info['waypoint']

        reward_scales = self._config.reward.scales
        # ego_x, ego_y = get_vehicle_pos(self.ego)
        nonego_loc = self.nonego.get_transform().location

        # if ego_x < nonego_loc.x + 0.1 and ego_x > nonego_loc.x - 0.1:
        #     p_stay_in_lane = 0.3 * reward_scales['stay_in_lane']
        # else:
        #     p_stay_in_lane = -reward_scales['stay_in_lane']

        ego_velocity = np.array([*get_vehicle_velocity(self.ego)])
        ego_speed = np.linalg.norm(ego_velocity)
        nonego_velocity = np.array([*get_vehicle_velocity(self.nonego)])
        nonego_speed = np.linalg. norm(nonego_velocity)
        if ego_speed > nonego_speed - 0.5 and ego_speed < nonego_speed + 0.5:
            p_speed = 0.2 * reward_scales['speed']
        else:
            p_speed = -reward_scales['speed']

        # original_dist = math.sqrt((self._config.lane_start_points[0] - self._config.nonego_spawn_points[0]) ** 2 + 
        #                           (self._config.lane_start_points[1] - self._config.nonego_spawn_points[1]) ** 2)
        # current_dist = math.sqrt((ego_x - nonego_loc.x) ** 2 + (ego_y - nonego_loc.y) ** 2)
        random_num = self._config.direction
        
        original_dist = get_location_distance((self._config.lane_start_points[random_num][0], self._config.lane_start_points[random_num][1]),
                                              (self._config.nonego_spawn_points[random_num][0], self._config.nonego_spawn_points[random_num][1]))
        current_dist = get_location_distance(get_vehicle_pos(self.ego), get_vehicle_pos(self.nonego))
        
        if current_dist < original_dist + 2 and current_dist >= original_dist:
            p_dist = 0.2 * reward_scales["distance"]
        else:
            p_dist = -reward_scales["distance"]
        
        ego_direction = get_vehicle_orientation(self.ego)
        nonego_direction = get_vehicle_orientation(self.nonego)
        if ego_direction < nonego_direction + 1.0 and ego_direction > nonego_direction - 1.0:
            p_stay_in_lane = 0.2 * reward_scales["stay_in_lane"]
        else:
            p_stay_in_lane = -reward_scales["stay_in_lane"]

        p_dest_reached = 0
        if get_vehicle_pos(self.nonego) == (self._config.lane_end_points[random_num][0], self._config.lane_end_points[random_num][1]):
            p_dest_reached = reward_scales["destination_reached"]

        total_reward += p_dist + p_speed + p_stay_in_lane + p_dest_reached
        return total_reward, info

    def get_terminal_conditions(self):
        terminal_config = self._config.terminal
        info = super().get_terminal_conditions()
        del info['destination_reached']

        ego_x, ego_y = get_vehicle_pos(self.ego)
        nonego_loc = self.nonego.get_transform().location
        dist = math.sqrt((ego_x - nonego_loc.x) ** 2 + (ego_y - nonego_loc.y) ** 2)
        # print(nonego_loc.x, nonego_loc.y)
        # print(ego_x, ego_y)
        # print(dist)
        if dist > terminal_config.terminal_dist:
            info['terminal_dist'] = True
            # print(nonego_loc.x, nonego_loc.y)
            # print(ego_x, ego_y)
            # print(dist)

        return info