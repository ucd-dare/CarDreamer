import carla
import math
import numpy as np
import random

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_vehicle_pos, get_location_distance, get_vehicle_velocity, get_vehicle_orientation

class CarlaFollowEnv(CarlaWptEnv):
    def on_reset(self) -> None:
        if self._config.direction == 4:
            self.random_num = random.randint(0, len(self._config.lane_start_point) - 1)
        elif self._config.direction >= 0 and self._config.direction < 4:
            self.random_num = self._config.direction

        # print(random_num)
        self.nonego_spawn_point = self._config.nonego_spawn_points[self.random_num]
        # self.nonego_spawn_point = self._config.nonego_spawn_points
        nonego_transform = carla.Transform(carla.Location(*self.nonego_spawn_point[:3]), carla.Rotation(yaw = self.nonego_spawn_point[4]))
        self.nonego = self._world.spawn_actor(transform=nonego_transform)
        # print(self.nonego_spawn_point)
        self.ego_src = self._config.lane_start_points[self.random_num]
        # self.ego_src = self._config.lane_start_points
        ego_transform = carla.Transform(carla.Location(*self.ego_src[:3]), carla.Rotation(yaw = self.nonego_spawn_point[4]))
        self.ego = self._world.spawn_actor(transform=ego_transform)
        # print(get_vehicle_pos(self.ego))

        self.prev_errors = {'last_error': 0.0, 'integral': 0.0}
        self.nonego_direction = self.nonego_spawn_point[4]
        self.list_waypoints = [0]
        self.list_velocity = [0]

        # Path planning
        nonego_dest = self._config.lane_end_points[self.random_num]
        dest_location = carla.Location(x=nonego_dest[0], y=nonego_dest[1], z=nonego_dest[2])
        self.nonego_planner = FixedEndingPlanner(self.nonego, dest_location)
        self.on_step()
    
    def on_step(self) -> None:
        nonego_x, nonego_y = get_vehicle_pos(self.nonego)
        dest_location = carla.Location(x = nonego_x, y = nonego_y, z=self._config.lane_end_points[self.random_num][2])
        # print(dest_location)
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        # self.num_completed = self.planner_stats['num_completed']

        # nonego_velocity = np.array([*get_vehicle_velocity(self.nonego)])
        # nonego_speed = np.linalg. norm(nonego_velocity)

        self.nonego_waypoints, self.nonego_planner_stats = self.nonego_planner.run_step()
        # self.nonego_num_completed = self.nonego_planner_stats['num_completed']       

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
        nonego_loc = get_vehicle_pos(self.nonego)

        # Keep constant speed
        nonego_velocity = self.nonego.get_velocity()
        nonego_speed = math.sqrt((nonego_velocity.x) ** 2 + (nonego_velocity.y) ** 2)
        if abs(nonego_speed) < 2:
            acc = 2
        else:
            acc = 0
        
        if len(self.nonego_waypoints) > 0:
            closest_waypoint = self.nonego_waypoints[0]

            if closest_waypoint != self.list_waypoints[0]: 
                self.list_waypoints.append(closest_waypoint)
                self.list_velocity.append(np.array([*get_vehicle_velocity(self.nonego)]))
                # print(self.list_velocity)

            # print(closest_waypoint)
            # print(len(self.nonego_waypoints))
            heading_angle = math.degrees(math.atan2(closest_waypoint[0] - nonego_loc[0], closest_waypoint[1] - nonego_loc[1]))
            if heading_angle < 90:
                heading_angle = 90 - heading_angle
            else: 
                heading_angle = 450 - heading_angle

            heading_error = heading_angle - get_vehicle_orientation(self.nonego)
            if heading_error > 180:
                heading_error -= 360
            elif heading_error < -180:
                heading_error += 360
            # heading_error = -heading_error

            coeffs = self._config.pid_coeffs
            self.control, self.prev_errors = self.pid_controller(heading_error, self.prev_errors, coeffs)
            # print(heading_angle, heading_error, control)

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc/3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc/3, 0, 1)

        return carla.VehicleControl(throttle=float(throttle), steer=np.clip(self.control, -1, 1), brake=float(brake))
        # return carla.VehicleControl(throttle=float(throttle), steer=control, brake=float(brake))
    
    def pid_controller(self, error, prev_errors, coeffs):
        """
        Calculate the PID control output to minimize the deviation.

        Args:
        target (float): The target for the PID controller (central line x-coordinate).
        current (float): The current measurement of the process variable (vehicle x-coordinate).
        prev_errors (dict): A dictionary holding the last error and the integral of errors.
        coeffs (tuple): A tuple of PID coefficients (Kp, Ki, Kd).

        Returns:
        float: The control output (steering angle adjustment).
        dict: Updated dictionary with the last error and integral.
        """
        Kp, Ki, Kd = coeffs
        integral = prev_errors['integral'] + error
        derivative = error - prev_errors['last_error']
        
        output = (Kp * error) + (Ki * integral) + (Kd * derivative)
        
        # Update the errors for the next call
        updated_errors = {'last_error': error, 'integral': integral}
        
        return output, updated_errors

    
    def reward(self):
        total_reward, info = super().reward()
        total_reward -= info['r_speed'] + info['r_waypoints']
        del info['r_speed']
        del info['r_waypoints']
        # total_reward -= info['waypoint']
        # del info['waypoint']

        reward_scales = self._config.reward.scales
        
        original_dist = get_location_distance((self._config.lane_start_points[self.random_num][0], self._config.lane_start_points[self.random_num][1]),
                                              (self._config.nonego_spawn_points[self.random_num][0], self._config.nonego_spawn_points[self.random_num][1]))
        current_dist = get_location_distance(get_vehicle_pos(self.ego), get_vehicle_pos(self.nonego))
        
        if current_dist < original_dist + 2 and current_dist >= original_dist:
            p_dist = reward_scales["distance"]
        else:
            p_dist = - abs(current_dist - original_dist) * reward_scales["distance"]
        
        ego_velocity = np.array([*get_vehicle_velocity(self.ego)])
        if np.array_equal(ego_velocity, self.list_velocity[0]):
            p_velocity = reward_scales["velocity"]
        else:
            p_velocity = -reward_scales["velocity"]

        if get_vehicle_pos(self.ego) == self.list_waypoints[0]:
            p_waypoints = reward_scales["waypoints"]
            self.list_velocity.pop(0)
            self.list_waypoints.pop(0)
        else:
            p_waypoints = -reward_scales["waypoints"]
        
        total_reward += p_dist + p_waypoints + p_velocity
        # total_reward += p_dist + p_waypoints
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