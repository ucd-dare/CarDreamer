from abc import abstractmethod

import numpy as np

from .carla_base_env import CarlaBaseEnv
from .toolkit import BasePlanner, TTCCalculator, get_location_distance, get_vehicle_pos, get_vehicle_velocity


class CarlaWptEnv(CarlaBaseEnv):
    """
    This is the base env for all waypoint following tasks.
    An ``ego_planner`` is required to provide waypoints for the ego vehicle.
    **DO NOT** instantiate this class directly.

    All envs that inherit from this class also inherits the following config parameters:

    * ``reward``: Reward configuration.

        * ``desired_speed``: Desired speed for the ego vehicle.
        * ``scales``: Dictionary of reward scales.

            * ``waypoint``: Reward for reaching waypoints.
            * ``speed``: Reward for speed.
            * ``collision``: Penalty for collision.
            * ``out_of_lane``: Penalty for going out of lane.
            * ``time``: Penalty for each time step.

    * ``terminal``: Terminal condition configuration.

        * ``time_limit``: Maximum number of time steps.
        * ``out_lane_thres``: Distance threshold for going out of lane.

    """

    @abstractmethod
    def get_ego_planner(self) -> BasePlanner:
        """
        Override this method to return the ego vehicle planner.
        The default behavior is to return self.ego_planner.
        """
        return self.ego_planner

    def get_state(self):
        return {"ego_waypoints": self.waypoints, "timesteps": self._time_step}

    def apply_control(self, action) -> None:
        control = self.get_vehicle_control(action)
        self.get_ego_vehicle().apply_control(control)

    def on_step(self) -> None:
        self.waypoints, self.planner_stats = self.get_ego_planner().run_step()
        self.num_completed = self.planner_stats["num_completed"]

    def reward(self):
        reward_scales = self._config.reward.scales
        ego = self.get_ego_vehicle()
        ego_location = np.array([*get_vehicle_pos(ego)])
        ego_velocity = np.array([*get_vehicle_velocity(ego)])
        speed_norm = np.linalg.norm(ego_velocity)

        # Reward for reaching waypoints
        r_waypoints = 0.0
        if self.num_completed > 0:
            r_waypoints = reward_scales["waypoint"]

        # Reward for speed
        r_speed = 0.0
        speed_parallel = 0.0
        speed_perpendicular = 0.0
        if len(self.waypoints) > 0:
            # compute the wpt line direction
            next_waypoint = self.waypoints[0]
            next_location = np.array([next_waypoint[0], next_waypoint[1]])
            yaw_radius = next_waypoint[2] * np.pi / 180
            waypoint_direction = np.array([np.cos(yaw_radius), np.sin(yaw_radius)])

            # compute the perpendicular direction
            goal_offset = next_location - ego_location
            perp_direction = goal_offset - np.dot(goal_offset, waypoint_direction) * waypoint_direction
            perp_direction_norm = np.linalg.norm(perp_direction)
            if perp_direction_norm > 0.05:
                perp_direction = perp_direction / perp_direction_norm
            else:
                perp_direction = np.array([0.0, 0.0])

            # compute the speed reward
            desired_speed = self._config.reward.desired_speed
            speed_parallel = np.dot(ego_velocity, waypoint_direction)
            speed_perpendicular = np.dot(ego_velocity, perp_direction)
            r_speed = (desired_speed - np.abs(speed_parallel - desired_speed) - 2 * max(speed_perpendicular, -0.5)) * reward_scales["speed"]

        # Reward for collision
        r_collision = 0.0
        if reward_scales["collision"] > 0 and self.is_collision():
            r_collision = -reward_scales["collision"] * np.abs(speed_norm)

        # Reward for going out of lane
        r_out_of_lane = 0.0
        if len(self.waypoints) > 0:
            dist = perp_direction_norm
            if dist > 0.5:
                r_out_of_lane = -reward_scales["out_of_lane"] * (dist - 0.5)

        # Reward for reaching the destination
        r_destination = 0.0
        if self.is_destination_reached():
            r_destination = reward_scales["destination_reached"]

        # Time penalty
        time_penalty = -reward_scales["time"]

        # Total reward
        total_reward = r_waypoints + r_speed + r_collision + r_out_of_lane + r_destination + time_penalty

        ttc = TTCCalculator.get_ttc(ego, self._world.carla_world, self._world.carla_map)

        info = {
            **self.planner_stats,
            "ego_x": ego_location[0],
            "ego_y": ego_location[1],
            "speed_parallel": speed_parallel,
            "speed_perpendicular": speed_perpendicular,
            "speed_norm": speed_norm,
            "wpt_dis": self.get_wpt_dist(ego_location),
            "r_waypoints": r_waypoints,
            "r_speed": r_speed,
            "r_collision": r_collision,
            "r_out_of_lane": r_out_of_lane,
            "ttc": ttc,
        }

        return total_reward, info

    def is_destination_reached(self):
        return len(self.waypoints) <= 3

    def get_terminal_conditions(self):
        terminal_config = self._config.terminal
        ego_location = get_vehicle_pos(self.get_ego_vehicle())
        conds = {
            "is_collision": self.is_collision(),
            "time_exceeded": self._time_step > terminal_config.time_limit,
            "out_of_lane": self.get_wpt_dist(ego_location) > terminal_config.out_lane_thres,
            "destination_reached": self.is_destination_reached(),
        }
        return conds

    def get_wpt_dist(self, ego_location):
        if len(self.waypoints) == 0:
            return 0
        else:
            return get_location_distance(ego_location, self.waypoints[0])
