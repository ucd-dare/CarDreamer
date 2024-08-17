import math

import carla
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_vehicle_pos


class CarlaOvertakeEnv(CarlaWptEnv):
    """
    This task places a slow vehicle in front of the ego vehicle for overtaking.

    **Provided Tasks**: ``carla_overtake``

    Available config parameters:

    * ``swing_steer``: The background vehicle steer for swing.
    * ``swing_amplitude``: The y-axis amplitude of background vehicle steer.
    * ``swing_trigger_dist``: The distance between ego and background vehicle that triggers swing.
    * ``pid_coeffs``: The PID controller parameter for background vehicle lane keeping.
    * ``reward_overtake_dist``: The distance from background vehicle to ego vehicle that triggers overtake reward.
    * ``early_lane_change_dist``: The distance that penalizes early lane change.
    * ``lane_width``: The width of the lane.
    * ``stay_same_lane``: The penalty for stay in the same lane when approaching the background vehicle.
    * ``overtake``: The reward for overtaking.
    * ``early_lane_change``: The reward for early lane change.

    """

    def on_reset(self) -> None:
        self.nonego_spawn_point = self._config.nonego_spawn_points[np.random.randint(0, len(self._config.nonego_spawn_points) - 1)]
        nonego_transform = carla.Transform(
            carla.Location(*self.nonego_spawn_point[:3]),
            carla.Rotation(*self.nonego_spawn_point[-3:]),
        )
        self.nonego = self._world.spawn_actor(transform=nonego_transform)
        self.ego_src = self._config.lane_start_points[np.random.randint(0, len(self._config.lane_start_points) - 1)]
        ego_transform = carla.Transform(
            carla.Location(x=self.nonego_spawn_point[0], y=self.ego_src[1], z=self.ego_src[2]),
            carla.Rotation(yaw=-90),
        )  # set x that always behind nonego
        self.ego = self._world.spawn_actor(transform=ego_transform)

        # Path planning
        ego_dest = self._config.lane_end_points
        dest_location = carla.Location(x=self.nonego_spawn_point[0], y=ego_dest[0][1], z=ego_dest[0][2])
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

        self.exceeding = False
        self.overtake = False
        self.last_ego_y = self.ego_src[1]

        # Set spectator for debugging
        spectator = self._world._world.get_spectator()
        ego_transform.location.z += 150
        ego_transform.rotation.pitch = -70
        spectator.set_transform(ego_transform)
        self.swing_direction = 1

        self.prev_errors = {"last_error": 0.0, "integral": 0.0}  # For PID controller

    def apply_control(self, action) -> None:
        control = self.get_vehicle_control(action)
        nonego_control = self.get_nonego_vehicle_control()
        self.ego.apply_control(control)
        self.nonego.apply_control(nonego_control)

    def get_nonego_vehicle_control(self):
        """
        Non-ego vehicle control is designed for the scenario.
        """
        ego_loc = self.ego.get_transform().location
        nonego_loc = self.nonego.get_transform().location

        # Keep constant speed
        if abs(self.nonego.get_velocity().y) < 2:
            acc = 2
        else:
            acc = 0

        dist = math.sqrt((ego_loc.x - nonego_loc.x) ** 2 + (ego_loc.y - nonego_loc.y) ** 2)
        swing_steer = self._config.swing_steer
        swing_amplitude = self._config.swing_amplitude
        swing_trigger_dist = self._config.swing_trigger_dist
        if dist < swing_trigger_dist:
            # Swing when ego vehicle approaching
            if self.nonego_spawn_point[0] + swing_amplitude <= nonego_loc.x:
                self.swing_direction = 1
            if self.nonego_spawn_point[0] - swing_amplitude >= nonego_loc.x:
                self.swing_direction = -1
            steer = swing_steer * self.swing_direction
            self.prev_errors = {
                "last_error": 0.0,
                "integral": 0.0,
            }  # Reset the prev_error
        else:
            # Implement PID controller for lane keeping
            coeffs = self._config.pid_coeffs
            steer, updated_errors = self.pid_controller(self.nonego_spawn_point[0], nonego_loc.x, self.prev_errors, coeffs)
            self.prev_errors.update(updated_errors)

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3, 0, 1)

        return carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))

    def pid_controller(self, target, current, prev_errors, coeffs):
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
        error = current - target
        integral = prev_errors["integral"] + error
        derivative = error - prev_errors["last_error"]

        output = (Kp * error) + (Ki * integral) + (Kd * derivative)

        # Update the errors for the next call
        updated_errors = {"last_error": error, "integral": integral}

        return output, updated_errors

    def reward(self):
        total_reward, info = super().reward()
        # remove the out of lane penalty
        total_reward -= info["r_out_of_lane"]
        del info["r_out_of_lane"]

        reward_scales = self._config.reward.scales
        ego = self.ego
        ego_x, ego_y = get_vehicle_pos(ego)
        nonego_spawn_x = self.nonego_spawn_point[0]

        # Reward vehicle to stay in the lane, while penalize vehicle staying in the lane when overtaking.
        if (
            ego_y - self._config.reward.early_lane_change_dist < self.nonego.get_transform().location.y
            and ego_y + self._config.reward.reward_overtake_dist > self.nonego.get_transform().location.y
        ):
            p_stay_same_lane = -1 / (0.5 + abs(ego_y - self.nonego.get_transform().location.y)) * reward_scales["stay_same_lane"]
        else:
            p_stay_same_lane = 1 / (0.5 + abs(ego_y - self.nonego.get_transform().location.y)) * reward_scales["stay_same_lane"]

        # Penalty for early lane change before overtake
        p_early_lane_change = 0.0
        if (
            ego_y - self._config.reward.early_lane_change_dist > self.nonego.get_transform().location.y
            and abs(ego_x - nonego_spawn_x) > self._config.reward.lane_width
        ):
            p_early_lane_change = -reward_scales["early_lane_change"]

        # Exceeding reward
        r_exceeding = 0.0
        if ego_y < self.nonego.get_transform().location.y and not self.exceeding:
            r_exceeding = reward_scales["exceeding"]
            self.exceeding = True

        # Overtake reward (exceed and come back to the same lane)
        r_overtake = 0.0
        if (
            ego_y + self._config.reward.reward_overtake_dist < self.nonego.get_transform().location.y
            and abs(ego_x - nonego_spawn_x) < self._config.terminal.lane_width / 5
            and not self.overtake
        ):
            r_overtake = reward_scales["overtake"]
            self.overtake = True

        # Total reward
        total_reward += p_stay_same_lane + p_early_lane_change + r_exceeding + r_overtake

        info.update(
            {
                "p_stay_same_lane": p_stay_same_lane,
                "r_exceeding": r_exceeding,
                "r_overtake": r_overtake,
                "p_early_lane_change": p_early_lane_change,
            }
        )

        return total_reward, info

    def get_terminal_conditions(self):
        ego_x = self.ego.get_location().x
        ego_location = get_vehicle_pos(self.get_ego_vehicle())
        terminal_config = self._config.terminal
        info = super().get_terminal_conditions()
        info["out_of_lane"] = (
            self.get_wpt_dist(ego_location) > terminal_config.out_lane_thres
            or ego_x < terminal_config.left_lane_boundry
            or ego_x > terminal_config.right_lane_boundry
        )
        return info
