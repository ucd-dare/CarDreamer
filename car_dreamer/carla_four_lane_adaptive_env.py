import carla
import numpy as np

from .carla_four_lane_env import CarlaFourLaneEnv
from .toolkit import FixedPathPlanner


class CarlaFourLaneAdaptiveEnv(CarlaFourLaneEnv):
    """
    Adaptive version of the four-lane driving task.

    **Provided Tasks**: ``carla_four_lane_adaptive``

    Available config parameters:
      * lane_start_points, lane_end_points, num_vehicles (inherited)
      * stuck_speed_threshold: speed below which the vehicle is considered stuck
      * stuck_timeout: number of consecutive steps below threshold to trigger reset
      * max_steps: hard cap on episode length (fallback)
      * adaptive_step_window: window size to compute avg recent reward
      * low_reward_threshold: avg reward below which to end early
    """

    def __init__(self, config):
        super().__init__(config)
        # adaptive counters & history
        self._stuck_counter = 0
        self._step_counter = 0
        self._reward_history = []

    def on_reset(self) -> None:
        # --- reset adaptive trackers ---
        self._stuck_counter = 0
        self._step_counter = 0
        self._reward_history.clear()

        # --- replicate parent spawn & path logic WITHOUT initial run_step() ---
        # pick random start
        self.ego_src = self._config.lane_start_points[
            np.random.randint(0, len(self._config.lane_start_points))
        ]
        ego_transform = carla.Transform(
            carla.Location(x=self.ego_src[0], y=self.ego_src[1], z=self.ego_src[2]),
            carla.Rotation(yaw=-90),
        )
        self.ego = self._world.spawn_actor(transform=ego_transform)
        self._world.spawn_auto_actors(self._config.num_vehicles)

        # generate path & planner
        self.ego_path = self.generate_vehicle_path(
            self.ego_src, self._config.lane_end_points[0][1]
        )
        self.ego_dest = self.ego_path[-1]
        self.ego_planner = FixedPathPlanner(vehicle=self.ego, vehicle_path=self.ego_path)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]
        # **Do not** call planner.run_step() here to avoid immediate destination check

    def step(self, action):
        # perform one env step (this will internally advance the planner)
        obs, reward, done, info = super().step(action)

        # update adaptive trackers
        self._step_counter += 1
        self._reward_history.append(reward)

        # --- 1) stuck detection ---
        vel = self.ego.get_velocity()
        speed = np.linalg.norm([vel.x, vel.y, vel.z])
        if speed < self._config.get("stuck_speed_threshold", 0.6):
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        if self._stuck_counter >= self._config.get("stuck_timeout", 15):
            # short‐circuit episode: mark done, reason='stuck_timeout'
            done = True
            info = dict(info)
            info['reason'] = 'stuck_timeout'
            return obs, reward, done, info

        # --- 2) low‐reward detection once enough history has accumulated ---
        window   = self._config.get("adaptive_step_window", 25)
        low_thr  = self._config.get("low_reward_threshold", 0.3)
        # Calculate the average immediately if the reward is stacked by the size of the window
        if len(self._reward_history) >= window:
            recent    = self._reward_history[-window:]
            avg_recent = float(np.mean(recent))
            if avg_recent < low_thr:
                done = True
                info = dict(info)
                info['reason'] = 'low_reward_early_stop'
                return obs, reward, done, info

        # --- 3) hard cap on steps (fallback) ---
        max_steps = self._config.get("max_steps", 1000)
        if self._step_counter >= max_steps:
            done = True
            info = dict(info)
            info['reason'] = 'max_steps_exceeded'
            return obs, reward, done, info
    
        return obs, reward, done, info
    # Inherit generate_vehicle_path() unchanged from parent
