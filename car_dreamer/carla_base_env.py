from abc import abstractmethod
from typing import Dict, Tuple

import carla
import gym
import numpy as np
from gym import spaces

from .toolkit import EnvMonitorOpenCV, Observer, WorldManager


class CarlaBaseEnv(gym.Env):
    def __init__(self, config):
        self._config = config

        self._monitor = EnvMonitorOpenCV(self._config)
        self._world = WorldManager(self._config)
        self._world.on_reset(self.on_reset)
        self._world.on_step(self.on_step)
        self._observer = Observer(self._world, self._config.observation)

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    @abstractmethod
    def on_reset(self) -> None:
        """
        Override this method to perform additional reset operations.
        Specifically, you can spawn actors and plan routes here.
        """
        pass

    @abstractmethod
    def apply_control(self, action) -> None:
        """
        Override this method to apply control to actors.
        This method will be called before the simulator ticks.
        """
        pass

    @abstractmethod
    def on_step(self) -> None:
        """
        Override this method to perform additional operations at each step.
        Specifically, you can update the planner and the route here.
        This method will be called after the simulator ticks.
        """
        pass

    @abstractmethod
    def reward(self) -> Tuple[float, Dict]:
        """
        Override this method to define the reward function.
        """
        pass

    @abstractmethod
    def get_terminal_conditions(self) -> Dict[str, bool]:
        """
        Override this method to define the terminal condition.
        If one of the keys in the returned dictionary gives True, the episode will be terminated.
        """
        pass

    def get_ego_vehicle(self) -> carla.Actor:
        """
        Override this method to return the ego vehicle.
        The default behavior is to return self.ego
        """
        return self.ego

    def get_state(self) -> Dict:
        """Return the environment state. Implement this method to define the env state."""
        return self._state

    def _get_action_space(self):
        action_config = self._config.action
        if action_config.discrete:
            self.n_steer = len(action_config.discrete_steer)
            self.n_acc = len(action_config.discrete_acc)
            return spaces.Discrete(self.n_steer * self.n_acc)
        else:
            return spaces.Box(
                low=np.array([action_config.continuous_acc[0], action_config.continuous_steer[0]]),
                high=np.array([action_config.continuous_acc[1], action_config.continuous_steer[1]]),
                dtype=np.float32,
            )

    def _get_observation_space(self):
        return self._observer.get_observation_space()

    def reset(self):
        print("[CARLA] Reset environment")

        self._observer.destroy()
        self._world.reset()
        self._observer.reset(self.get_ego_vehicle())

        self._time_step = 0

        print("[CARLA] Environment reset")
        self.obs, _ = self._observer.get_observation(self.get_state())
        return self.obs

    def get_vehicle_control(self, action):
        """
        Convert actions in the action space to vehicle control in CARLA
        """
        action_config = self._config.action
        # Calculate acceleration and steering
        if action_config.discrete:
            acc = action_config.discrete_acc[action // self.n_steer]
            steer = action_config.discrete_steer[action % self.n_steer]
        else:
            acc = action[0]
            steer = action[1]
        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3, 0, 1)

        return carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))

    def _is_terminal(self):
        terminal_conds = self.get_terminal_conditions()
        terminal = False
        for k, v in terminal_conds.items():
            if v:
                print(f"[CARLA] Terminal condition triggered: {k}")
                terminal = True
            terminal_conds[k] = np.array([v], dtype=np.bool_)
        if terminal:
            terminal_conds["episode_timesteps"] = self._time_step
        terminal_conds["terminal"] = terminal
        return terminal, terminal_conds

    def step(self, action):
        self.apply_control(action)
        self._world.step()
        self._time_step += 1

        env_state = self.get_state()
        is_terminal, terminal_conds = self._is_terminal()
        self.obs, obs_info = self._observer.get_observation(env_state)
        reward, reward_info = self.reward()

        info = {
            **env_state,
            **terminal_conds,
            **obs_info,
            **reward_info,
            "action": action,
        }
        if self._config.eval:
            info = {f"eval_{k}": v for k, v in info.items()}
            self.obs = {**self.obs, **info}
        if self._config.display.enable:
            self._render(self.obs, info)

        return (self.obs, reward, is_terminal, info)

    def is_collision(self):
        """
        Check if the ego vehicle is in collision.
        You must include 'collsion' in observation.names to use this method.
        """
        return self.obs["collision"][0] > 0

    def _render(self, obs, info):
        self._monitor.render(obs, info)
