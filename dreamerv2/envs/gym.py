import functools

import gym
import numpy as np

import dreamerv2 as dm2


class FromGym(dm2.Env):

    def __init__(self, env, obs_key="image", act_key="action", **kwargs):
        if isinstance(env, str):
            self._env = gym.make(env, **kwargs)
        else:
            assert not kwargs, kwargs
            self._env = env
        self._obs_dict = hasattr(self._env.observation_space, "spaces")
        self._act_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None
        self._require_carry = False

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        if self._obs_dict:
            spaces = self._flatten(self._env.observation_space.spaces)
        else:
            spaces = {self._obs_key: self._env.observation_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        return {
            **spaces,
            "reward": dm2.Space(np.float32),
            "is_first": dm2.Space(bool),
            "is_last": dm2.Space(bool),
            "is_terminal": dm2.Space(bool),
        }

    @functools.cached_property
    def act_space(self):
        if self._act_dict:
            spaces = self._flatten(self._env.action_space.spaces)
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces["reset"] = dm2.Space(bool)
        return spaces

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            obs = self._env.reset()
            return self._obs(obs, 0.0, is_first=True), {}
        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]
        obs, reward, self._done, self._info = self._env.step(action)
        is_last = bool(self._done)
        is_terminal = bool(self._info.get("is_terminal", self._done))
        self._info = self._flatten(self._info)
        return self._obs(obs, reward, is_last=is_last, is_terminal=is_terminal), self._info

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        obs.update(reward=np.float32(reward), is_first=is_first, is_last=is_last, is_terminal=is_terminal)
        return obs

    def render(self):
        image = self._env.render("rgb_array")
        assert image is not None
        return image

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = str(prefix) + "/" + key if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, "n"):
            return dm2.Space(np.int32, (), 0, space.n)
        return dm2.Space(space.dtype, space.shape, space.low, space.high)
