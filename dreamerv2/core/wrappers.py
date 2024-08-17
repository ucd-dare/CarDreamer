import functools
import time

import numpy as np
import tensorflow as tf

from . import base
from . import space as spacelib


class InfoWrapper(base.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.supported_keys = None

    def _is_supported_type(self, value):
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            return True
        if isinstance(value, np.ndarray) and value.ndim == 1:
            return True
        return False

    def step(self, action):
        obs, info = self.env.step(action)
        if info is None or not isinstance(info, dict):
            return obs, {}
        if self.supported_keys is None:
            self.supported_keys = [key for key in info if self._is_supported_type(info[key])]
            print("Supported info keys cached:", self.supported_keys)
        info_filtered = {key: info[key] for key in self.supported_keys if key in info}
        return obs, info_filtered


class CarryWrapper(base.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._require_carry = True

    def set_carry(self, carry):
        self._carry = carry

    def step(self, action):
        if hasattr(self._env.unwrapped, "set_carry") and self._carry is not None:
            state = tf.nest.map_structure(lambda x: x.numpy(), self._carry)
            self._env.unwrapped.set_carry(state)
        return self.env.step(action)


class TimeLimit(base.Wrapper):
    def __init__(self, env, duration, reset=True):
        super().__init__(env)
        self._duration = duration
        self._reset = reset
        self._step = 0
        self._done = False

    def step(self, action):
        if action["reset"] or self._done:
            self._step = 0
            self._done = False
            if self._reset:
                action.update(reset=True)
                return self.env.step(action)
            else:
                action.update(reset=False)
                obs = self.env.step(action)
                obs["is_first"] = True
                return obs
        self._step += 1
        obs = self.env.step(action)
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
        self._done = obs["is_last"]
        return obs


class ActionRepeat(base.Wrapper):
    def __init__(self, env, repeat):
        super().__init__(env)
        self._repeat = repeat
        self._done = False

    def step(self, action):
        if action["reset"] or self._done:
            return self.env.step(action)
        reward = 0.0
        for _ in range(self._repeat):
            obs = self.env.step(action)
            reward += obs["reward"]
            if obs["is_last"] or obs["is_terminal"]:
                break
        obs["reward"] = reward
        self._done = obs["is_last"]
        return obs


class NormalizeAction(base.Wrapper):
    def __init__(self, env, key="action"):
        super().__init__(env)
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    @functools.cached_property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = spacelib.Space(np.float32, None, low, high)
        return {**self.env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self.env.step({**action, self._key: orig})


class OneHotAction(base.Wrapper):
    def __init__(self, env, key="action"):
        super().__init__(env)
        self._count = int(env.act_space[key].high)
        self._key = key

    @functools.cached_property
    def act_space(self):
        shape = (self._count,)
        space = spacelib.Space(np.float32, shape, 0, 1)
        space.sample = functools.partial(self._sample_action, self._count)
        space.discrete = True
        return {**self.env.act_space, self._key: space}

    def step(self, action):
        if not action["reset"]:
            assert (action[self._key].min() == 0.0).all(), action
            assert (action[self._key].max() == 1.0).all(), action
            assert (action[self._key].sum() == 1.0).all(), action
        index = np.argmax(action[self._key])
        return self.env.step({**action, self._key: index})

    @staticmethod
    def _sample_action(count):
        index = np.random.randint(0, count)
        action = np.zeros(count, dtype=np.float32)
        action[index] = 1.0
        return action


class ExpandScalars(base.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._obs_expanded = []
        self._act_expanded = []

    @functools.cached_property
    def obs_space(self):
        spaces = self.env.obs_space.copy()
        for key, value in spaces.items():
            if value.shape == () and key != "reward" and not value.discrete:
                value.shape = (1,)
                self._obs_expanded.append(key)
        return spaces

    @functools.cached_property
    def act_space(self):
        spaces = self.env.act_space.copy()
        for key, value in spaces.items():
            if value.shape == () and not value.discrete:
                value.shape = (1,)
                self._act_expanded.append(key)
        return spaces

    def step(self, action):
        action = {key: np.squeeze(value, 0) if key in self._act_expanded else value for key, value in action.items()}
        obs = self.env.step(action)
        obs = {key: np.expand_dims(value, 0) if key in self._obs_expanded else value for key, value in obs.items()}
        return obs


class FlattenTwoDimObs(base.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._keys = []
        self._obs_space = self.env.obs_space.copy()
        for key, value in self._obs_space.items():
            if len(value.shape) == 2:
                value.shape = (int(np.prod(value.shape)),)
                self._keys.append(key)

    @functools.cached_property
    def obs_space(self):
        return self._obs_space

    def step(self, action):
        obs = self.env.step(action).copy()
        for key in self._keys:
            obs[key] = obs[key].flatten()
        return obs


class CheckSpaces(base.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        for key, value in action.items():
            space = self.env.act_space[key]
            assert value in space, (
                value.dtype,
                value.shape,
                np.min(value),
                np.max(value),
                space,
            )
        obs = self.env.step(action)
        for key, value in obs.items():
            space = self.env.obs_space[key]
            assert value in space, (
                key,
                np.array(value).dtype,
                np.array(value).shape,
                np.min(value),
                np.max(value),
                space,
            )
        return obs


class DiscretizeAction(base.Wrapper):
    def __init__(self, env, key="action", bins=5):
        super().__init__(env)
        self._dims = np.squeeze(env.act_space[key].shape, 0).item()
        self._values = np.linspace(-1, 1, bins)
        self._key = key

    @functools.cached_property
    def act_space(self):
        shape = (self._dims, len(self._values))
        space = spacelib.Space(np.float32, shape, 0, 1)
        space.sample = functools.partial(self._sample_action, self._dims, self._values)
        space.discrete = True
        return {**self.env.act_space, self._key: space}

    def step(self, action):
        if not action["reset"]:
            assert (action[self._key].min(-1) == 0.0).all(), action
            assert (action[self._key].max(-1) == 1.0).all(), action
            assert (action[self._key].sum(-1) == 1.0).all(), action
        indices = np.argmax(action[self._key], axis=-1)
        continuous = np.take(self._values, indices)
        return self.env.step({**action, self._key: continuous})

    @staticmethod
    def _sample_action(dims, values):
        indices = np.random.randint(0, len(values), dims)
        action = np.zeros((dims, len(values)), dtype=np.float32)
        action[np.arange(dims), indices] = 1.0
        return action


class ResizeImage(base.Wrapper):
    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self._size = size
        self._keys = [k for k, v in env.obs_space.items() if len(v.shape) > 1 and v.shape[:2] != size]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image

            self._Image = Image

    @functools.cached_property
    def obs_space(self):
        spaces = self.env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = spacelib.Space(np.uint8, shape)
        return spaces

    def step(self, action):
        obs = self.env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage(base.Wrapper):
    def __init__(self, env, key="image"):
        super().__init__(env)
        self._key = key
        self._shape = self.env.render().shape

    @functools.cached_property
    def obs_space(self):
        spaces = self.env.obs_space
        spaces[self._key] = spacelib.Space(np.uint8, self._shape)
        return spaces

    def step(self, action):
        obs = self.env.step(action)
        obs[self._key] = self.env.render()
        return obs


class RestartOnException(base.Wrapper):
    def __init__(self, ctor, exceptions=(Exception,), window=300, maxfails=2, wait=20):
        if not isinstance(exceptions, (tuple, list)):
            exceptions = [exceptions]
        self._ctor = ctor
        self._exceptions = tuple(exceptions)
        self._window = window
        self._maxfails = maxfails
        self._wait = wait
        self._last = time.time()
        self._fails = 0
        super().__init__(self._ctor())

    def step(self, action):
        try:
            return self.env.step(action)
        except self._exceptions as e:
            if time.time() > self._last + self._window:
                self._last = time.time()
                self._fails = 1
            else:
                self._fails += 1
            if self._fails > self._maxfails:
                raise RuntimeError("The env crashed too many times.")
            message = f"Restarting env after crash with {type(e).__name__}: {e}"
            print(message, flush=True)
            time.sleep(self._wait)
            self.env = self._ctor()
            action["reset"] = np.ones_like(action["reset"])
            return self.env.step(action)
