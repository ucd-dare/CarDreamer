import numpy as np
import tensorflow as tf


class RandomAgent:
    def __init__(self, act_space):
        self.act_space = act_space

    def policy(self, obs, state=None, mode="train"):
        batch_size = len(next(iter(obs.values())))
        act = {k: np.stack([v.sample() for _ in range(batch_size)]) for k, v in self.act_space.items() if k != "reset"}
        return act, state


class RandomCarryAgent:
    def __init__(self, act_space, skill_shape):
        self.act_space = act_space
        self.skill_shape = skill_shape  # The shape of the skill tensor

    def policy(self, obs, state=None, mode="train"):
        batch_size = len(next(iter(obs.values())))
        act = {k: np.stack([v.sample() for _ in range(batch_size)]) for k, v in self.act_space.items() if k != "reset"}

        skills = tf.one_hot(
            indices=tf.random.uniform(shape=[batch_size], minval=0, maxval=self.skill_shape[1], dtype=tf.int32),
            depth=self.skill_shape[1],
        )

        carry = [{}, {}, {"skill": skills}, {}]

        return act, carry
