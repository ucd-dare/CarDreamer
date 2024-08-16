import random

import numpy as np


class RandomAgent:
    def __init__(self, act_space, actor_dist_disc):
        self.act_space = act_space
        self.actor_dist_disc = actor_dist_disc

    def policy(self, obs, state=None, mode="train"):
        batch_size = len(next(iter(obs.values())))

        if self.actor_dist_disc != "twohot":
            act = {k: np.stack([v.sample() for _ in range(batch_size)]) for k, v in self.act_space.items() if k != "reset"}
            return act, state
