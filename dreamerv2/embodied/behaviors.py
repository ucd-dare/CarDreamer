import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import agent, expl, tfutils


class Greedy(tfutils.Module):
    def __init__(self, wm, act_space, config):
        rewfn = lambda s: wm.heads["reward"](s).mean()[1:]
        if config.critic_type == "vfunction":
            critics = {"extr": agent.VFunction(rewfn, config)}
        elif config.critic_type == "qfunction":
            critics = {"extr": agent.QFunction(rewfn, config)}
        self.ac = agent.ImagActorCritic(critics, {"extr": 1.0}, act_space, config)

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        return self.ac.train(imagine, start, data)

    def report(self, data):
        return {}


class Random(tfutils.Module):
    def __init__(self, wm, act_space, config):
        self.config = config
        self.act_space = act_space

    def initial(self, batch_size):
        return tf.zeros(batch_size)

    def policy(self, latent, state):
        batch_size = len(state)
        shape = (batch_size,) + self.act_space.shape
        if self.act_space.discrete:
            dist = tfutils.OneHotDist(tf.zeros(shape))
        else:
            dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
            dist = tfd.Independent(dist, 1)
        return {"action": dist}, state

    def train(self, imagine, start, data):
        return None, {}

    def report(self, data):
        return {}


class Explore(tfutils.Module):
    REWARDS = {
        "disag": expl.Disag,
        "vae": expl.LatentVAE,
        "ctrl": expl.CtrlDisag,
        "pbe": expl.PBE,
    }

    def __init__(self, wm, act_space, config):
        self.config = config
        self.rewards = {}
        critics = {}
        for key, scale in config.expl_rewards.items():
            if not scale:
                continue
            if key == "extr":
                reward = lambda traj: wm.heads["reward"](traj).mean()[1:]
                critics[key] = agent.VFunction(reward, config)
            else:
                reward = self.REWARDS[key](wm, act_space, config)
                critics[key] = agent.VFunction(
                    reward,
                    config.update(
                        discount=config.expl_discount,
                        retnorm=config.expl_retnorm,
                    ),
                )
                self.rewards[key] = reward
        scales = {k: v for k, v in config.expl_rewards.items() if v}
        self.ac = agent.ImagActorCritic(critics, scales, act_space, config)

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        metrics = {}
        for key, reward in self.rewards.items():
            metrics.update(reward.train(data))
        traj, mets = self.ac.train(imagine, start, data)
        metrics.update(mets)
        return traj, metrics

    def report(self, data):
        return {}
