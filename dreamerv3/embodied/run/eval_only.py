import os
import re

import embodied
import numpy as np
from jax import tree_map


def average_ensemble(checkpoint, filenames, save_filename):
    num_checkpoints = len(filenames)
    all_varibs = []
    if num_checkpoints == 0:
        raise ValueError("No filenames provided to ensemble.")

    for filename in filenames:
        checkpoint.load(filename, keys=["agent"])
        agent = checkpoint.get("agent")
        all_varibs.append(agent.save())

    # weights = np.array([0.5, 0.3, 0.2])  # Example weights, replace with actual weights
    # ensemble_varibs = tree_map(lambda *xs: np.average(xs, axis=0, weights=weights), *all_varibs)

    ensemble_varibs = tree_map(lambda *xs: np.mean(xs, axis=0), *all_varibs)

    # Load average values into a new agent
    agent.load(ensemble_varibs)
    checkpoint.agent = agent

    # Save the new agent
    checkpoint.save(save_filename)
    return checkpoint


def eval_only(agent, env, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_log = embodied.when.Clock(args.log_every)
    step = logger.step
    metrics = embodied.Metrics()
    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add({"length": length, "score": score}, prefix="episode")
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep))
    driver.on_step(lambda tran, info, _: step.increment())

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, keys=["agent"])
    elif args.ensemble.from_checkpoints is not None and len(args.ensemble.from_checkpoints) > 0:
        ensemble_model_path = os.path.join(args.logdir, "ensemble_model.ckpt")
        checkpoint = average_ensemble(checkpoint, args.ensemble.from_checkpoints, ensemble_model_path)
    else:
        raise ValueError("No checkpoint specified.")

    print("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")
    while step < args.steps:
        driver(policy, steps=100)
        if should_log(step):
            logger.add(metrics.result())
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)
    logger.write()
