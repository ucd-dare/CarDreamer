import re
import time

import numpy as np

import dreamerv2 as dm2


def acting(agent, env, replay, logger, actordir, args):
    logdir = dm2.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir:", logdir)
    actordir = dm2.Path(actordir)
    actordir.mkdirs()
    should_sync = dm2.when.Clock(args.sync_every)
    should_expl = dm2.when.Until(args.expl_until)
    should_video = dm2.when.Every(args.eval_every)
    step = logger.step

    timer = dm2.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])

    nonzeros = set()

    def per_episode(ep):
        metrics = {}
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"Episode has {length} steps and return {score:.1f}.")
        metrics["length"] = length
        metrics["score"] = score
        metrics["reward_rate"] = (ep["reward"] - ep["reward"].min() >= 0.1).mean()
        logs = {}
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                logs[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                logs[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                logs[f"max_{key}"] = ep[key].max(0).mean()
        if should_video(step):
            for key in args.log_keys_video:
                metrics[f"policy_{key}"] = ep[key]
        logger.add(metrics, prefix="episode")
        logger.add(logs, prefix="logs")
        logger.add(replay.stats, prefix="replay")
        logger.write()

    driver = dm2.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(replay.add)

    actor_cp = dm2.Checkpoint(actordir / "actor.pkl")
    actor_cp.step = step
    actor_cp.load_or_save()

    fill = max(0, args.train_fill - int(step))
    if fill:
        print(f"Fill dataset ({fill} steps).")
        random_agent = dm2.RandomAgent(env.act_space)
        driver(random_agent.policy, steps=fill, episodes=1)

    # Initialize dataset and agent variables.
    agent.train(next(iter(agent.dataset(replay.dataset))))

    agent_cp = dm2.Checkpoint(logdir / "agent.pkl")
    agent_cp.agent = agent

    print("Start collection loop.")
    policy = lambda *args: agent.policy(*args, mode="explore" if should_expl(step) else "train")

    while step < args.steps:
        if should_sync(step):
            print("Syncing.")
            actor_cp.save()

            while not agent_cp.exists():
                print("Waiting for agent checkpoint to be created.")
                time.sleep(10)
            for attempts in range(10):
                try:
                    timestamp = agent_cp.load()
                    if timestamp:
                        logger.scalar("agent_cp_age", time.time() - timestamp)
                    break
                except Exception as e:
                    print(f"Could not load checkpoint: {e}")
                time.sleep(np.random.uniform(10, 60))
            else:
                raise RuntimeError("Failed to load checkpoint.")

        driver(policy, steps=100)
