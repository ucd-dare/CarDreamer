import re
import numpy as np
import dreamerv2 as dm2
import ruamel.yaml as yaml


def eval_only(agent, env, replay, eval_replay, logger, args):
    print("Start evaluation.")
    print("args:", args)
    step = logger.step
    should_log = dm2.when.Every(args.log_every)

    timer = dm2.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])

    nonzeros = set()

    def per_episode(ep, ep_info):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {"length": length, "score": score}

        for key in args.log_keys_video:
            if key == "none":
                continue
            if key in ep:
                stats[f"policy_{key}"] = ep[key]

        def log(key, value):
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = value.sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = value.mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = value.max(0).mean()

        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            log(key, value)
        for key, value in ep_info.items():
            log(key, value)

        logger.add(stats, prefix="stats")
        logger.add(timer.stats(), prefix="timer")
        logger.write(fps=True)

    def per_step(tran, worker):
        step.increment()
        if should_log(step):
            logger.write()

    driver = dm2.Driver(env)
    driver.on_episode(lambda ep, info, worker: per_episode(ep, info))
    driver.on_step(lambda tran, info, worker: per_step(tran, worker))
    driver.on_step(lambda ep, info, worker: replay.add(ep, worker))
    fill = args.eval_fill - len(eval_replay)
    if fill:
        print(f"Fill train dataset ({fill} steps).")
        driver(agent.policy, steps=fill, episodes=1)

    print("Initializing agent variables...")
    dataset_train = iter(agent.dataset(replay.dataset))
    state = None
    try:
        _, state, _ = agent.train(next(dataset_train), state)
    except Exception as e:
        print(f"Warning: Could not initialize through training: {e}")
        print("Attempting alternative initialization...")
        # Alternative initialization if training fails
        dummy_data = next(iter(agent.dataset(lambda: iter([{}]))))
        _, state, _ = agent.train(dummy_data, None)

    print("Loading checkpoint...")
    checkpoint = dm2.Checkpoint(args.from_checkpoint)
    checkpoint.step = step
    checkpoint.train_replay = replay
    checkpoint.eval_replay = eval_replay
    checkpoint.agent = agent

    try:
        checkpoint.load()
    except Exception as e:
        print(f"Error during checkpoint loading: {e}")
        print("Agent structure before loading:", agent.save())
        raise

    step.load(0)
    print("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")
    while step < args.steps:
        driver(policy, steps=100)
    logger.write()
