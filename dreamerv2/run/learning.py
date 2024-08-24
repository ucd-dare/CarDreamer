import collections
import time
import warnings

import numpy as np

import dreamerv2 as dm2


def learning(agent, train_replay, eval_replay, logger, args):
    logdir = dm2.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_sync = dm2.when.Clock(args.sync_every)
    should_log = dm2.when.Every(args.log_every)
    should_eval = dm2.when.Every(args.eval_every)
    step = logger.step

    timer = dm2.Timer()
    timer.wrap("agent", agent, ["train", "report", "save"])
    if hasattr(train_replay, "_sample"):
        timer.wrap("replay", train_replay, ["_sample"])

    print("Initializing training replay...")
    dataset_train = iter(agent.dataset(train_replay.dataset))
    # Initialize on first eval, so we don't wait needlessly for first data here
    dataset_eval = None
    print("Initializing agent...")
    _, state, _ = agent.train(next(dataset_train))
    print("Initialization done.")

    agent_cp = dm2.Checkpoint(logdir / "agent.pkl")
    agent_cp.agent = agent
    agent_cp.load_or_save()

    learner_cp = dm2.Checkpoint(logdir / "learner.pkl")
    learner_cp.train_replay = train_replay
    learner_cp.step = step
    learner_cp.load_or_save()

    # Wait for prefill data from at least one actor to avoid overfitting to only
    # small amount of data that is read first.
    while len(train_replay) < args.train_fill:
        print("Waiting for train data prefill...")
        time.sleep(10)

    print("Start loop...")
    metrics = collections.defaultdict(list)

    while step < args.steps:
        batch = next(dataset_train)
        outs, state, mets = agent.train(batch, state)
        [metrics[key].append(value) for key, value in mets.items()]
        if "priority" in outs:
            train_replay.prioritize(outs["key"], outs["priority"])
        step.increment()

        if should_log(step):
            with warnings.catch_warnings():  # Ignore empty slice warnings.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                agg = {k: np.nanmean(x, dtype=np.float64) for k, x in metrics.items()}
                logger.add(agg, prefix="train")
                metrics.clear()
            logger.add(agent.report(batch), prefix="report")
            if dataset_eval:
                logger.add(agent.report(next(dataset_eval)), prefix="report_eval")
            logger.add(train_replay.stats, prefix="replay")
            logger.add(eval_replay.stats, prefix="replay_eval")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

        if should_sync(step):
            agent_cp.save()
            learner_cp.save()

        if should_eval(step):
            print("Evaluation.")
            if not dataset_eval:
                print("Initializing eval replay...")
                dataset_eval = iter(agent.dataset(eval_replay.dataset))
            scalars = collections.defaultdict(list)
            for _ in range(args.eval_samples):
                for key, value in agent.report(next(dataset_eval)).items():
                    if value.shape == ():
                        scalars[key].append(value)
            logger.add({k: np.mean(xs) for k, xs in scalars.items()}, prefix="eval")
            logger.write()
