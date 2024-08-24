import pathlib
import subprocess
import sys
import warnings
from datetime import datetime

import ruamel.yaml as yaml

warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*using stateful random seeds*")
warnings.filterwarnings("ignore", ".*is a deprecated alias for.*")

directory = pathlib.Path(__file__)
directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name


def save_configs(config, logdir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cardreamer_id = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(directory.parent)).decode("utf-8").strip()
    config = config.update(cardreamer_id=cardreamer_id)
    config_filename = f"config_{timestamp}.yaml"
    config_path = pathlib.Path(logdir) / config_filename
    config.save(str(config_path))


import dreamerv2 as dm2
from car_dreamer import load_task_configs


def main(argv=None):
    from dreamerv2.embodied import agent as agnt
    from dreamerv2.embodied import train_with_viz

    dreamer_dir = dm2.Path(__file__).parent
    model_configs = yaml.YAML(typ="safe").load((dreamer_dir / "dreamerv2.yaml").read())

    parsed, other = dm2.Flags(task=["carla_message"], actor_id=0, actors=0).parse_known(argv)

    config = dm2.Config({"dreamerv2": model_configs["defaults"]})
    config = config.update({"dreamerv2": model_configs["small"]})

    for name in parsed.task:
        print("Using task: ", name)
        env_config = load_task_configs(name)
        config = config.update(env_config)

    config = dm2.Flags(config).parse(other)

    config = config.update(logdir=str(dm2.Path(config.dreamerv2.logdir)))
    args = dm2.Config(
        logdir=config.dreamerv2.logdir,
        task_behavior=config.dreamerv2.task_behavior,
        skill_shape=config.dreamerv2.skill_shape,
        require_carry=config.env.require_carry,
        **config.dreamerv2.train,
    )
    args = args.update(expl_until=args.expl_until // config.dreamerv2.env.repeat)
    print(config)

    logdir = dm2.Path(config.dreamerv2.logdir)
    step = dm2.Counter()
    cleanup = []

    if config.dreamerv2.run == "acting":
        actordir = logdir / f"actor{parsed.actor_id}"
        logger = dm2.Logger(
            step,
            [
                dm2.logger.TerminalOutput(config.dreamerv2.filter),
                dm2.logger.JSONLOutput(actordir, "metrics.jsonl"),
                dm2.logger.TensorBoardOutput(actordir),
            ],
            multiplier=config.dreamerv2.env.repeat * parsed.actors,
        )
    else:
        logger = dm2.Logger(
            step,
            [
                dm2.logger.TerminalOutput(config.dreamerv2.filter),
                dm2.logger.JSONLOutput(logdir, "metrics.jsonl"),
                dm2.logger.TensorBoardOutput(logdir),
            ],
            multiplier=config.dreamerv2.env.repeat,
        )

    chunk = config.dreamerv2.replay_chunk
    if config.dreamerv2.replay == "fixed":

        def make_replay(name, capacity):
            directory = logdir / name
            store = dm2.replay.CkptRAMStore(directory, capacity, parallel=True)
            cleanup.append(store)
            return dm2.replay.FixedLength(store, chunk, **config.dreamerv2.replay_fixed)

    elif config.dreamerv2.replay == "consec":

        def make_replay(name, capacity):
            directory = logdir / name
            store = dm2.replay.CkptRAMStore(directory, capacity, parallel=True)
            cleanup.append(store)
            return dm2.replay.Consecutive(store, chunk, **config.dreamerv2.replay_consec)

    elif config.dreamerv2.replay == "prio":

        def make_replay(name, capacity):
            directory = logdir / name
            store = dm2.replay.CkptRAMStore(directory, capacity, parallel=True)
            cleanup.append(store)
            return dm2.replay.Prioritized(store, chunk, **config.dreamerv2.replay_prio)

    else:
        raise NotImplementedError(config.dreamerv2.replay)

    try:
        config = config.update({"dreamerv2.env.seed": hash((config.dreamerv2.seed, parsed.actor_id))})
        env = dm2.envs.load_env(config.env.name, mode="train", logdir=logdir, config=config.env)
        agent = agnt.Agent(env.obs_space, env.act_space, step, config.dreamerv2)
        save_configs(config, args.logdir)
        if config.dreamerv2.run == "train":
            replay = make_replay("episodes", config.dreamerv2.replay_size)
            dm2.run.train(agent, env, replay, logger, args)
        elif config.dreamerv2.run == "train_with_viz":
            if config.dreamerv2.eval_dir:
                assert not config.dreamerv2.train.eval_fill
                eval_replay = make_replay(config.dreamerv2.eval_dir, config.dreamerv2.replay_size // 10)
            else:
                assert config.dreamerv2.train.eval_fill
                eval_replay = make_replay("eval_episodes", config.dreamerv2.replay_size // 10)
            replay = make_replay("episodes", config.dreamerv2.replay_size)
            train_with_viz.train_with_viz(agent, env, replay, eval_replay, logger, args)
        elif config.dreamerv2.run == "learning":
            assert config.dreamerv2.replay.sync
            env.close()
            replay = make_replay("episodes", config.dreamerv2.replay_size)
            eval_replay = make_replay(config.dreamerv2.eval_dir, config.dreamerv2.replay_size // 10) if config.dreamerv2.eval_dir else replay
            dm2.run.learning(agent, replay, eval_replay, logger, args)
        elif config.dreamerv2.run == "acting":
            replay = make_replay("episodes", args.train_fill)
            dm2.run.acting(agent, env, replay, logger, actordir, args)
        else:
            raise NotImplementedError(config.dreamerv2.run)
    finally:
        for obj in cleanup:
            obj.close()


if __name__ == "__main__":
    main()
