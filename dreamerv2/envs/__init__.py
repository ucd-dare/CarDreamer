import functools

import dreamerv2 as dm2


def load_env(task, amount=1, parallel="none", daemon=False, restart=False, seed=None, **kwargs):
    ctors = []
    for index in range(amount):
        ctor = functools.partial(load_single_env, task, **kwargs)
        if seed is not None:
            ctor = functools.partial(ctor, seed=hash((seed, index)) % (2**31 - 1))
        if parallel != "none":
            ctor = functools.partial(dm2.Parallel, ctor, parallel, daemon)
        if restart:
            ctor = functools.partial(dm2.wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return dm2.BatchEnv(envs, parallel=(parallel != "none"))


def load_single_env(task, config, **kwargs):
    import car_dreamer

    from . import gym

    env = gym.FromGym(task, obs_key="camera", config=config)

    for name, space in env.act_space.items():
        if name == "reset":
            continue
        if space.discrete:
            env = dm2.wrappers.OneHotAction(env, name)
        else:
            env = dm2.wrappers.NormalizeAction(env, name)
    env = dm2.wrappers.ExpandScalars(env)
    env = dm2.wrappers.InfoWrapper(env)
    return env


__all__ = [k for k, v in list(locals().items()) if type(v).__name__ in ("type", "function") and not k.startswith("_")]
