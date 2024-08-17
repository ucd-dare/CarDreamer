import contextlib

import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from tensorflow.python.distribute import values

import dreamerv2 as dm2

from . import tfutils


class TFAgent(tfutils.Module, dm2.Agent):
    def __new__(subcls, obs_space, act_space, step, config):
        self = super().__new__(TFAgent)
        self.config = config.tf
        self.strategy = self._setup()
        self.agent = super().__new__(subcls)
        with self._strategy_scope():
            self.agent.__init__(obs_space, act_space, step, config)
        self._cache_fns = config.tf.jit and not self.strategy
        self._cached_fns = {}
        return self

    def dataset(self, generator):
        with self._strategy_scope():
            dataset = self.agent.dataset(generator)
        return dataset

    def policy(self, obs, state=None, mode="train"):
        obs = {k: v for k, v in obs.items() if not k.startswith("log_")}
        if state is None:
            state = self.agent.initial_policy_state(obs)
        fn = self.agent.policy
        if self._cache_fns:
            key = f"policy_{mode}"
            if key not in self._cached_fns:
                self._cached_fns[key] = fn.get_concrete_function(obs, state, mode)
            fn = self._cached_fns[key]
        act, state = fn(obs, state, mode)
        act = self._convert_outs(act)
        return act, state

    def train(self, data, state=None):
        data = self._convert_inps(data)
        if state is None:
            state = self._strategy_run(self.agent.initial_train_state, data)
        fn = self.agent.train
        if self._cache_fns:
            key = "train"
            if key not in self._cached_fns:
                self._cached_fns[key] = fn.get_concrete_function(data, state)
            fn = self._cached_fns[key]
        outs, state, metrics = self._strategy_run(fn, data, state)
        outs = self._convert_outs(outs)
        metrics = self._convert_mets(metrics)
        return outs, state, metrics

    def report(self, data):
        data = self._convert_inps(data)
        fn = self.agent.report
        if self._cache_fns:
            key = "report"
            if key not in self._cached_fns:
                self._cached_fns[key] = fn.get_concrete_function(data)
            fn = self._cached_fns[key]
        metrics = self._strategy_run(fn, data)
        metrics = self._convert_mets(metrics)
        return metrics

    @contextlib.contextmanager
    def _strategy_scope(self):
        if self.strategy:
            with self.strategy.scope():
                yield None
        else:
            yield None

    def _strategy_run(self, fn, *args, **kwargs):
        if self.strategy:
            return self.strategy.run(fn, args, kwargs)
        else:
            return fn(*args, **kwargs)

    def _convert_inps(self, value):
        if not self.strategy:
            return value
        if isinstance(value, (tuple, dict)):
            return tf.nest.map_structure(self._convert_inps, value)
        if isinstance(value, values.PerReplica):
            return value
        if isinstance(value, (tf.Tensor, tf.Variable)):
            return value
        return dm2.convert(value)

        # replicas = self.strategy.num_replicas_in_sync
        # value = tf.convert_to_tensor(value)
        # if len(value) < replicas:
        #   raise ValueError(
        #       f'Cannot split input dim {len(value)} into {replicas} replicas.')
        # elif len(value) == replicas:
        #   pass
        # elif len(value) % replicas == 0:
        #   splits = [len(value) // replicas] * replicas
        #   value = tf.split(value, splits, 0)
        # else:
        #   size = int(np.ceil(len(value) / replicas))
        #   splits = [size] * (replicas - 1) + [len(value) % size]
        #   assert sum(splits) == len(value), (splits, value)
        #   value = tf.split(value, splits, 0)
        # return self.strategy.experimental_distribute_values_from_function(
        #     lambda ctx: value[ctx.replica_id_in_sync_group])

        replicas = self.strategy.num_replicas_in_sync
        assert len(value) % replicas == 0, (len(value), replicas)
        value = tf.split(value, replicas, 0)
        return self.strategy.experimental_distribute_values_from_function(lambda ctx: value[ctx.replica_id_in_sync_group])

    def _convert_outs(self, value):
        if isinstance(value, (tuple, list, dict)):
            return tf.nest.map_structure(self._convert_outs, value)
        if isinstance(value, values.PerReplica):
            value = self.strategy.gather(value, axis=0)
        if hasattr(value, "numpy"):  # Tensor, Variable, MirroredVariable
            value = value.numpy()
        return value

    def _convert_mets(self, value):
        if isinstance(value, (tuple, list, dict)):
            return tf.nest.map_structure(self._convert_mets, value)
        if isinstance(value, values.PerReplica):
            value = value.values[0]  # Only use metrics from first replica.
        if hasattr(value, "numpy"):  # Tensor, Variable, MirroredVariable
            value = value.numpy()
        return value

    def _setup(self):
        assert self.config.precision in (16, 32)

        tf.config.run_functions_eagerly(not self.config.jit)
        if self.config.placement:
            tf.config.set_soft_device_placement(False)
        if self.config.debug_nans:
            tf.debugging.enable_check_numerics()

        tf.config.experimental.enable_tensor_float_32_execution(self.config.tensorfloat)

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if self.config.logical_gpus:
            conf = tf.config.LogicalDeviceConfiguration(memory_limit=1024)
            tf.config.set_logical_device_configuration(gpus[0], [conf] * self.config.logical_gpus)

        if self.config.platform == "cpu":
            return None

        elif self.config.platform == "gpu":
            assert len(gpus) >= 1, gpus
            if not self.config.logical_gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, self.config.growth)
            if self.config.precision == 16:
                prec.set_global_policy(prec.Policy("mixed_float16"))
            return None

        elif self.config.platform == "multi_gpu":
            assert len(gpus) >= 1, gpus
            if self.config.precision == 16:
                prec.set_global_policy(prec.Policy("mixed_float16"))
            return tf.distribute.MirroredStrategy()

        elif self.config.platform == "multi_worker":
            assert len(gpus) >= 1, gpus
            if self.config.precision == 16:
                prec.set_global_policy(prec.Policy("mixed_float16"))
            return tf.distribute.MultiWorkerMirroredStrategy()

        elif self.config.platform == "tpu":
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            return tf.distribute.TPUStrategy(resolver)

        else:
            raise NotImplementedError(self.config.platform)
