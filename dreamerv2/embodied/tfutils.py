import inspect
import logging
import os
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel("ERROR")

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import mixed_precision as prec
from tensorflow.python.distribute import values
from tensorflow_probability import distributions as tfd


def sg(x):
    return tf.stop_gradient(x)


for base in (tf.Tensor, tf.Variable, values.PerReplica):
    base.mean = tf.math.reduce_mean
    base.std = tf.math.reduce_std
    base.var = tf.math.reduce_variance
    base.sum = tf.math.reduce_sum
    base.prod = tf.math.reduce_prod
    base.any = tf.math.reduce_any
    base.all = tf.math.reduce_all
    base.min = tf.math.reduce_min
    base.max = tf.math.reduce_max
    base.abs = tf.math.abs
    base.logsumexp = tf.math.reduce_logsumexp
    base.transpose = tf.transpose
    base.reshape = tf.reshape
    base.astype = tf.cast
    base.flatten = lambda x: tf.reshape(x, [-1])


def tensor(value):
    if isinstance(value, values.PerReplica):
        return value
    return tf.convert_to_tensor(value)


tf.tensor = tensor


def shuffle(tensor, axis):
    perm = list(range(len(tensor.shape)))
    perm.pop(axis)
    perm.insert(0, axis)
    tensor = tensor.transpose(perm)
    tensor = tf.random.shuffle(tensor)
    tensor = tensor.transpose(perm)
    return tensor


def scan(fn, inputs, start, static=True, reverse=False, axis=0):
    assert axis in (0, 1), axis
    if axis == 1:
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        inputs = tf.nest.map_structure(swap, inputs)
    if not static:
        return tf.scan(fn, inputs, start, reverse=reverse)
    last = start
    outputs = [[] for _ in tf.nest.flatten(start)]
    indices = range(tf.nest.flatten(inputs)[0].shape[0])
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = tf.nest.map_structure(lambda x: x[index], inputs)
        last = fn(last, inp)
        tf.nest.assert_same_structure(last, start)
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [tf.stack(x, axis) for x in outputs]
    return tf.nest.pack_sequence_as(start, outputs)


def symlog(x):
    return tf.sign(x) * tf.math.log(1 + tf.abs(x))


def symexp(x):
    return tf.sign(x) * (tf.math.exp(tf.abs(x)) - 1)


def action_noise(action, amount, act_space):
    if amount == 0:
        return action
    amount = tf.cast(amount, action.dtype)
    if act_space.discrete:
        probs = amount / action.shape[-1] + (1 - amount) * action
        return OneHotDist(probs=probs).sample()
    else:
        return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * tf.ones_like(reward)
    dims = list(range(reward.shape.ndims))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = tf.transpose(reward, dims)
        value = tf.transpose(value, dims)
        pcont = tf.transpose(pcont, dims)
    if bootstrap is None:
        bootstrap = tf.zeros_like(value[-1])
    next_values = tf.concat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = scan(
        lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
        (inputs, pcont),
        bootstrap,
        static=True,
        reverse=True,
    )
    if axis != 0:
        returns = tf.transpose(returns, dims)
    return returns


class Module(tf.Module):
    def save(self):
        values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
        amount = len(tf.nest.flatten(values))
        count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
        print(f"Saving module with {amount} tensors and {count} parameters.")
        return values

    def load(self, values):
        amount = len(tf.nest.flatten(values))
        count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
        print(f"Loading module with {amount} tensors and {count} parameters.")
        tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

    def get(self, name, ctor, *args, **kwargs):
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            if "name" in inspect.signature(ctor).parameters:
                kwargs["name"] = name
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]


class Optimizer(Module):
    def __init__(
        self,
        name,
        lr,
        opt="adam",
        eps=1e-5,
        clip=0.0,
        warmup=0,
        wd=0.0,
        wd_pattern="kernel",
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._clip = clip
        self._warmup = warmup
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._updates = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._lr = lr
        if warmup:
            self._lr = lambda: lr * tf.clip_by_value(self._updates.astype(tf.float32) / warmup, 0.0, 1.0)
        self._opt = {
            "adam": lambda: tf.optimizers.Adam(self._lr, epsilon=eps),
            "sgd": lambda: tf.optimizers.SGD(self._lr),
            "momentum": lambda: tf.optimizers.SGD(self._lr, 0.9),
        }[opt]()
        self._scaling = prec.global_policy().compute_dtype == tf.float16
        if self._scaling:
            self._grad_scale = tf.Variable(1e4, trainable=False, dtype=tf.float32)
            self._fine_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._once = True

    @property
    def variables(self):
        return self._opt.variables()

    def __call__(self, tape, loss, modules):
        assert loss.dtype is tf.float32, (self._name, loss.dtype)
        assert len(loss.shape) == 0, (self._name, loss.shape)
        metrics = {}

        # Find variables.
        modules = modules if hasattr(modules, "__len__") else (modules,)
        varibs = tf.nest.flatten([module.trainable_variables for module in modules])
        count = sum(int(np.prod(x.shape)) for x in varibs)
        self._once and print(f"Found {count} {self._name} parameters.")

        # Check loss.
        tf.debugging.check_numerics(loss, self._name + "_loss")
        metrics[f"{self._name}_loss"] = loss

        # Compute scaled gradient.
        if self._scaling:
            with tape:
                loss = self._grad_scale * loss
        grads = tape.gradient(loss, varibs)
        for var, grad in zip(varibs, grads):
            if grad is None:
                raise RuntimeError(f"{self._name} optimizer found no gradient for {var.name}.")

        # Distributed sync.
        if tf.distribute.has_strategy():
            context = tf.distribute.get_replica_context()
            grads = context.all_reduce("mean", grads)

        if self._scaling:
            grads = tf.nest.map_structure(lambda x: x / self._grad_scale, grads)
            overflow = ~tf.reduce_all([tf.math.is_finite(x).all() for x in tf.nest.flatten(grads)])
            metrics[f"{self._name}_grad_scale"] = self._grad_scale
            metrics[f"{self._name}_grad_overflow"] = overflow.astype(tf.float32)
            keep = ~overflow & (self._fine_steps < 1000)
            incr = ~overflow & (self._fine_steps >= 1000)
            decr = overflow
            self._fine_steps.assign(keep.astype(tf.int64) * (self._fine_steps + 1))
            self._grad_scale.assign(
                tf.clip_by_value(
                    keep.astype(tf.float32) * self._grad_scale
                    + incr.astype(tf.float32) * self._grad_scale * 2
                    + decr.astype(tf.float32) * self._grad_scale / 2,
                    1e-4,
                    1e4,
                )
            )
        else:
            overflow = False

        # Gradient clipping.
        norm = tf.linalg.global_norm(grads)
        if self._clip:
            grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
        if self._scaling:
            norm = tf.where(tf.math.is_finite(norm), norm, np.nan)
        else:
            tf.debugging.check_numerics(norm, self._name + "_norm")
        metrics[f"{self._name}_grad_norm"] = norm

        # Weight decay.
        if self._wd:
            if ~overflow:
                self._apply_weight_decay(varibs)

        # Apply gradients.
        if ~overflow:
            self._opt.apply_gradients(zip(grads, varibs), experimental_aggregate_gradients=False)
            self._updates.assign_add(1)
        metrics[f"{self._name}_grad_steps"] = self._updates

        self._once = False
        return metrics

    def _apply_weight_decay(self, varibs):
        lr = self._lr() if callable(self._lr) else self._lr
        log = (self._wd_pattern != r".*") and self._once
        if log:
            print(f"Optimizer applied weight decay to {self._name} variables:")
        included, excluded = [], []
        for var in sorted(varibs, key=lambda x: x.name):
            if re.search(self._wd_pattern, self._name + "/" + var.name):
                var.assign((1 - self._wd * lr) * var)
                included.append(var.name)
            else:
                excluded.append(var.name)
        if log:
            for name in included:
                print(f"[x] {name}")
            for name in excluded:
                print(f"[ ] {name}")
            print("")


class MSEDist(tfd.Distribution):
    def __init__(self, pred, dims, agg="sum"):
        super().__init__(pred.dtype, tfd.FULLY_REPARAMETERIZED, False, True)
        self.pred = pred
        self._dims = dims
        self._axes = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return {"pred": tfp.util.ParameterProperties()}

    def _batch_shape(self):
        return self.pred.shape[: len(self.pred.shape) - self._dims]

    def _event_shape(self):
        return self.pred.shape[len(self.pred.shape) - self._dims :]

    def mean(self):
        return self.pred

    def mode(self):
        return self.pred

    def sample(self, sample_shape=(), seed=None):
        return tf.broadcast_to(self.pred, sample_shape + self.pred.shape)

    def log_prob(self, value):
        assert len(self.pred.shape) == len(value.shape), (self.pred.shape, value.shape)
        distance = (self.pred - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._axes)
        elif self._agg == "sum":
            loss = distance.sum(self._axes)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class CosineDist(tfd.Distribution):
    def __init__(self, pred):
        super().__init__(pred.dtype, tfd.FULLY_REPARAMETERIZED, False, True)
        self.pred = tf.nn.l2_normalize(pred, -1)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return {"pred": tfp.util.ParameterProperties()}

    def _batch_shape(self):
        return self.pred.shape[:-1]

    def _event_shape(self):
        return self.pred.shape[-1:]

    def mean(self):
        return self.pred

    def mode(self):
        return self.pred

    def sample(self, sample_shape=(), seed=None):
        return tf.broadcast_to(self.pred, sample_shape + self.pred.shape)

    def log_prob(self, value):
        assert len(self.pred.shape) == len(value.shape), (self.pred.shape, value.shape)
        return tf.einsum("...i,...i->...", self.pred, value)


class DirDist(tfd.MultivariateNormalDiag):
    def __init__(self, mean, std):
        self.mean = tf.nn.l2_normalize(mean.astype(tf.float32), -1)
        self.std = std.astype(tf.float32)
        super().__init__(self.mean, self.std)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return {k: tfp.util.ParameterProperties() for k in ("mean", "std")}

    def _batch_shape(self):
        return self.mean.shape[:-1]

    def _event_shape(self):
        return self.mean.shape[-1:]

    def sample(self, sample_shape=(), seed=None):
        sample = super().sample(sample_shape, seed)
        sample = tf.nn.l2_normalize(sample, -1)
        return sample

    def log_prob(self, value):
        value = tf.nn.l2_normalize(value.astype(tf.float32), -1)
        return super().log_prob(value)


class SymlogDist:
    def __init__(self, mode, dims, agg="sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims :]

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - symlog(value)) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class OneHotDist(tfd.OneHotCategorical):
    def __init__(self, logits=None, probs=None, dtype=tf.float32):
        super().__init__(logits, probs, dtype)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return super()._parameter_properties(dtype)

    def sample(self, sample_shape=(), seed=None):
        sample = sg(super().sample(sample_shape, seed))
        probs = self._pad(super().probs_parameter(), sample.shape)
        return sg(sample) + (probs - sg(probs)).astype(sample.dtype)

    def _pad(self, tensor, shape):
        while len(tensor.shape) < len(shape):
            tensor = tensor[None]
        return tensor


class TwoHotDist(tfd.Distribution):
    def __init__(self, dist1, dist2):
        super().__init__(
            dtype=tf.int32,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=True,
        )
        self.dist1 = dist1
        self.dist2 = dist2

    def _batch_shape_tensor(self):
        return tf.broadcast_static_shape(self.dist1.batch_shape_tensor(), self.dist2.batch_shape_tensor())

    def _event_shape_tensor(self):
        return [self.dist1.event_shape_tensor()[0] + self.dist2.event_shape_tensor()[0]]

    def sample(self, sample_shape=(), seed=None):
        samples1 = self.dist1.sample(sample_shape, seed=seed)
        samples2 = self.dist2.sample(sample_shape, seed=seed)
        return tf.concat([samples1, samples2], axis=-1)

    def log_prob(self, value):
        split1 = self.dist1.event_shape_tensor()[0]
        split2 = self.dist2.event_shape_tensor()[0]
        value1, value2 = value[..., :split1], value[..., split1 : split1 + split2]
        log_prob1 = self.dist1.log_prob(value1)
        log_prob2 = self.dist2.log_prob(value2)
        return log_prob1 + log_prob2

    def entropy(self):
        return self.dist1.entropy() + self.dist2.entropy()

    def mode(self):
        """Compute the mode of the concatenated one-hot distributions."""
        mode1 = self.dist1.mode()
        mode2 = self.dist2.mode()
        return tf.concat([mode1, mode2], axis=-1)


def video_grid(video):
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


def balance_stats(dist, target, thres):
    # Values are NaN when there are no positives or negatives in the current
    # batch, which means they will be ignored when aggregating metrics via
    # np.nanmean() later, as they should.
    pos = (target.astype(tf.float32) > thres).astype(tf.float32)
    neg = (target.astype(tf.float32) <= thres).astype(tf.float32)
    pred = (dist.mean().astype(tf.float32) > thres).astype(tf.float32)
    loss = -dist.log_prob(target)
    return dict(
        pos_loss=(loss * pos).sum() / pos.sum(),
        neg_loss=(loss * neg).sum() / neg.sum(),
        pos_acc=(pred * pos).sum() / pos.sum(),
        neg_acc=((1 - pred) * neg).sum() / neg.sum(),
        rate=pos.mean(),
        avg=target.astype(tf.float32).mean(),
        pred=dist.mean().astype(tf.float32).mean(),
    )


class AutoAdapt(Module):
    def __init__(self, shape, impl, scale, target, min, max, vel=0.1, thres=0.1, inverse=False):
        self._shape = shape
        self._impl = impl
        self._target = target
        self._min = min
        self._max = max
        self._vel = vel
        self._inverse = inverse
        self._thres = thres
        if self._impl == "fixed":
            self._scale = tf.tensor(scale)
        elif self._impl == "mult":
            self._scale = tf.Variable(tf.ones(shape, tf.float32), trainable=False)
        elif self._impl == "prop":
            self._scale = tf.Variable(tf.ones(shape, tf.float32), trainable=False)
        else:
            raise NotImplementedError(self._impl)

    def __call__(self, reg, update=True):
        update and self.update(reg)
        scale = self.scale()
        loss = scale * (-reg if self._inverse else reg)
        metrics = {
            "mean": reg.mean(),
            "std": reg.std(),
            "scale_mean": scale.mean(),
            "scale_std": scale.std(),
        }
        return loss, metrics

    def scale(self):
        if self._impl == "fixed":
            scale = self._scale
        elif self._impl == "mult":
            scale = self._scale
        elif self._impl == "prop":
            scale = self._scale
        else:
            raise NotImplementedError(self._impl)
        return sg(tf.tensor(scale))

    def update(self, reg):
        avg = reg.mean(list(range(len(reg.shape) - len(self._shape))))
        if self._impl == "fixed":
            pass
        elif self._impl == "mult":
            below = avg < (1 / (1 + self._thres)) * self._target
            above = avg > (1 + self._thres) * self._target
            if self._inverse:
                below, above = above, below
            inside = ~below & ~above
            adjusted = (
                above.astype(tf.float32) * self._scale * (1 + self._vel)
                + below.astype(tf.float32) * self._scale / (1 + self._vel)
                + inside.astype(tf.float32) * self._scale
            )
            self._scale.assign(tf.clip_by_value(adjusted, self._min, self._max))
        elif self._impl == "prop":
            direction = avg - self._target
            if self._inverse:
                direction = -direction
            self._scale.assign(tf.clip_by_value(self._scale + self._vel * direction, self._min, self._max))
        else:
            raise NotImplementedError(self._impl)


class Normalize:
    def __init__(self, impl="mean_std", decay=0.99, max=1e8, vareps=0.0, stdeps=0.0):
        self._impl = impl
        self._decay = decay
        self._max = max
        self._stdeps = stdeps
        self._vareps = vareps
        self._mean = tf.Variable(0.0, trainable=False, dtype=tf.float64)
        self._sqrs = tf.Variable(0.0, trainable=False, dtype=tf.float64)
        self._step = tf.Variable(0, trainable=False, dtype=tf.int64)

    def __call__(self, values, update=True):
        update and self.update(values)
        return self.transform(values)

    def update(self, values):
        x = values.astype(tf.float64)
        m = self._decay
        self._step.assign_add(1)
        self._mean.assign(m * self._mean + (1 - m) * x.mean())
        self._sqrs.assign(m * self._sqrs + (1 - m) * (x**2).mean())

    def transform(self, values):
        correction = 1 - self._decay ** self._step.astype(tf.float64)
        mean = self._mean / correction
        var = (self._sqrs / correction) - mean**2
        if self._max > 0.0:
            scale = tf.math.rsqrt(tf.maximum(var, 1 / self._max**2 + self._vareps) + self._stdeps)
        else:
            scale = tf.math.rsqrt(var + self._vareps) + self._stdeps
        if self._impl == "off":
            pass
        elif self._impl == "mean_std":
            values -= mean.astype(values.dtype)
            values *= scale.astype(values.dtype)
        elif self._impl == "std":
            values *= scale.astype(values.dtype)
        else:
            raise NotImplementedError(self._impl)
        return values
