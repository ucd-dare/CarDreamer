import pickle
from abc import abstractmethod
from collections import defaultdict, deque
from functools import partial as bind

import embodied
import numpy as np
import tensorflow as tf


class BasePrioritizedReverb:
    def __init__(self, length, capacity=None, directory=None, chunks=None, flush=100, hyper=None):
        del chunks
        import reverb

        self.length = length
        self.capacity = capacity
        self.directory = directory and embodied.Path(directory)
        self.checkpointer = None
        self.server = None
        self.client = None
        self.writers = None
        self.counters = None
        self.signature = None
        self.flush = flush

        self.hyper = hyper
        self.should_track_visit_counts = False

        # Constants
        self.priority_scalar = 10.0  # Used to scale all priorities. Avoids reverb precision issue.
        self.maximum_attempts_to_find_key = 10000
        max_steps = int(self.capacity * 2)

        self.step_to_keyA = np.zeros((max_steps,), dtype=np.uint32)
        self.step_to_keyB = np.zeros((max_steps,), dtype=np.uint32)
        self.visit_count = np.zeros((max_steps,), dtype=np.uint32)

        self.env_step_count = defaultdict(int)
        self.queue = deque(maxlen=2 * flush)

        if self.directory:
            self.directory.mkdirs()
            path = str(self.directory)
            try:
                self.checkpointer = reverb.checkpointers.DefaultCheckpointer(path)
            except AttributeError:
                self.checkpointer = reverb.checkpointers.RecordIOCheckpointer(path)
            self.sigpath = self.directory.parent / (self.directory.name + "_sig.pkl")
        if self.directory and self.sigpath.exists():
            with self.sigpath.open("rb") as file:
                self.signature = pickle.load(file)
            self._create_server()

    def _create_server(self):
        import reverb
        import tensorflow as tf

        self.server = reverb.Server(
            tables=[
                reverb.Table(
                    name="table",
                    sampler=reverb.selectors.Prioritized(1.0),
                    remover=reverb.selectors.Fifo(),
                    max_size=int(self.capacity),
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                    signature={key: tf.TensorSpec(shape, dtype) for key, (shape, dtype) in self.signature.items()},
                )
            ],
            port=None,
            checkpointer=self.checkpointer,
        )
        self.client = reverb.Client(f"localhost:{self.server.port}")
        self.writers = defaultdict(bind(self.client.trajectory_writer, self.length))
        self.counters = defaultdict(int)

    def __len__(self):
        if not self.client:
            return 0
        return self.client.server_info()["table"].current_size

    @property
    def stats(self):
        return {"size": len(self)}

    def add(self, step, worker=0):
        step = {k: v for k, v in step.items() if not k.startswith("log_")}
        step = {k: embodied.convert(v) for k, v in step.items()}
        step["id"] = np.asarray(embodied.uuid(step.get("id")))
        step["env_step"] = np.asarray(self.env_step_count[worker])
        step["worker"] = np.asarray(worker)
        if not self.server:
            self.signature = {k: ((self.length, *v.shape), v.dtype) for k, v in step.items()}
            self._create_server()

        step = {k: v for k, v in step.items() if not k.startswith("log_")}
        writer = self.writers[worker]
        self.queue.append(step)

        if (self.env_step_count[worker] + 1) < self.length:
            writer.append(self.queue.popleft())

        else:
            self.counters[worker] += 1
            if self.counters[worker] >= self.flush:
                for i in range(self.flush):
                    writer.append(self.queue.popleft())
                    seq = {k: v[-self.length :] for k, v in writer.history.items()}
                    writer.create_item(
                        "table",
                        priority=self.hyper["key_find_priority"],
                        trajectory=seq,
                    )
                self.counters[worker] = 0
                writer.flush()
                self._find_keys_up_to_step(step["env_step"])

        self.env_step_count[worker] += 1

    def _find_keys_up_to_step(self, fill_to_step):
        """Find the key for all steps just created in the table so that we can set their priorities later.
        The keys are likely to be sampled because they are given key_find_priority initially. This is set to the
        initial_priority after the keys are found."""

        import reverb

        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=f"localhost:{self.server.port}",
            table="table",
            max_in_flight_samples_per_worker=10,
        )

        found_so_far = np.zeros((int(self.flush),), dtype=np.uint8)
        fill_start_step = fill_to_step - self.flush + 1

        priorities_to_set = {}
        attempts = 0

        for sample in dataset:
            seq = sample.data
            step_sampled = int(seq["env_step"][-1])
            if step_sampled >= fill_start_step:
                key = sample.info.key
                (
                    self.step_to_keyA[step_sampled],
                    self.step_to_keyB[step_sampled],
                ) = self._split_key(key)
                priorities_to_set[int(key)] = self.hyper["initial_priority"] / self.priority_scalar
                found_so_far[step_sampled - fill_start_step] = 1

            if np.all(found_so_far):
                break

            attempts += 1
            if attempts > self.maximum_attempts_to_find_key:
                raise Exception(
                    f"dreamerv3/embodied/replay/reverb.py: _fill_step_to_key -> "
                    f"did not find env_step in {self.maximum_attempts_to_find_key} attempts"
                )

        self.client.mutate_priorities("table", priorities_to_set)

    def dataset(self):
        import reverb

        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=f"localhost:{self.server.port}",
            table="table",
            max_in_flight_samples_per_worker=1,
            num_workers_per_iterator=1,
            max_samples_per_stream=1,
        )
        for sample in dataset:
            seq = sample.data
            seq = {k: embodied.convert(v) for k, v in seq.items()}
            seq["keyA"], seq["keyB"] = self._split_key(sample.info.key)
            seq["key"] = (seq["keyA"], seq["keyB"])
            seq["probability"] = sample.info.probability
            seq["priority"] = sample.info.priority
            seq["times_sampled"] = sample.info.times_sampled

            if "is_first" in seq:
                seq["is_first"] = np.array(seq["is_first"])
                seq["is_first"][0] = True

            yield seq

    def _split_key(self, key):
        """Split the uint64 key into two 32 bit ints"""
        keyA_tf = key // tf.constant(2**32, dtype=tf.uint64)
        keyB_tf = key % tf.constant(2**32, dtype=tf.uint64)
        return np.uint32(keyA_tf), np.uint32(keyB_tf)

    def _combine_key(self, keyA, keyB) -> tf.uint64:
        """Combine the two 32bit ints into a single 64bit int"""
        keyA_tf = tf.convert_to_tensor(keyA, dtype=tf.uint64)
        keyB_tf = tf.convert_to_tensor(keyB, dtype=tf.uint64)

        return keyA_tf * tf.constant(2**32, dtype=tf.uint64) + keyB_tf

    def update_visit_count(self, env_steps):
        flat_env_steps = env_steps.flatten()
        self.visit_count[flat_env_steps] += 1

    @abstractmethod
    def prioritize(self, key, env_steps, losses, td_error):
        pass

    def save(self, wait=False):
        for writer in self.writers.values():
            writer.flush()
        with self.sigpath.open("wb") as file:
            file.write(pickle.dumps(self.signature))
        if self.directory:
            self.client.checkpoint()

    def load(self, data=None):
        pass
