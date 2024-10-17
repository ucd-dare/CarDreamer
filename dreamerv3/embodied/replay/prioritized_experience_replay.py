import numpy as np

from dreamerv3.embodied.replay.base_prioritized_reverb import BasePrioritizedReverb


class PrioritizedExperienceReplay(BasePrioritizedReverb):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_track_visit_counts = False

    @staticmethod
    def _calculate_priority_score(td_error, hyper):
        return np.power(np.abs(td_error) + hyper["epsilon"], hyper["alpha"])

    def prioritize(self, key, env_steps, losses, td_error):
        flat_steps = env_steps[:, 1:].flatten()
        flat_td = td_error.flatten()
        flat_priority = self._calculate_priority_score(flat_td, self.hyper) / self.priority_scalar
        flat_keys = self._combine_key(self.step_to_keyA[flat_steps], self.step_to_keyB[flat_steps])
        flat_updates = {int(k): p for k, p in zip(flat_keys, flat_priority)}
        self.client.mutate_priorities("table", flat_updates)
