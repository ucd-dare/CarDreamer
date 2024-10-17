import numpy as np

from dreamerv3.embodied.replay import CuriousReplay


class CountBasedReplay(CuriousReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_track_visit_counts = True

    @staticmethod
    def _calculate_priority_score(model_loss, visit_count, hyper):
        return hyper["c"] * np.power(hyper["beta"], visit_count) + hyper["epsilon"]
