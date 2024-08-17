import carla
import numpy as np

from ..carla_manager import get_vehicle_orientation, get_vehicle_pos
from .base_planner import BasePlanner


class RandomPlanner(BasePlanner):
    """Generate random routes based on map topology"""

    def __init__(self, vehicle: carla.Actor, sampling_radius=0.8):
        super().__init__(vehicle)
        self._sampling_radius = sampling_radius

    def init_route(self):
        pos = get_vehicle_pos(self._vehicle)
        orientation = get_vehicle_orientation(self._vehicle)
        self.add_waypoint((pos[0], pos[1], orientation))

    def extend_route(self):
        while self.get_waypoint_num() < 100:
            self._compute_next_waypoint()

    def _compute_next_waypoint(self):
        waypoints = self.get_all_waypoints()
        last_waypoint = waypoints[-1]
        last_waypoint = self._map.get_waypoint(carla.Location(last_waypoint[0], last_waypoint[1], 0), project_to_road=True)
        next_waypoints = list(last_waypoint.next(self._sampling_radius))

        if len(next_waypoints) == 0:
            return
        else:
            next_waypoint = np.random.choice(next_waypoints)
            self.add_waypoint(next_waypoint)
