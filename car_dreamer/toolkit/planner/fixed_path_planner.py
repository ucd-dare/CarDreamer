from typing import List, Tuple

import carla
import numpy as np

from .agents.navigation.global_route_planner import GlobalRoutePlanner
from .base_planner import BasePlanner


class FixedPathPlanner(BasePlanner):
    """
    Route planner for a given vehicle and a given route
    """

    def __init__(
        self,
        vehicle: carla.Actor,
        vehicle_path: List[Tuple[float, float, float]],
        use_road_waypoints: List[bool] = None,
        sampling_radius=0.8,
    ):
        super().__init__(vehicle)
        self._vehicle_path = vehicle_path
        self._use_road_waypoints = use_road_waypoints
        self._grp = GlobalRoutePlanner(self._map, sampling_resolution=sampling_radius)
        self._sampling_radius = sampling_radius

    def init_route(self):
        for i, start in enumerate(self._vehicle_path[:-1]):
            end = self._vehicle_path[i + 1]
            if self._use_road_waypoints is not None and self._use_road_waypoints[i]:
                segment_waypoints = self._grp.trace_route(
                    carla.Location(x=start[0], y=start[1], z=start[2]),
                    carla.Location(x=end[0], y=end[1], z=end[2]),
                )
                if i > 0:
                    segment_waypoints = segment_waypoints[1:]
                for waypoint in segment_waypoints:
                    self.add_waypoint(waypoint[0])
            else:
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                yaw = np.arctan2(dy, dx) * 180 / np.pi
                dist = np.sqrt(dx**2 + dy**2)
                sample_num = max(2, int(dist / self._sampling_radius))
                for theta in np.linspace(0, 1, sample_num):
                    x = start[0] + theta * dx
                    y = start[1] + theta * dy
                    self.add_waypoint((x, y, yaw))

    def extend_route(self):
        pass
