import carla

from .agents.navigation.global_route_planner import GlobalRoutePlanner
from .base_planner import BasePlanner


class FixedEndingPlanner(BasePlanner):
    """Route planner for a given vehicle and a given ending"""

    def __init__(self, vehicle: carla.Actor, end: carla.Location, sampling_radius=0.8):
        super().__init__(vehicle)
        self.end = end
        self._grp = GlobalRoutePlanner(self._map, sampling_resolution=sampling_radius)

    def init_route(self):
        global_route = self._grp.trace_route(self._vehicle.get_location(), self.end)
        for waypoint in global_route:
            self.add_waypoint(waypoint[0])
        assert self.get_waypoint_num() > 0, "Empty waypoint queue"

    def extend_route(self):
        pass
