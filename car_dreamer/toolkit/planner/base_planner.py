from abc import ABC, abstractmethod
from collections import deque

import carla

from ..carla_manager import get_location_distance, get_vehicle_pos


class BasePlanner(ABC):
    """
    Base class for route planner.
    All route planners exposes a :py:meth:`run_step` method.
    To inherit, implement :py:meth:`init_route` and :py:meth:`extend_route`.
    """

    def __init__(self, vehicle: carla.Actor, max_waypoints=60, reach_threshold=0.5):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._waypoints_queue = deque()
        self._initialized = False

        # The maximum waypoints returned by the planner
        self._max_waypoints = max_waypoints
        # The distance below which the waypoint is considered as reached
        self._reach_threshold = reach_threshold
        self._vehicle_location = get_vehicle_pos(self._vehicle)
        self._prev_location = self._vehicle_location

    @abstractmethod
    def init_route(self):
        """
        Initialize waypoints queue.
        It is called only once at the first step.
        """
        pass

    @abstractmethod
    def extend_route(self):
        """
        Extend waypoints queue.
        It is called at every step.
        """
        pass

    def get_all_waypoints(self):
        """
        Get all waypoints in the queue.

        :return: list of waypoints ``(x, y, yaw)``
        """
        return self._waypoints_queue

    def add_waypoint(self, waypoint):
        """
        Add waypoints to the queue.

        :param waypoints: waypoint ``(x, y, yaw)`` or carla.Waypoint
        """
        if isinstance(waypoint, carla.Waypoint):
            waypoint = self.from_carla_waypoint(waypoint)
        self._waypoints_queue.append(waypoint)

    def pop_waypoint(self):
        """
        Pop a waypoint from the queue.
        """
        return self._waypoints_queue.popleft()

    def clear_waypoints(self):
        """
        Clear waypoints queue.
        """
        self._waypoints_queue.clear()

    def get_waypoint_num(self):
        """
        Get the number of waypoints in the queue.

        :return: int, number of waypoints in the queue
        """
        return len(self._waypoints_queue)

    def from_carla_waypoint(self, waypoint):
        """
        Convert a carla.Waypoint to a waypoint ``(x, y, yaw)``.

        :param waypoint: carla.Waypoint

        :return: tuple ``(x, y, yaw)``
        """
        return (
            waypoint.transform.location.x,
            waypoint.transform.location.y,
            waypoint.transform.rotation.yaw,
        )

    def run_step(self):
        """
        Run one step of the route planner, extending the route and removing expired waypoints.

        :return: tuple(list of waypoints ``(x, y, yaw)``, additional stats)
        """
        if not self._initialized:
            self.init_route()
            self._initialized = True

        self.extend_route()
        self._vehicle_location = get_vehicle_pos(self._vehicle)
        planner_stats = self._update_waypoints_queue()
        waypoints = self._get_waypoints()
        self._prev_location = self._vehicle_location
        return waypoints, planner_stats

    def _get_waypoints(self):
        waypoints = []
        for i, waypoint in enumerate(self._waypoints_queue):
            if i >= self._max_waypoints:
                break
            waypoints.append(waypoint)
        return waypoints

    def _update_waypoints_queue(self):
        num_completed = 0
        num_obsolete = 0
        num_to_delete = 0
        min_distance = 100
        for i, waypoint in enumerate(self._waypoints_queue):
            dist = get_location_distance(self._vehicle_location, waypoint)
            if dist < self._reach_threshold:
                num_completed += 1
                num_to_delete = i + 1
                num_obsolete = i + 1
            elif dist < min_distance:
                min_distance = dist
                num_to_delete = i
                num_obsolete = i
        for _ in range(num_to_delete):
            self.pop_waypoint()
        num_obsolete -= num_completed
        planner_stats = dict(
            num_completed=num_completed,
            num_obsolete=num_obsolete,
            travel_distance=get_location_distance(self._prev_location, self._vehicle_location),
        )
        return planner_stats
