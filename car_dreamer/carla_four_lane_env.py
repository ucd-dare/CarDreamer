import carla
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedPathPlanner


class CarlaFourLaneEnv(CarlaWptEnv):
    """
    This task generates random routes within a four-lane system.

    **Provided Tasks**: ``carla_four_lane``

    Available config parameters:

    * ``lane_start_points``: Possible starting points of the ego vehicle in ``[[x, y, z], ...]``
    * ``lane_end_points``: Possible ending points of the ego vehicle in ``[[x, y, z], ...]``
    * ``num_vehicles``: Number of vehicles to spawn in the environment

    """

    def on_reset(self) -> None:
        self.ego_src = self._config.lane_start_points[np.random.randint(0, len(self._config.lane_start_points) - 1)]
        ego_transform = carla.Transform(
            carla.Location(x=self.ego_src[0], y=self.ego_src[1], z=self.ego_src[2]),
            carla.Rotation(yaw=-90),
        )
        self.ego = self._world.spawn_actor(transform=ego_transform)
        self._world.spawn_auto_actors(self._config.num_vehicles)
        self.ego_path = self.generate_vehicle_path(self.ego_src, self._config.lane_end_points[0][1])
        self.ego_dest = self.ego_path[-1]
        self.ego_planner = FixedPathPlanner(vehicle=self.ego, vehicle_path=self.ego_path)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

    def generate_vehicle_path(self, start_point, end_y, min_lane_keeping=15, max_lane_keeping=30):
        lane_start_points = self._config.lane_start_points

        waypoints = [start_point]
        current_lane = None
        for i in range(len(lane_start_points)):
            if lane_start_points[i][0] == start_point[0]:
                current_lane = i
                break
        assert current_lane is not None, "Start point not in any lane"

        current_y = start_point[1]

        while current_y > end_y:
            # Random lane keeping distance
            distance = min(
                np.abs(end_y - current_y),
                np.random.uniform(min_lane_keeping, max_lane_keeping),
            )
            current_x = lane_start_points[current_lane][0]
            current_y -= distance
            waypoints.append([current_x, current_y, 0.1])
            if current_y <= end_y:
                break
            # Decide whether to change lane
            lane_changing_options = [current_lane]
            if current_lane < len(lane_start_points) - 1:
                lane_changing_options.append(current_lane + 1)
            if current_lane > 0:
                lane_changing_options.append(current_lane - 1)
            current_lane = np.random.choice(lane_changing_options)
            current_x = lane_start_points[current_lane][0]
            current_y -= 8.0
            waypoints.append([current_x, current_y, 0.1])
        for i, p in enumerate(waypoints):
            p_calib = self._world.carla_map.get_waypoint(carla.Location(x=p[0], y=p[1], z=p[2]))
            waypoints[i] = [
                p_calib.transform.location.x,
                p_calib.transform.location.y,
                p_calib.transform.location.z,
            ]
        return waypoints
