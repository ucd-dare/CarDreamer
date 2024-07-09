import carla
import math
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_vehicle_pos

class CarlaFollowEnv(CarlaWptEnv):
    def on_reset(self) -> None:
        self.nonego_spawn_point = self._config.nonego_spawn_point
        nonego_transform = carla.Transform(carla.Location(*self.nonego_spawn_point[:3]), carla.Rotation(*self.nonego_spawn_point[-3:]))
        self.nonego = self._world.spawn_actor(transform=nonego_transform)
        self.ego_src = self._config.lane_start_point
        ego_transform = carla.Transform(carla.Location(x=self.nonego_spawn_point[0], y=self.ego_src[1], z=self.ego_src[2]), carla.Rotation(yaw=-90))
        self.ego = self._world.spawn_actor(transform=ego_transform)

        # Path planning
        ego_dest = self._config.lane_end_points
        dest_location = carla.Location(x=self.nonego_spawn_point[0], y=ego_dest[0][1], z=ego_dest[0][2])
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats['num_completed']
        super().on_step()

    def on_step(self) -> None:
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(self.actor_flow[0])
            if y < -181.9 or y > -81.2 or x < -38.4 or x > 31.6:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()
    
    def apply_control(self, action) -> None:
        control = self.get_vehicle_control(action)
        nonego_control = self.get_nonego_vehicle_control()
        self.ego.apply_control(control)
        self.nonego.apply_control(nonego_control)
    
    def get_nonego_vehicle_control(self):
        '''
        Non-ego vehicle control is designed for the scenario.
        '''
        ego_loc = self.ego.get_transform().location
        nonego_loc = self.nonego.get_transform().location

        # Keep constant speed
        if abs(self.nonego.get_velocity().y) < 2:
            acc = 2
        else:
            acc = 0

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc/3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc/3, 0, 1)

        return carla.VehicleControl(throttle=float(throttle), brake=float(brake))

    
    def reward(self):
        total_reward, info = super().reward()

        ego_loc = self.ego.get_transform().location
        nonego_loc = self.nonego.get_transform().location
        dist = math.sqrt((ego_loc.x - nonego_loc.x) ** 2 + (ego_loc.y - nonego_loc.y) ** 2)

        if dist < 5:
            total_reward += 200
        else:
            total_reward -= 30

        return total_reward