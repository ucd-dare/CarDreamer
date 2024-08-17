from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import get_vehicle_pos


class CarlaLaneMergeEnv(CarlaWptFixedEnv):
    """
    Vehicle merges into a lane and avoid collision.

    **Provided Tasks**: ``carla_lane_merge``
    """

    def on_step(self) -> None:
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(vehicle)
            if y < 37.5:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()
        super().on_step()
