from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import get_vehicle_pos


class CarlaRoundaboutEnv(CarlaWptFixedEnv):
    """
    Vehicle passes the roundabout and avoid collision.

    **Provided Tasks**: ``carla_roundabout``
    """

    def on_step(self) -> None:
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(vehicle)
            if (y < 0.0 and x < -39.8) or y < -47.2 or y > 46.0 or x > 44.8:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()
        super().on_step()
