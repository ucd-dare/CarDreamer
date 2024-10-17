from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import get_vehicle_pos


class CarlaLeftTurnEnv(CarlaWptFixedEnv):
    """
    Vehicle passes the crossing (turn left) and avoid collision.

    **Provided Tasks**: ``carla_left_turn_simple``, ``carla_left_turn_medium``, ``carla_left_turn_hard``
    """

    def on_step(self) -> None:
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(self.actor_flow[0])
            if y > -99.4 or y < -171.4 or x > 57.6:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()
        super().on_step()
