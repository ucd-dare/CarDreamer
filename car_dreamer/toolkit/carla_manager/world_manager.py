import time
from functools import wraps
from typing import Callable, Dict, List, Union

import carla
import numpy as np

from .utils import ActorActionDict, ActorPolygonDict, ActorTransformDict, Command
from .vehicle_manager import VehicleManager


def cached_step_wise(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        cache_key = (func.__name__,) + tuple(args) + tuple(kwargs.items())
        if not hasattr(self, "_cache") or self._cache["step"] != self._time_step:
            self._cache = {"step": self._time_step}
        if cache_key not in self._cache:
            self._cache[cache_key] = func(self, *args, **kwargs)
        return self._cache[cache_key]

    return wrapper


class WorldManager:
    """
    The class to manage the world in CARLA.
    You can spawn various actors using this class.
    The actors spawned by this class will be automatically destroyed when reset.
    This class also provides methods to get information about these actors.
    """

    def __init__(self, env_config):
        self._config = env_config.world
        self._env_config = env_config

        print(f"[CARLA] Connecting to Carla server at {self._config.carla_port}...")
        self._client = carla.Client("127.0.0.1", self._config.carla_port)
        self._client.set_timeout(20.0)
        self._world = self._client.load_world(self._config.town)
        self._map = self._world.get_map()
        print(f"[CARLA] Map {self._config.town} loaded")

        settings = self._world.get_settings()
        settings.synchronous_mode = False
        settings.actor_active_distance = self._config.actor_active_distance
        settings.fixed_delta_seconds = self._config.fixed_delta_seconds
        self._world.apply_settings(settings)
        self._settings = settings

        self._tm_port = self._config.carla_port + 6000
        self._vehicle_manager = VehicleManager(self._client, self._tm_port, self._config.traffic)

        self._on_reset = None
        self._apply_control = None
        self._on_step = None
        self.actor_dict = {}
        self._time_step = 0

        self._ego_planner = None

    def on_reset(self, callback: Callable[[], None]) -> None:
        """
        Register a callback function to be called when the environment is reset.
        If called multiple times, it will overwrite the previous callback.
        """
        self._on_reset = callback

    def on_step(self, callback: Callable[[], None]) -> None:
        """
        Register a callback function to be called when the environment steps.
        If called multiple times, it will overwrite the previous callback.
        """
        self._on_step = callback

    def reset(self) -> None:
        # destroy all actors
        self._time_step = 0
        self._client.apply_batch_sync([carla.command.DestroyActor(id) for id in self.actor_dict])
        self.actor_dict = {}

        self._set_synchronous_mode(False)

        if self._on_reset is not None:
            self._on_reset()

        self._set_synchronous_mode(True)
        # This prevents some synchronization bugs
        time.sleep(1)

    def step(self) -> None:
        self._time_step += 1
        self._world.tick()
        if self._on_step is not None:
            self._on_step()

    def get_blueprint_library(self, pattern_filter: str, attribute_filter: Dict[str, str] = None) -> carla.BlueprintLibrary:
        """
        Get blueprint library based on the pattern filter and attribute filter.
        """
        bps = self._world.get_blueprint_library().filter(pattern_filter)
        if attribute_filter is not None:
            for name, value in attribute_filter.items():
                bps = bps.filter_by_attribute(name, value)
        return bps

    def get_blueprint(self, pattern_filter: str, attribute_filter: Dict[str, str] = None) -> carla.ActorBlueprint:
        """
        Randomly get a blueprint from the library based on the pattern filter and attribute filter.
        """
        bps = self.get_blueprint_library(pattern_filter, attribute_filter)
        assert len(bps) > 0, f"No blueprint found for filter {pattern_filter} {attribute_filter}"
        return np.random.choice(bps)

    def get_spawn_points(self) -> List[carla.Transform]:
        """
        Get spawn points of the map.
        """
        return self._map.get_spawn_points()

    def get_random_spawn_point(self) -> carla.Transform:
        """
        Get a random spawn point of the map.
        """
        spawn_points = self.get_spawn_points()
        assert len(spawn_points) > 0, "No spawn points found"
        return np.random.choice(spawn_points)

    def try_spawn_actor(
        self,
        transform: Union[carla.Transform, None] = None,
        blueprint: Union[carla.ActorBlueprint, None] = None,
    ) -> Union[carla.Actor, None]:
        """
        Spawn an actor with the given blueprint and transform.

        :param transform: if None, use a random spawn point.
        :param blueprint: if None, use vehicle.audi* with number_of_wheels in 4 as default.

        :return: the spawned actor. If fails, return None.
        """
        if transform is None:
            transform = self.get_random_spawn_point()
        if blueprint is None:
            blueprint = self.get_blueprint("vehicle.audi*", {"number_of_wheels": "4"})
            if blueprint.has_attribute("color"):
                color = np.random.choice(blueprint.get_attribute("color").recommended_values)
                blueprint.set_attribute("color", color)
            blueprint.set_attribute("role_name", "hero")
        actor = self._world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            self.actor_dict[actor.id] = actor
        return actor

    def spawn_actor(
        self,
        transform: Union[carla.Transform, None] = None,
        blueprint: Union[carla.ActorBlueprint, None] = None,
        max_try_time: int = None,
    ) -> carla.Actor:
        """
        Equivalent to ``try_spawn_actor(transform, blueprint)``, but retry if failed.

        :param max_try_time: if None, try until success, else raise an exception after ``max_try_time``.

        .. seealso:: :py:meth:`try_spawn_actor`
        """
        actor = self.try_spawn_actor(transform, blueprint)
        try_time = 0
        while actor is None and (max_try_time is None or try_time < max_try_time):
            print("[CARLA] Failed to spawn actor, retrying...")
            time.sleep(0.1)
            actor = self.try_spawn_actor(transform, blueprint)
            try_time += 1
        if actor is None:
            raise Exception("Failed to spawn actor")
        return actor

    def spawn_unmanaged_actor(self, transform: carla.Transform, blueprint: carla.ActorBlueprint, **kwargs) -> carla.Actor:
        """
        Spawn an actor with the given blueprint and transform.
        Actors spawned by this method will be omitted by this manager.
        That is, they will not be included when retrieving actor information or destroyed when reset.
        This is useful when creating sensors for :py:class:`car_dreamer.toolkit.observer.handlers.SensorHandler`.
        """
        return self._world.spawn_actor(blueprint, transform, **kwargs)

    def spawn_auto_actors(
        self,
        n: int,
        transforms: List[carla.Transform] = None,
        blueprints: carla.BlueprintLibrary = None,
    ) -> List[carla.Actor]:
        """
        Spawn ``n`` actors that are automatically controlled by autopilot.

        :param n: number of actors to spawn.
        :param transforms: if None, use random spawn points.
        :param blueprints: if None, use vehicle.* with number_of_wheels 4 as default.

        :return: a list of spawned actors, note that the length of the list may be less than n.
        """
        if transforms is None:
            transforms = self.get_spawn_points()
        if blueprints is None:
            blueprints = self.get_blueprint_library("vehicle.*", {"number_of_wheels": "4"})
        batch = []
        actor_list = []
        np.random.shuffle(transforms)
        for transform in transforms[: min(n, len(transforms))]:
            bp = np.random.choice(blueprints)
            if bp.has_attribute("color"):
                color = np.random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            if bp.has_attribute("driver_id"):
                driver_id = np.random.choice(bp.get_attribute("driver_id").recommended_values)
                bp.set_attribute("driver_id", driver_id)
                bp.set_attribute("role_name", "autopilot")
            batch.append(carla.command.SpawnActor(bp, transform).then(carla.command.SetAutopilot(carla.command.FutureActor, True, self._tm_port)))
        for response in self._client.apply_batch_sync(batch, False):
            if response.error:
                print("[CARLA]", response.error)
            else:
                actor = self._world.get_actor(response.actor_id)
                actor_list.append(actor)
                self.actor_dict[actor.id] = actor
                self._vehicle_manager.set_auto_lane_change(actor, self._config.auto_lane_change)
                if "background_speed" in self._config:
                    self._vehicle_manager.set_desired_speed(actor, self._config.background_speed)
        return actor_list

    def try_spawn_aggresive_actor(
        self,
        transform: Union[carla.Transform, None] = None,
        blueprint: Union[carla.ActorBlueprint, None] = None,
    ) -> Union[carla.Actor, None]:
        """
        Similar to ``try_spawn_actor(transform, blueprint)``.
        But the actor will be automatically controlled by autopilot and ignore traffic lights and other vehicles.

        .. seealso:: :py:meth:`try_spawn_actor`
        """
        vehicle = self.try_spawn_actor(transform, blueprint)
        if vehicle is None:
            return None
        vehicle.set_autopilot(True, self._tm_port)
        self._vehicle_manager.set_auto_lane_change(vehicle, True)
        if "background_speed" in self._config:
            self._vehicle_manager.set_desired_speed(vehicle, self._config.background_speed)
        self._vehicle_manager._tm.ignore_lights_percentage(vehicle, 100)
        self._vehicle_manager._tm.ignore_vehicles_percentage(vehicle, 100)
        return vehicle

    def destroy_actor(self, actor_id: int) -> None:
        """
        Destroy an actor. Call this method if you want to manually destroy an actor spawned by this manager.

        .. warning::
           Do not call this method for actors spawned by :py:meth:`spawn_unmanaged_actor`.
           Directly call :py:meth:`carla.Actor.destroy` instead.
        """
        actor = self.actor_dict.pop(actor_id)
        actor.destroy()

    @property
    def actor_ids(self) -> List[int]:
        """
        Get the ids of all actors spawned by this manager.
        """
        return list(self.actor_dict.keys())

    @property
    def actors(self) -> List[carla.Actor]:
        """
        Get all actors spawned by this manager.
        """
        return list(self.actor_dict.values())

    @cached_step_wise
    def _get_actor_polygons(self) -> ActorPolygonDict:
        actor_polygons: ActorPolygonDict = {}

        for actor in self.actors:
            actor_transform = actor.get_transform()
            x = actor_transform.location.x
            y = actor_transform.location.y

            yaw = actor_transform.rotation.yaw * np.pi / 180

            # Get length and width of the bounding box
            bb = actor.bounding_box
            l, w = bb.extent.x, bb.extent.y

            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).T

            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).T + np.repeat([[x, y]], 4, axis=0)
            actor_polygons[actor.id] = poly.tolist()

        return actor_polygons

    @property
    def actor_polygons(self) -> ActorPolygonDict:
        """
        Get the bounding box polygons of all actors spawned by this manager.

        :return: a dictionary mapping actor IDs to their bounding box polygons.
        :rtype: dict[int, list[tuple[float, float]]]
        """
        return self._get_actor_polygons()

    @cached_step_wise
    def _get_actor_actions(self) -> ActorActionDict:
        actor_actions: ActorActionDict = {}

        for actor in self.actor_dict.values():
            try:
                actions = self._vehicle_manager._tm.get_all_actions(actor)
                actor_actions[actor.id] = [(Command(command), waypoint) for command, waypoint in actions]
            except Exception as e:  # noqa: F841
                pass

        return actor_actions

    @property
    def actor_actions(self) -> ActorActionDict:
        """
        Get the actions of all actors spawned by this manager.

        :return: a dictionary mapping vehicle IDs to their known actions.
        :rtype: dict[int, list[tuple[Command, carla.Waypoint]]]

        .. warning::
           Actors not controlled by autopilot will not have actions.
           They will not be included in the returned dictionary.
           And some actors may have an empty list if there is no known action.
        """
        return self._get_actor_actions()

    @cached_step_wise
    def _get_actor_transforms(self) -> ActorTransformDict:
        return {actor.id: actor.get_transform() for actor in self.actor_dict.values()}

    @property
    def actor_transforms(self) -> ActorTransformDict:
        """
        Get the transforms of all actors spawned by this manager.

        :return: a dictionary mapping actor IDs to their transforms.
        :rtype: dict[int, carla.Transform]
        """
        return self._get_actor_transforms()

    def _set_synchronous_mode(self, synchronous=True):
        self._settings.synchronous_mode = synchronous
        self._world.apply_settings(self._settings)
        self._vehicle_manager.set_synchronous_mode(synchronous)

    def _get_world(self):
        return self._world

    @property
    def carla_world(self):
        return self._get_world()

    def _get_map(self):
        return self._map

    @property
    def carla_map(self):
        return self._get_map()

    @cached_step_wise
    def _get_carla_actors(self, actor_type: str = "") -> List[carla.Actor]:
        filtered_actors = []
        carla_actors = self._world.get_actors()
        for actor in carla_actors:
            if actor_type in actor.type_id:
                filtered_actors.append(actor)
        return filtered_actors

    def carla_actors(self, actor_type: str = "") -> List[carla.Actor]:
        """
        Get all actors of a specific type directly through CARLA APIs.

        :param actor_type: the type of the actors to retrieve (e.g., 'vehicle', 'traffic_light').
        :return: a list of actors of the specified type.
        """
        return self._get_carla_actors(actor_type)
