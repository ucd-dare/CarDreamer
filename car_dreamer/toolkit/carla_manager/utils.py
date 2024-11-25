from enum import Enum
from typing import Dict, List, Tuple

import carla
import numpy as np


class Command(Enum):
    """Enumeration for high-level commands."""

    LaneFollow = "LaneFollow"
    LaneChangeLeft = "ChangeLaneLeft"
    LaneChangeRight = "ChangeLaneRight"
    Straight = "Straight"
    Left = "Left"
    Right = "Right"

class FollowDirections(Enum):
    """Enumeration for the follow task directions"""
    STRAIGHT = 0
    RIGHT_TURN = 1
    LEFT_TURN = 2
    RANDOM = -1


Index2Command = {index: command for index, command in enumerate(Command, start=0)}
Command2Index = {command: index for index, command in enumerate(Command, start=0)}

ActorPolygon = List[Tuple[float, float]]
ActorPolygonDict = Dict[int, ActorPolygon]

ActorAction = Tuple[Command, carla.Waypoint]
ActorActionDict = Dict[int, List[ActorAction]]

ActorTransformDict = Dict[int, carla.Transform]


def get_vehicle_pos(vehicle: carla.Actor) -> Tuple[float, float]:
    """
    Get the position of a vehicle.

    :param vehicle: carla.Actor

    :return: x, y position of the vehicle
    """
    location = vehicle.get_transform().location
    return location.x, location.y


def get_vehicle_orientation(vehicle: carla.Actor) -> float:
    """
    Get the orientation of a vehicle.

    :param vehicle: carla.Actor

    :return: orientation of the vehicle
    """
    return vehicle.get_transform().rotation.yaw


def get_vehicle_velocity(vehicle: carla.Actor) -> Tuple[float, float]:
    """
    Get the velocity of a vehicle.

    :param vehicle: carla.Actor

    :return: tuple(float, float), x, y velocity of the vehicle
    """
    velocity = vehicle.get_velocity()
    return velocity.x, velocity.y


def get_location_distance(location1: Tuple[float, float], location2: Tuple[float, float]) -> float:
    """
    Compute the distance between two locations

    :param location1: tuple(float, float)
    :param location2: tuple(float, float)

    :return: float, distance between the two locations
    """
    return np.linalg.norm(np.array([location1[0] - location2[0], location1[1] - location2[1]]))


class TTCCalculator:
    """
    A class for calculating Time-to-Collision (TTC) between vehicles as static methods.
    """

    TTC_THRESHOLD = 100.0
    DIST_THRESHOLD = 100.0

    @staticmethod
    def is_vehicle_ahead(ego_vehicle, map, target_location):
        """
        Check if a target vehicle is ahead of the ego vehicle within the proximity threshold.

        :param ego_vehicle: The ego vehicle.
        :param map: The simulation map.
        :param target_location: The location of the target vehicle.
        :param dist_threshold: The maximum distance to consider a vehicle as nearby.
        :return: True if the target vehicle is ahead and within the proximity threshold, False otherwise.
        """
        ego_location = ego_vehicle.get_location()
        ego_waypoint = map.get_waypoint(ego_location)
        target_waypoint = map.get_waypoint(target_location)

        if target_waypoint.road_id != ego_waypoint.road_id or target_waypoint.lane_id != ego_waypoint.lane_id:
            return False

        ego_forward_vector = ego_vehicle.get_transform().get_forward_vector()
        target_vector = target_location - ego_location
        distance = target_vector.length()

        if distance > TTCCalculator.DIST_THRESHOLD:
            return False

        target_vector_normalized = target_vector.make_unit_vector()
        dot_product = ego_forward_vector.dot(target_vector_normalized)

        return dot_product > 0.0

    @staticmethod
    def find_nearby_vehicles(world, ego_vehicle, map):
        """
        Find nearby vehicles within the specified proximity threshold.

        :param world: The simulation world.
        :param ego_vehicle: The ego vehicle.
        :param map: The simulation map.
        :param dist_threshold: The distance threshold to consider vehicles as nearby.
        :return: List of nearby vehicles.
        """
        nearby_vehicles = []
        vehicle_list = world.get_actors().filter("vehicle.*")

        for target_vehicle in vehicle_list:
            if target_vehicle.id == ego_vehicle.id:
                continue

            target_location = target_vehicle.get_location()
            if TTCCalculator.is_vehicle_ahead(ego_vehicle, map, target_location):
                nearby_vehicles.append(target_vehicle)

        return nearby_vehicles

    @staticmethod
    def get_ttc_to_target(ego_vehicle, target_vehicle):
        """
        Compute the Time-to-Collision (TTC) between the ego vehicle and a target vehicle.

        :param ego_vehicle: The ego vehicle.
        :param target_vehicle: The target vehicle.
        :return: The computed TTC value.
        """
        ego_location = ego_vehicle.get_location()
        target_location = target_vehicle.get_location()

        ego_velocity = ego_vehicle.get_velocity()
        target_velocity = target_vehicle.get_velocity()

        relative_velocity = target_velocity - ego_velocity
        relative_speed = relative_velocity.length()

        distance = ego_location.distance(target_location)

        if relative_speed > 0:
            ttc = distance / relative_speed
        else:
            ttc = TTCCalculator.TTC_THRESHOLD

        return ttc

    @staticmethod
    def get_ttc(ego_vehicle, world, map):
        """
        Compute the minimum Time-to-Collision (TTC) between the ego vehicle and nearby vehicles.

        :param world: The simulation world.
        :param ego_vehicle: The ego vehicle.
        :param map: The simulation map.
        :param ttc_threshold: The threshold to stop calculations if a low TTC is found.
        :return: The minimum TTC value.
        """
        nearby_vehicles = TTCCalculator.find_nearby_vehicles(world, ego_vehicle, map)
        min_ttc = TTCCalculator.TTC_THRESHOLD

        for target_vehicle in nearby_vehicles:
            ttc = TTCCalculator.get_ttc_to_target(ego_vehicle, target_vehicle)
            if ttc < min_ttc:
                min_ttc = ttc

        return min_ttc if min_ttc < TTCCalculator.TTC_THRESHOLD else 0.0
