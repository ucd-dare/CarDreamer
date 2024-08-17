import math
from enum import Enum
from typing import Dict, List, Tuple

import carla
import numpy as np

from ...carla_manager import ActorPolygonDict, ActorTransformDict


class Observability(Enum):
    FULL = "full"  # Full observation
    FOV = "fov"  # Partial observation baesd on Field of View
    RECURSIVE_FOV = "recursive_fov"  # Parital observation based on Sufficient Information


class WaypointObservability(Enum):
    ALL = "all"  # All vehicles
    VISIBLE = "visible"  # Only visible vehicles
    NEIGHBOR = "neighbor"  # Only neighbors


def is_point_in_fov(obs_location, obs_yaw, point, fov):
    # Calculate the distance between the observer and the point
    distance = math.sqrt((point[0] - obs_location[0]) ** 2 + (point[1] - obs_location[1]) ** 2)
    # If the point is beyond the maximum distance, return False
    if distance > 30:
        return False
    # Calculate angle between ego vehicle front direction and point
    direction_vector = np.array([math.cos(math.radians(obs_yaw)), math.sin(math.radians(obs_yaw))])
    point_vector = np.array([point[0] - obs_location[0], point[1] - obs_location[1]])
    point_vector = point_vector / np.linalg.norm(point_vector)

    dot_product = np.dot(direction_vector, point_vector)
    angle = math.degrees(math.acos(dot_product))

    # Check if point is within the FOV
    return abs(angle) <= fov / 2


def segments_intersect(a1, a2, b1, b2):
    # Check if line segments a1a2 and b1b2 intersect
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)


def is_line_of_sight_clear(point1, point2, polygons, id_filter):
    # Check if line from point1 to point2 intersects with any polygon
    for id, poly in polygons.items():
        if id in id_filter:
            continue
        for i in range(len(poly)):
            if segments_intersect(point1, point2, poly[i], poly[(i + 1) % len(poly)]):
                return False
    return True


def is_fov_visible(obs_location, obs_yaw, obs_id, id, poly, actor_polygons, fov):
    visible = False
    if id == obs_id:
        return visible
    for p in poly:
        if is_point_in_fov(obs_location, obs_yaw, p, fov) and is_line_of_sight_clear(obs_location, p, actor_polygons, id_filter=[obs_id, id]):
            visible = True
            break
    return visible


def get_visibility(
    ego: carla.Actor,
    actor_transforms: ActorTransformDict,
    actor_polys: ActorPolygonDict,
    fov: float,
) -> Tuple[Dict[int, bool], Dict[int, bool]]:
    """
    Get the visibility of the actors with respect to the ego
    Return:
        fov_visible: The first dictionary indicates if the actor is fov visible
        recursive_visible: The second dictionary indicates if the actor is recursive_fov visible
    """

    fov_visible, recursive_visible = {}, {}
    for id in actor_polys.keys():
        fov_visible[id] = False
        recursive_visible[id] = False

    ego_id = ego.id
    ego_transform = ego.get_transform()
    ego_yaw = ego_transform.rotation.yaw
    ego_location = ego_transform.location

    # FOV visibility
    for id, poly in actor_polys.items():
        if id == ego_id:
            continue
        fov_visible[id] = is_fov_visible(
            (ego_location.x, ego_location.y),
            ego_yaw,
            ego_id,
            id,
            poly,
            actor_polys,
            fov,
        )

    # For recursive_fov, iterate over vehicles and check if their surroundings are visible to themselves
    for obs_id, vis in fov_visible.items():
        if not vis:
            continue
        obs_location = actor_transforms[obs_id].location
        obs_yaw = actor_transforms[obs_id].rotation.yaw

        for id, poly in actor_polys.items():
            if id == obs_id or fov_visible[id] or recursive_visible[id]:
                continue
            if is_fov_visible(
                (obs_location.x, obs_location.y),
                obs_yaw,
                obs_id,
                id,
                poly,
                actor_polys,
                fov,
            ):
                recursive_visible[id] = True

    return fov_visible, recursive_visible


def should_filter(ego_transform, actor_transform):
    return abs(actor_transform.location.z - ego_transform.location.z) > 3


def get_neighbors(ego: carla.Actor, actor_transforms: ActorTransformDict, fov_visible: Dict[int, bool]) -> List[int]:
    """
    Get the nearest actors in the left (0), front (1), and right (2) direction in the four-lane system
    Return:
        neighbors: A list of actor ids in the left, front, and right direction
    """
    neighbors = [None, None, None]
    ego_id = ego.id
    ego_transform = ego.get_transform()
    ego_location = ego_transform.location

    for id, transform in actor_transforms.items():
        if id == ego_id or not fov_visible[id] or should_filter(ego_transform, transform):
            continue
        actor_location = transform.location
        if actor_location.x > 5.0 and actor_location.x < 16.2 and abs(actor_location.x - ego_location.x) < 4.0:
            if actor_location.x < ego_location.x - 1.0:
                if neighbors[0] is None or actor_location.y > actor_transforms[neighbors[0]].location.y:
                    neighbors[0] = id
            elif actor_location.x > ego_location.x + 1.0:
                if neighbors[2] is None or actor_location.y > actor_transforms[neighbors[2]].location.y:
                    neighbors[2] = id
            else:
                if neighbors[1] is None or actor_location.y > actor_transforms[neighbors[1]].location.y:
                    neighbors[1] = id

    return neighbors
