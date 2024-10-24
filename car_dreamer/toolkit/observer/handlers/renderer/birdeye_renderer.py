import math
from typing import Dict, List, Tuple

import carla
import cv2
import random
import numpy as np


from ....carla_manager import ActorPolygon, Command, WorldManager
from ..utils import should_filter
from .constants import BirdeyeEntity, Color
from .map_renderer import MapRenderer


class BirdeyeRenderer:
    def __init__(
        self,
        world_manager: WorldManager,
        pixels_per_meter: float,
        screen_size: int,
        pixels_ahead_vehicle: int,
        fov: float,
    ):
        self._pixels_per_meter = pixels_per_meter
        self._screen_size = screen_size
        self._pixels_ahead_vehicle = pixels_ahead_vehicle
        self._fov = fov
        self._world_manager = world_manager

        self._map_renderer = MapRenderer(world_manager.carla_world, world_manager.carla_map, pixels_per_meter)
        self._world_to_pixel = self._map_renderer.world_to_pixel
        self._map_size = self._map_renderer._width_in_pixels

        self._surface = np.zeros((self._map_size, self._map_size, 3), dtype=np.uint8)
        self._font_size = screen_size // 5
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._command_initial = {
            Command.LaneFollow: "^",
            Command.LaneChangeLeft: "<",
            Command.LaneChangeRight: ">",
        }

    def set_ego(self, ego: carla.Actor):
        """Set the ego actor for rendering."""
        self._ego = ego

    def render(
        self,
        display: np.ndarray,
        entities: List[BirdeyeEntity],
        env_state: Dict,
    ):
        """Render the specified entities on the display."""
        self._clear_surface()
        self._render_entities(entities, **env_state)
        self._blit_to_display(display)

    def _clear_surface(self):
        """Clear the rendering surface."""
        # self._surface.fill(0)
        pass

    def _render_entities(self, entities: List[BirdeyeEntity], **env_state):
        """Render the specified entities on the surface."""
        for entity in entities:
            render_method = self._RENDER_METHODS.get(entity)
            if render_method:
                render_method(self, **env_state)
            else:
                raise ValueError(f"Invalid content type: {entity}")

    def _render_roadmap(self, **env_state):
        """Render the road map on the surface."""
        ego_transform = self._ego.get_transform()
        ego_location = self._world_to_pixel(ego_transform.location)
        x_start, x_end = max(0, ego_location[0] - self._screen_size), min(self._map_size, ego_location[0] + self._screen_size)
        y_start, y_end = max(0, ego_location[1] - self._screen_size), min(self._map_size, ego_location[1] + self._screen_size)
        self._surface[y_start:y_end, x_start:x_end, :] = self._map_renderer._surface[y_start:y_end, x_start:x_end, :]

    def _render_ego_vehicle(self, **env_state):
        """Render the ego actor on the surface."""
        color = env_state.get("ego_vehicle_color", Color.RED)
        ego_polygon = self._world_manager.actor_polygons[self._ego.id]
        self._render_polygon(self._surface, ego_polygon, color)

    def _render_fov_lines(self, **env_state):
        """Render the field of view lines on the surface."""
        ego_transform = self._ego.get_transform()
        fov = self._fov
        surface = self._surface
        line_length = 100

        ego_yaw = ego_transform.rotation.yaw
        left_fov_yaw = ego_yaw - fov / 2
        right_fov_yaw = ego_yaw + fov / 2

        ego_location = ego_transform.location
        left_fov_endpoint = (
            ego_location.x + line_length * math.cos(math.radians(left_fov_yaw)),
            ego_location.y + line_length * math.sin(math.radians(left_fov_yaw)),
        )
        right_fov_endpoint = (
            ego_location.x + line_length * math.cos(math.radians(right_fov_yaw)),
            ego_location.y + line_length * math.sin(math.radians(right_fov_yaw)),
        )

        ego_location_pixel = self._world_to_pixel(ego_location)
        left_fov_endpoint_pixel = self._world_to_pixel(carla.Location(*left_fov_endpoint))
        right_fov_endpoint_pixel = self._world_to_pixel(carla.Location(*right_fov_endpoint))

        cv2.line(surface, ego_location_pixel, left_fov_endpoint_pixel, Color.ORANGE_1, 3)
        cv2.line(surface, ego_location_pixel, right_fov_endpoint_pixel, Color.ORANGE_1, 3)

    def _render_waypoints(self, **env_state):
        """
        Render the waypoints for the ego actor on the surface.
        You must provide 'ego_waypoints'.
        """
        if "dest_x" in env_state:
            dest_start = carla.Location(x=env_state["dest_x"], y=self._ego.get_transform().location.y - 16)
            dest_start = self._world_to_pixel(dest_start)
            dest_end = carla.Location(x=env_state["dest_x"], y=self._ego.get_transform().location.y + 10)
            dest_end = self._world_to_pixel(dest_end)
            cv2.line(self._surface, dest_start, dest_end, Color.SKY_BLUE_0, 6)
        color = env_state.get("waypoints_color", Color.BLUE)
        ego_waypoints = env_state["ego_waypoints"]
        ego_polygon = self._world_manager.actor_polygons[self._ego.id]
        self._render_path(self._surface, ego_polygon, ego_waypoints, color)

    def _render_background_vehicles(self, **env_state):
        """Render the background vehicles on the surface."""
        color = env_state.get("background_vehicles_color")
        vehicle_polygons = self._world_manager.actor_polygons
        ego_id = self._ego.id

        for vehicle_id, polygon in vehicle_polygons.items():
            if vehicle_id == ego_id or should_filter(
                self._ego.get_transform(),
                self._world_manager.actor_transforms[vehicle_id],
            ):
                continue
            vehicle_color = color.get(vehicle_id, None)
            if vehicle_color is not None:
                self._render_polygon(self._surface, polygon, vehicle_color)

    def _render_background_waypoints(self, **env_state):
        """Render the waypoints for background actors on the surface."""
        color = env_state.get("background_waypoints_color")
        extend_waypoints = env_state.get("extend_waypoints", False)
        background_waypoints = self._world_manager.actor_actions
        background_waypoints = {
            id: [(action[1].transform.location.x, action[1].transform.location.y) for action in actions]
            for id, actions in background_waypoints.items()
            if actions
        }
        vehicle_polygons = self._world_manager.actor_polygons

        for vehicle_id, path in background_waypoints.items():
            if vehicle_id == self._ego.id or should_filter(
                self._ego.get_transform(),
                self._world_manager.actor_transforms[vehicle_id],
            ):
                continue
            vehicle_polygon = vehicle_polygons.get(vehicle_id, None)
            if vehicle_polygon is None:
                continue
            waypoint_color = color.get(vehicle_id, None)
            if waypoint_color is None:
                continue
            if extend_waypoints:
                last = path[-1]
                path.append((last[0], last[1] - 10.0))
            self._render_path(self._surface, vehicle_polygon, path, waypoint_color)

    def _render_error_background_waypoints(self, **env_state):
        """Render the waypoints with error for background actors on the surface."""
        color = env_state.get("background_waypoints_color")
        extend_waypoints = env_state.get("extend_waypoints", False)
        error_rate = env_state.get("error_rate")
        background_waypoints = self._world_manager.actor_actions
        background_waypoints = {
            id: [(action[1].transform.location.x, action[1].transform.location.y) for action in actions]
            for id, actions in background_waypoints.items()
            if actions
        }
        vehicle_polygons = self._world_manager.actor_polygons

        for vehicle_id, path in background_waypoints.items():
            if vehicle_id == self._ego.id or should_filter(self._ego.get_transform(), self._world_manager.actor_transforms[vehicle_id]):
                continue
            vehicle_polygon = vehicle_polygons.get(vehicle_id, None)
            if vehicle_polygon is None:
                continue
            waypoint_color = color.get(vehicle_id, None)
            if waypoint_color is None:
                continue
            if extend_waypoints:
                last = path[-1]
                path.append((last[0], last[1] - 10.0))
            if random.random() > error_rate:
                self._render_path(self._surface, vehicle_polygon, path, waypoint_color)

    def _render_messages(self, **env_state):
        """
        Render the messages for background actors on the surface.
        You must expose the command and goal through environment states.
        """
        color = env_state.get("messages_color", Color.WHITE)

        def render_character(location, message, message_color):
            font_scale, font_thickness = 3, 2
            pixel = self._world_to_pixel(location)
            text_size = cv2.getTextSize(message, self._font, font_scale, font_thickness)[0]
            # Adjusting position to center the text
            text_x = pixel[0] - text_size[0] // 2
            text_y = pixel[1] + text_size[1] // 2
            cv2.putText(
                self._surface,
                message,
                (text_x, text_y),
                self._font,
                font_scale,
                message_color,
                font_thickness,
                cv2.LINE_AA,
            )

        command = env_state.get("command", None)
        if command is not None:
            polygon = self._world_manager.actor_polygons[self._ego.id]
            location = carla.Location(
                x=(polygon[0][0] + polygon[1][0]) / 2,
                y=(polygon[0][1] + polygon[1][1]) / 2,
            )
            message = self._command_initial.get(command, None)
            message_color = Color.WHITE
            if message:
                render_character(location, message, message_color)

        background_messages = self._world_manager.actor_actions
        background_messages = {id: actions[0][0] for id, actions in background_messages.items() if actions}

        for vehicle_id, message in background_messages.items():
            if vehicle_id == self._ego.id or should_filter(
                self._ego.get_transform(),
                self._world_manager.actor_transforms[vehicle_id],
            ):
                continue
            polygon = self._world_manager.actor_polygons.get(vehicle_id, None)
            if polygon is None:
                continue
            message_color = color.get(vehicle_id)
            if message_color is None:
                continue
            message = self._command_initial.get(message, None)
            if message:
                location = carla.Location(
                    x=(polygon[0][0] + polygon[1][0]) / 2,
                    y=(polygon[0][1] + polygon[1][1]) / 2,
                )
                render_character(location, message, message_color)

    def _render_traffic_lights(self, **env_state):
        traffic_lights = self._world_manager.carla_actors("traffic_light")
        for traffic_light in traffic_lights:
            # Get the color based on the traffic light state
            if traffic_light.state == carla.TrafficLightState.Red:
                color = Color.SCARLET_RED_0
            elif traffic_light.state == carla.TrafficLightState.Yellow:
                color = Color.ORANGE_0
            elif traffic_light.state == carla.TrafficLightState.Green:
                color = Color.CHAMELEON_0
            else:
                continue

            # Render the traffic light
            self._render_traffic_light(self._surface, traffic_light, color)

    def _render_traffic_light(self, surface, traffic_light: carla.Actor, color: Color):
        """Render a traffic light on the surface."""
        world_pos = traffic_light.get_location()
        pos = self._world_to_pixel(world_pos)
        radius = int(self._pixels_per_meter * 1.2)
        cv2.circle(surface, center=pos, radius=radius, color=color, thickness=cv2.FILLED)

    def _render_stop_signs(self, **env_state):
        stop_sign_state = env_state.get("stop_sign_state")
        stop_signs = self._world_manager.carla_actors("stop")
        for stop_sign in stop_signs:
            if stop_sign.id in stop_sign_state:
                if stop_sign_state[stop_sign.id]["color"] == 0:
                    color = Color.SCARLET_RED_0
                else:
                    color = Color.CHAMELEON_0
            else:
                color = Color.CHAMELEON_0

            self._render_stop_sign(self._surface, stop_sign, color)

    def _render_stop_sign(self, surface, stop_sign: carla.Actor, color: Color):
        """Render a stop sign on the surface."""
        world_pos = stop_sign.get_location()
        pos = self._world_to_pixel(world_pos)
        radius = int(self._pixels_per_meter * 1.5)
        cv2.circle(surface, center=pos, radius=radius, color=color, thickness=cv2.FILLED)

    def _blit_to_display(self, display: np.ndarray):
        """Blit the rendered surface to the display."""
        if self._ego:
            self._blit_centered_and_rotated(display)
        else:
            self._blit_centered(display)

    def _blit_centered_and_rotated(self, display: np.ndarray):
        ego_transform = self._ego.get_transform()
        ego_center = self._world_to_pixel(ego_transform.location)

        rotate_matrix = cv2.getRotationMatrix2D(ego_center, ego_transform.rotation.yaw + 90, 1)
        rotate_matrix[0][2] -= ego_center[0] - self._screen_size / 2
        rotate_matrix[1][2] -= ego_center[1] - self._screen_size / 2 - self._pixels_ahead_vehicle
        self._rotated_surface = cv2.warpAffine(self._surface, rotate_matrix, (self._screen_size, self._screen_size))
        display[:] = self._rotated_surface

    def _blit_centered(self, display: np.ndarray):
        center_offset = (
            max(0, (display.shape[1] - self._map_size) // 2),
            max(0, (display.shape[0] - self._map_size) // 2),
        )
        display[
            center_offset[1] : center_offset[1] + min(self._map_size, display.shape[0]),
            center_offset[0] : center_offset[0] + min(self._map_size, display.shape[1]),
        ] = self._surface[
            : min(self._map_size, display.shape[0]),
            : min(self._map_size, display.shape[1]),
        ]

    def _render_polygon(self, surface: np.ndarray, vehicle_polygon: ActorPolygon, color: Color):
        """Render a list of polygons on the surface."""
        actor_corners = np.array(
            [self._world_to_pixel(carla.Location(x=p[0], y=p[1])) for p in vehicle_polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(surface, [actor_corners], color)

    def _render_path(
        self,
        surface: np.ndarray,
        vehicle_polygon: ActorPolygon,
        path: List[Tuple[float, float]],
        color: Color,
    ):
        """Render a path on the surface using the centroid of the polygon."""
        # Calculate the centroid of the polygon
        num_points = len(vehicle_polygon)
        centroid_x = sum(p[0] for p in vehicle_polygon) / num_points
        centroid_y = sum(p[1] for p in vehicle_polygon) / num_points
        polygon_center = (centroid_x, centroid_y)

        # skip rendering obsolete path points
        start = 0
        for i, p in enumerate(path):
            p_direction = np.array(p[:2]) - np.array(polygon_center)
            vehicle_direction = np.array(vehicle_polygon[1]) - np.array(vehicle_polygon[2])
            if np.dot(p_direction, vehicle_direction) < 0:
                start = i + 1
        if start >= len(path):
            return
        path = path[start:]

        # Convert the centroid and path points to pixel coordinates
        corners = np.array(
            [self._world_to_pixel(carla.Location(x=p[0], y=p[1])) for p in [polygon_center] + path],
            dtype=np.int32,
        )
        # Draw the path as a polyline on the surface
        cv2.polylines(surface, [corners], False, color, 15)

    _RENDER_METHODS = {
        BirdeyeEntity.ROADMAP: _render_roadmap,
        BirdeyeEntity.BACKGROUND_VEHICLES: _render_background_vehicles,
        BirdeyeEntity.EGO_VEHICLE: _render_ego_vehicle,
        BirdeyeEntity.FOV_LINES: _render_fov_lines,
        BirdeyeEntity.WAYPOINTS: _render_waypoints,
        BirdeyeEntity.BACKGROUND_WAYPOINTS: _render_background_waypoints,
        BirdeyeEntity.TRAFFIC_LIGHTS: _render_traffic_lights,
        BirdeyeEntity.STOP_SIGNS: _render_stop_signs,
        BirdeyeEntity.MESSAGES: _render_messages,
        BirdeyeEntity.ERROR_BACKGROUND_WAYPOINTS: _render_error_background_waypoints,
    }
