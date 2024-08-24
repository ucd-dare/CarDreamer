from typing import List, Tuple

import carla
import cv2
import numpy as np

from .constants import Color


def lane_marking_color_to_tango(lane_marking_color: carla.LaneMarkingColor) -> Color:
    tango_color = Color.BLACK

    if lane_marking_color == carla.LaneMarkingColor.White:
        tango_color = Color.ALUMINIUM_2
    elif lane_marking_color == carla.LaneMarkingColor.Blue:
        tango_color = Color.SKY_BLUE_0
    elif lane_marking_color == carla.LaneMarkingColor.Green:
        tango_color = Color.CHAMELEON_0
    elif lane_marking_color == carla.LaneMarkingColor.Red:
        tango_color = Color.SCARLET_RED_0
    elif lane_marking_color == carla.LaneMarkingColor.Yellow:
        tango_color = Color.ORANGE_0

    return tango_color


def render_solid_line(
    surface: np.ndarray,
    color: Color,
    closed: bool,
    points: List[Tuple[int, int]],
    width: int,
):
    if len(points) >= 2:
        points_array = np.array(points, dtype=np.int32)
        cv2.polylines(surface, [points_array], closed, color, width)


def render_dotted_line(
    surface: np.ndarray,
    color: Color,
    closed: bool,
    points: List[Tuple[int, int]],
    width: int,
):
    dotted_lines = [points[i : i + 20] for i in range(0, len(points), 60)]
    for line in dotted_lines:
        line_array = np.array(line, dtype=np.int32)
        cv2.polylines(surface, [line_array], closed, color, width)


def lateral_shift(transform: carla.Transform, shift: float) -> carla.Location:
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


class MapRenderer:
    def __init__(self, world: carla.World, map: carla.Map, pixels_per_meter: float):
        self._world = world
        self._map = map
        self._pixels_per_meter = pixels_per_meter
        self._scale = 1.0
        self._precision = 0.05
        self._topology = self._map.get_topology()

        waypoints = map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self._font_size = self.world_to_pixel_width(1)
        self._font = cv2.FONT_HERSHEY_SIMPLEX

        self._width_in_meter = max(max_x - min_x, max_y - min_y)
        self._world_offset_in_meter = (min_x, min_y)
        self._width_in_pixels = int(self._pixels_per_meter * self._width_in_meter)
        self._surface = np.zeros((self._width_in_pixels, self._width_in_pixels, 3), dtype=np.uint8)

        self.render_topology()
        self.render_traffic_signs()

    def render_topology(self, index=0):
        topology = self._topology
        topology = [x[index] for x in topology]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        set_waypoints = []

        for waypoint in topology:
            waypoints = [waypoint]

            nxt = waypoint.next(self._precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(self._precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            set_waypoints.append(waypoints)
            self.render_lanes(waypoints)

        for waypoints in set_waypoints:
            self.render_road(waypoints)
            self.render_road_marking(waypoints)

    def render_road(self, waypoints: List[carla.Waypoint]):
        road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
        road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
        polygon = road_left_side + list(reversed(road_right_side))
        polygon_pixel = np.array([self.world_to_pixel(x) for x in polygon], dtype=np.int32)
        if len(polygon) > 2:
            cv2.fillPoly(self._surface, [polygon_pixel], Color.ALUMINIUM_5)
            cv2.polylines(self._surface, [polygon_pixel], True, Color.ALUMINIUM_5, 5)

    def render_road_marking(self, waypoints: List[carla.Waypoint]):
        if waypoints[0].is_junction:
            return
        self.render_lane_marking(waypoints, -1)
        self.render_lane_marking(waypoints, 1)

    def render_lane_marking(self, waypoints: List[carla.Waypoint], sign: int):
        lane_marking = None

        marking_type = carla.LaneMarkingType.NONE
        previous_marking_type = carla.LaneMarkingType.NONE

        marking_color = carla.LaneMarkingColor.Other
        previous_marking_color = carla.LaneMarkingColor.Other

        markings_list = []
        temp_waypoints = []
        current_lane_marking = carla.LaneMarkingType.NONE

        for sample in waypoints:
            lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking
            if lane_marking is None:
                continue

            marking_type = lane_marking.type
            marking_color = lane_marking.color

            if current_lane_marking != marking_type:
                markings = self.get_lane_markings(previous_marking_type, previous_marking_color, temp_waypoints, sign)
                current_lane_marking = marking_type
                markings_list.extend(markings)
                temp_waypoints = temp_waypoints[-1:]
            else:
                temp_waypoints.append(sample)
                previous_marking_type = marking_type
                previous_marking_color = marking_color

        last_markings = self.get_lane_markings(previous_marking_type, previous_marking_color, temp_waypoints, sign)
        markings_list.extend(last_markings)

        for marking_type, marking_color, marking_points in markings_list:
            if marking_type == carla.LaneMarkingType.Solid:
                render_solid_line(self._surface, marking_color, False, marking_points, 5)
            elif marking_type == carla.LaneMarkingType.Broken:
                render_dotted_line(self._surface, marking_color, False, marking_points, 5)

    def get_lane_markings(
        self,
        marking_type: carla.LaneMarkingType,
        marking_color: carla.LaneMarkingColor,
        waypoints: List[carla.Waypoint],
        sign: int,
    ) -> List[Tuple[carla.LaneMarkingType, Color, List[Tuple[int, int]]]]:
        margin = 0.25
        marking_1 = np.array(
            [self.world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints],
            dtype=np.int32,
        )
        if marking_type == carla.LaneMarkingType.Broken or marking_type == carla.LaneMarkingType.Solid:
            return [(marking_type, lane_marking_color_to_tango(marking_color), marking_1)]
        else:
            marking_2 = np.array(
                [self.world_to_pixel(lateral_shift(w.transform, sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints],
                dtype=np.int32,
            )
            if marking_type == carla.LaneMarkingType.SolidBroken:
                return [
                    (
                        carla.LaneMarkingType.Broken,
                        lane_marking_color_to_tango(marking_color),
                        marking_1,
                    ),
                    (
                        carla.LaneMarkingType.Solid,
                        lane_marking_color_to_tango(marking_color),
                        marking_2,
                    ),
                ]
            elif marking_type == carla.LaneMarkingType.BrokenSolid:
                return [
                    (
                        carla.LaneMarkingType.Solid,
                        lane_marking_color_to_tango(marking_color),
                        marking_1,
                    ),
                    (
                        carla.LaneMarkingType.Broken,
                        lane_marking_color_to_tango(marking_color),
                        marking_2,
                    ),
                ]
            elif marking_type == carla.LaneMarkingType.BrokenBroken:
                return [
                    (
                        carla.LaneMarkingType.Broken,
                        lane_marking_color_to_tango(marking_color),
                        marking_1,
                    ),
                    (
                        carla.LaneMarkingType.Broken,
                        lane_marking_color_to_tango(marking_color),
                        marking_2,
                    ),
                ]
            elif marking_type == carla.LaneMarkingType.SolidSolid:
                return [
                    (
                        carla.LaneMarkingType.Solid,
                        lane_marking_color_to_tango(marking_color),
                        marking_1,
                    ),
                    (
                        carla.LaneMarkingType.Solid,
                        lane_marking_color_to_tango(marking_color),
                        marking_2,
                    ),
                ]

        return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]

    def render_lanes(self, waypoints: List[carla.Waypoint]):
        SHOULDER_COLOR = Color.ALUMINIUM_5
        PARKING_COLOR = Color.ALUMINIUM_4
        SIDEWALK_COLOR = Color.ALUMINIUM_3

        shoulder = [[], []]
        parking = [[], []]
        sidewalk = [[], []]

        for w in waypoints:
            left_lane = w.get_left_lane()
            while left_lane and left_lane.lane_type != carla.LaneType.Driving:
                if left_lane.lane_type == carla.LaneType.Shoulder:
                    shoulder[0].append(left_lane)
                if left_lane.lane_type == carla.LaneType.Parking:
                    parking[0].append(left_lane)
                if left_lane.lane_type == carla.LaneType.Sidewalk:
                    sidewalk[0].append(left_lane)
                left_lane = left_lane.get_left_lane()

            r = w.get_right_lane()
            while r and r.lane_type != carla.LaneType.Driving:
                if r.lane_type == carla.LaneType.Shoulder:
                    shoulder[1].append(r)
                if r.lane_type == carla.LaneType.Parking:
                    parking[1].append(r)
                if r.lane_type == carla.LaneType.Sidewalk:
                    sidewalk[1].append(r)
                r = r.get_right_lane()

        self.render_lane(shoulder, SHOULDER_COLOR)
        self.render_lane(parking, PARKING_COLOR)
        self.render_lane(sidewalk, SIDEWALK_COLOR)

    def render_lane(self, lane: List[List[carla.Waypoint]], color: Color):
        for side in lane:
            lane_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
            lane_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]

            polygon = lane_left_side + list(reversed(lane_right_side))
            polygon_pixel = np.array([self.world_to_pixel(x) for x in polygon], dtype=np.int32)

            if len(polygon) > 2:
                cv2.fillPoly(self._surface, [polygon_pixel], color)
                cv2.polylines(self._surface, [polygon_pixel], True, color, 5)

    def render_traffic_signs(self):
        actors = self._world.get_actors()
        stops = [actor for actor in actors if "stop" in actor.type_id]
        yields = [actor for actor in actors if "yield" in actor.type_id]

        stop_font_surface = np.zeros((self._font_size * 2, self._font_size * 4, 3), dtype=np.uint8)
        cv2.putText(
            stop_font_surface,
            "STOP",
            (0, self._font_size * 2),
            self._font,
            1,
            Color.ALUMINIUM_2,
            2,
            cv2.LINE_AA,
        )

        yield_font_surface = np.zeros((self._font_size * 2, self._font_size * 4, 3), dtype=np.uint8)
        cv2.putText(
            yield_font_surface,
            "YIELD",
            (0, self._font_size * 2),
            self._font,
            1,
            Color.ALUMINIUM_2,
            2,
            cv2.LINE_AA,
        )

        for ts_stop in stops:
            self.render_traffic_sign(ts_stop, stop_font_surface, Color.SCARLET_RED_1)

        for ts_yield in yields:
            self.render_traffic_sign(ts_yield, yield_font_surface, Color.ORANGE_1)

    def render_traffic_sign(self, actor: carla.Actor, font_surface: np.ndarray, trigger_color: Color):
        transform = actor.get_transform()
        waypoint = self._map.get_waypoint(transform.location)

        angle = -waypoint.transform.rotation.yaw - 90.0
        font_surface = cv2.warpAffine(
            font_surface,
            cv2.getRotationMatrix2D((font_surface.shape[1] // 2, font_surface.shape[0] // 2), angle, 1.0),
            (font_surface.shape[1], font_surface.shape[0]),
        )
        pixel_pos = self.world_to_pixel(waypoint.transform.location)
        offset = (
            pixel_pos[0] - font_surface.shape[1] // 2,
            pixel_pos[1] - font_surface.shape[0] // 2,
        )
        self._surface[
            offset[1] : offset[1] + font_surface.shape[0],
            offset[0] : offset[0] + font_surface.shape[1],
        ] = font_surface

        forward_vector = carla.Location(waypoint.transform.get_forward_vector())
        left_vector = carla.Location(-forward_vector.y, forward_vector.x, forward_vector.z) * waypoint.lane_width / 2 * 0.7

        line = [
            (waypoint.transform.location + (forward_vector * 1.5 + left_vector)),
            (waypoint.transform.location + (forward_vector * 1.5) - left_vector),
        ]
        line_pixel = [self.world_to_pixel(p) for p in line]
        cv2.polylines(
            self._surface,
            [np.array(line_pixel, dtype=np.int32)],
            True,
            trigger_color,
            5,
        )

    def world_to_pixel(self, location: carla.Location, offset: Tuple[float, float] = (0, 0)) -> Tuple[int, int]:
        x = self._scale * self._pixels_per_meter * (location.x - self._world_offset_in_meter[0])
        y = self._scale * self._pixels_per_meter * (location.y - self._world_offset_in_meter[1])
        return int(x - offset[0]), int(y - offset[1])

    def world_to_pixel_width(self, width: float) -> int:
        return int(self._scale * self._pixels_per_meter * width)
