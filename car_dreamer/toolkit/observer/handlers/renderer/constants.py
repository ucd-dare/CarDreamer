from enum import Enum


class ColorBGR:
    BUTTER_0 = (79, 233, 252)
    BUTTER_1 = (0, 212, 237)
    BUTTER_2 = (0, 160, 196)
    ORANGE_0 = (62, 175, 252)
    ORANGE_1 = (0, 121, 245)
    ORANGE_2 = (0, 92, 209)
    CHOCOLATE_0 = (110, 185, 233)
    CHOCOLATE_1 = (17, 125, 193)
    CHOCOLATE_2 = (2, 89, 143)
    CHAMELEON_0 = (52, 226, 138)
    CHAMELEON_1 = (22, 210, 115)
    CHAMELEON_2 = (6, 154, 78)
    SKY_BLUE_0 = (207, 159, 114)
    SKY_BLUE_1 = (164, 101, 52)
    SKY_BLUE_2 = (135, 74, 32)
    PLUM_0 = (168, 127, 173)
    PLUM_1 = (123, 80, 117)
    PLUM_2 = (102, 53, 92)
    SCARLET_RED_0 = (41, 41, 239)
    SCARLET_RED_1 = (0, 0, 204)
    SCARLET_RED_2 = (0, 0, 164)
    ALUMINIUM_0 = (236, 238, 238)
    ALUMINIUM_1 = (207, 215, 211)
    ALUMINIUM_2 = (182, 189, 186)
    ALUMINIUM_3 = (133, 138, 136)
    ALUMINIUM_4 = (83, 87, 85)
    ALUMINIUM_5 = (64, 62, 66)
    ALUMINIUM_6 = (54, 52, 46)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)


class Color:
    BUTTER_0 = (252, 233, 79)
    BUTTER_1 = (237, 212, 0)
    BUTTER_2 = (196, 160, 0)
    ORANGE_0 = (252, 175, 62)
    ORANGE_1 = (245, 121, 0)
    ORANGE_2 = (209, 92, 0)
    CHOCOLATE_0 = (233, 185, 110)
    CHOCOLATE_1 = (193, 125, 17)
    CHOCOLATE_2 = (143, 89, 2)
    CHAMELEON_0 = (138, 226, 52)
    CHAMELEON_1 = (115, 210, 22)
    CHAMELEON_2 = (78, 154, 6)
    SKY_BLUE_0 = (114, 159, 207)
    SKY_BLUE_1 = (52, 101, 164)
    SKY_BLUE_2 = (32, 74, 135)
    PLUM_0 = (173, 127, 168)
    PLUM_1 = (117, 80, 123)
    PLUM_2 = (92, 53, 102)
    SCARLET_RED_0 = (239, 41, 41)
    SCARLET_RED_1 = (204, 0, 0)
    SCARLET_RED_2 = (164, 0, 0)
    ALUMINIUM_0 = (238, 238, 236)
    ALUMINIUM_1 = (211, 215, 207)
    ALUMINIUM_2 = (186, 189, 182)
    ALUMINIUM_3 = (136, 138, 133)
    ALUMINIUM_4 = (85, 87, 83)
    ALUMINIUM_5 = (66, 62, 64)
    ALUMINIUM_6 = (46, 52, 54)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)


class BirdeyeEntity(Enum):
    ROADMAP = "roadmap"
    EGO_VEHICLE = "ego_vehicle"
    BACKGROUND_VEHICLES = "background_vehicles"
    FOV_LINES = "fov_lines"
    WAYPOINTS = "waypoints"
    BACKGROUND_WAYPOINTS = "background_waypoints"
    TRAFFIC_LIGHTS = "traffic_lights"
    STOP_SIGNS = "stop_signs"
    MESSAGES = "messages"
    ERROR_BACKGROUND_WAYPOINTS = "error_background_waypoints"
