import carla
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_vehicle_pos

class CarlaFollowEnv(CarlaWptEnv)