Customization
================

To customize a driving task, one can either adapt existing tasks by modifying the configurations or create a new task from scratch.

Configurations
----------------

The simplest way to customize your own task is to modify the configurations in ``car_dreamer/configs/common.yaml`` and ``car_dreamer/configs/tasks.yaml``.  The common configurations define the observations, terminal conditions, etc., that are re-used by all different tasks; task configurations define each the observation, action, terminal conditions, reward scales, display options, for each specific task. See :doc:`configurations <./tasks>` for more details.

Creating a new environment
----------------------------

In this section, we will guide you through the process of creating a new task script, defining a new observation source, and implementing a new planner.

.. py:currentmodule:: car_dreamer

Create a new file ``carla_custom_env.py`` in ``car_dreamer/``. Define a class ``CarlaCustomEnv`` inherited from :py:class:`CarlaBaseEnv`, and implement the following methods. Specifically, you can call the APIs from :py:class:`car_dreamer.toolkit.WorldManager` to spawn vehicles, and use various :ref:`planners <planner>` to generate waypoints for the ego vehicle.

.. autoclass:: CarlaBaseEnv
   :members: on_reset, apply_control, on_step, reward, get_terminal_conditions, get_ego_vehicle, get_state

A typical implementation looks like:

.. code-block:: python

   import carla

   from .carla_base_env import CarlaBaseEnv
   from .toolkit import RandomPlanner

   class CarlaCustomEnv(CarlaBaseEnv):
       def on_reset(self):
           # spawn the ego vehicle at a specific location in Town04
           # self.ego is a must to inherit from CarlaBaseEnv
           spawn_transform = carla.Transform(carla.Location(x=5.8, y=100, z=0.1), carla.Rotation(yaw=-90))
           self.ego = self._world_manager.spawn_actor(transform=spawn_transform)

           # use random planner to generate waypoints for the ego vehicle
           # self.ego_planner is a must to inherit from CarlaWptEnv or CarlaWptFixedEnv
           self.ego_planner = RandomPlanner(vehicle=self.ego)
           self.waypoints, self.planner_stats = self.ego_planner.run_step()

       def apply_control(self, action):
           # apply control to the ego vehicle according to action
           control = self.get_vehicle_control(action)
           self.ego.apply_control(control)

       def on_step(self):
           # run the planner to generate waypoints for the ego vehicle
           self.waypoints, self.planner_stats = self.ego_planner.run_step()
           self.num_waypoint_reached = self.planner_stats['num_completed']

       def reward(self):
           r_waypoint = self.num_waypoint_reached * self._config.waypoint_reward
           return r_waypoint, {
               'r_waypoint': r_waypoint,
           }

       def get_terminal_conditions(self):
           return {
               'is_collision': self.is_collision(),
               'time_exceeded': self._time_step > self._config.time_limit,
           }

       def get_state(self):
           return {
               'timesteps': self._time_step,
               'waypoints': self.waypoints,
           }

You can also inherit from :py:class:`CarlaWptEnv` or :py:class:`CarlaWptFixedEnv` where some of the methods have been implemented.

After you create the environment, the task will be automatically registered in gym with id ``CarlaCustomEnv-v0``. Note that you must follow the same naming style in order to be recognized by the toolkit (``Carla{TaskName}Env`` and ``carla_{task_name}_env.py``).

Finally, create a task configuration in ``car_dreamer/configs/tasks.yaml``. It will be available in ``self._config`` in the environment class. For example:

.. code-block:: yaml

   carla_custom:
       env:
       name: 'CarlaCustomEnv-v0'
       observation.enabled: [camera, collision, birdeye_wpt]
       # specify value for constants used in the environment
       waypoint_reward: 1.0
       time_limit: 500

Now you will be able to create the task by ``car_dreamer.create_task('carla_custom')`` where ``carla_custom`` is the task name in ``tasks.yaml``.

Defining a new observation source
---------------------------------

.. py:currentmodule:: car_dreamer.toolkit.observer.handlers

For observations, we have defined several items in ``car_dreamer/configs/common.yaml``. Define ``env.observation.enabled`` to specify the items you want to include in the observation data. See :ref:`observation configuration <config-observation>` for more details.

To customize another observation source, create a new handler in ``car_dreamer/toolkit/observer/handlers/`` inherited from :py:class:`BaseHandler`. Then implement the following methods:

.. autoclass:: BaseHandler
   :members: get_observation_space, get_observation, destroy, reset

For example, if you want to include the speed and the number of remaining waypoints in the observation data, you can implement a new handler as follows:

.. code-block:: python

   from gym import spaces
   import numpy as np

   from .base_handler import BaseHandler
   from ...carla_manager import get_vehicle_velocity

   class CustomHandler(BaseHandler):

       def get_observation_space(self):
           return {
                'num_waypoints': spaces.Box(low=0, high=65535, shape=(1,), dtype=np.uint16),
                'speed': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
              }

       def get_observation(self, env_state):
           # 'waypoints' should be returned by the environment
           num_waypoints = len(env_state['waypoints'])
           # speed is directly obtained from CARLA
           speed = get_vehicle_velocity(self.ego)
           return {
               'num_waypoints': np.array([num_waypoints], dtype=np.uint16),
                'speed': np.array(speed, dtype=np.float32),
           }

       def destroy(self):
           pass

       def reset(self, ego):
           self.ego = ego

Add a new enum in ``car_dreamer/toolkit/observer/utils.py``:

.. code-block:: python

   from .handlers.custom_handler import CustomHandler

   class HandlerType(Enum):
       ...
       CUSTOM = 'custom'

   HANDLER_DICT = {
       ...
       HandlerType.CUSTOM: CustomHandler,
   }

Add a new configuration item in ``car_dreamer/configs/common.yaml`` where ``custom`` is the handler name above (``CUSTOM = 'custom'``):

.. code-block:: yaml

   env:
       ...
       observation:
           ...
           custom:
               handler: 'custom'

Finally, you can include the new observation source by adding ``custom`` to ``env.observation.enabled`` in the task configuration.

Creating a new planner
----------------------

.. py:currentmodule:: car_dreamer.toolkit.planner

All planners should expose the interface:

.. automethod:: BasePlanner.run_step

We have implemented a base class :py:class:`BasePlanner` for you to inherit from. All you need to do is to define how to initialize and extend the route when running each step:

.. automethod:: BasePlanner.init_route

.. automethod:: BasePlanner.extend_route

Additionally, you can call the following methods to get, add, or pop waypoints stored in the planner:

.. autoclass:: BasePlanner
   :members: get_all_waypoints, add_waypoint, pop_waypoint, clear_waypoints, get_waypoint_num, from_carla_waypoint
