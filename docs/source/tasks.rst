Tasks and Configurations
=============================

``car_dreamer`` provides a collection of urban driving tasks with gym interfaces. Each task is defined by a set of configurations, and an environment python script, which can be easily modified to create new tasks or adapt existing ones. This document describes the configurations and how to use them to create tasks.

File Structure
--------------

Each top-level key in ``car_dreamer/configs/tasks.yaml`` is a task name and contains task-specific configurations. Configurations within the sub-level key ``env``, combined with those in ``car_dreamer/configs/common.yaml``, constitutes the complete configurations for the task. Configurations under other sub-level keys such as ``dreamerv3`` are recommended configurations for the corresponding backend algorithms. They are not used by the task itself, but as a way to automatically adapt the backend algorithm to the task.

Common Configurations
---------------------

Here is the descriptions of some common configurations in ``car_dreamer/configs/common.yaml``. Key-values not listed here should remain unchanged unless you know exactly how they work. **All keys below are under the top-level key ``env``.**

* ``world``: the configuration for :py:class:`car_dreamer.toolkit.WorldManager`.

    .. _config-port:

    * ``carla_port`` (default: ``2000``): the port number of the CARLA server on which you start.
      The port for Carla's Traffic Manager is set to ``carla_port + 6000``.

      The port for Visualization Server is set to ``carla_port + 7000``.
    * ``town`` (default: ``Town04``): the map name of the CARLA world.
    * ``fixed_delta_seconds`` (default: ``0.1``): the fixed time step of the simulation.
    * ``auto_lane_change`` (default: ``True``): whether to enable automatic lane change for vehicles controlled by autopilot.
    * ``background_speed`` (default: ``None``): set desired speed for vehicles controlled by autopilot.

.. _config-observation:

* ``observation``: the configuration for :py:class:`car_dreamer.toolkit.Observer`.
  Each sub-item is enabled if included in the list ``env.observation.enabled``.

  For each sub-item, at least ``handler`` should be specified, which is used to find the corresponding handler in :py:mod:`car_dreamer.toolkit.observer.handlers`. Here is a list of items already defined and you may configure your own.

    * ``camera``: the configuration for :py:class:`car_dreamer.toolkit.observer.handlers.CameraHandler`.
      If enabled, provides ``{${camera.key}: image of ndarray(np.int8) of size ${camera.shape}}`` in the observation data.

        * ``key`` (default: ``camera``): the key of the camera in the observation data.
        * ``shape`` (default: ``[128, 128, 3]``): the shape of the image, should follow the form of ``[n, n, 3]``.
        * ``blueprint``: the blueprint of the camera sensor in CARLA.
        * ``transform``: the relative position of the camera sensor to the vehicle.
        * ``attributes``: attributes passed to the constructor of camera sensor in CARLA.

            * ``image_size_x`` (default: ``128``): should equal to ``n`` in ``shape``.
            * ``image_size_y`` (default: ``128``): should equal to ``n`` in ``shape``.
            * ``fov`` (default: ``120.0``): the field of view of the camera.

    * ``lidar``: the configuration for :py:class:`car_dreamer.toolkit.observer.handlers.LidarHandler`.
      If enabled, provides ``{${lidar.key}: ndarray(np.int8) of size ${lidar.shape}}`` in the observation data.

        * ``key`` (default: ``lidar``): the key of the lidar in the observation data.
        * ``shape`` (default: ``[128, 128, 3]``): the shape of the lidar, should follow the form of ``[n, n, 3]``.
        * ``blueprint``: the blueprint of the lidar sensor in CARLA.
        * ``lidar_bin`` (default: ``0.25``): the bin size of the lidar in meter (in unit of CARLA).
        * ``ego_offset`` (default: ``12``): the offset of the ego vehicle to the bottom of the lidar image in meter (in unit of CARLA).
        * ``transform``: the relative position of the lidar sensor to the vehicle.
        * ``attributes``: attributes passed to the constructor of lidar sensor in CARLA.

            * ``range`` (default: ``32.0``): the range of the lidar in meter (in unit of CARLA).

    * ``collision``: the configuration for :py:class:`car_dreamer.toolkit.observer.handlers.CollisionHandler`.
      If enabled, provides ``{${collision.key}: ndarray(np.float32) of size (1,)}`` in the observation data.
      It is the impulse of the collision, or zero if no collision happens.

        * ``key`` (default: collision): the key of the collision in the observation data.
        * ``blueprint``: the blueprint of the collision sensor in CARLA.

    * ``birdeye_*``: the configuration for :py:class:`car_dreamer.toolkit.observer.handlers.BirdEyeHandler`.
      If enabled, provides ``{${birdeye.key}: ndarray(np.int8) of size ${birdeye.shape}}`` in the observation data.

      **Note that** multiple birdeye handlers can be enabled at the same time to provide different birdeye views.

        * ``key``: the key of the birdeye in the observation data.
        * ``shape`` (default: ``[128, 128, 3]``): the shape of the birdeye, should follow the form of ``[n, n, 3]``.
        * ``obs_range`` (default: ``32``): the range of the birdeye in meter (in unit of CARLA).
        * ``ego_offset`` (default: ``12``): the offset of the ego vehicle to the bottom of the birdeye in meter (in unit of CARLA).
        * ``camera_fov`` (default: ``150``): the field of view of the camera.
        * ``observability`` (default: ``full``): can be ``full, recursive_fov, fov``. ``full`` means all background vehicles are visible. ``fov`` means only vehicles in the field of view are visible. ``recursive_fov`` means vehicles in the fov of vehicles in ego's fov are also visible.
        * ``color_by_obs``: whether to color the vehicles by observability.
        * ``waypoint_obs``: can be ``neighbor, visible, all``. It controls whether to render the intended waypoints of background vehicles. ``neighbor`` means only the neighboring vehicle's waypoint is rendered. ``visible`` means only visible vehicles' waypoints are rendered. ``all`` means all vehicles' waypoints are rendered.
        * ``entities``: can be a subset of ``[roadmap, waypoints, background_waypoints, ego_vehicle, background_vehicles, fov_lines, messages]`` to control which entities are rendered in the birdeye view. Entities are rendered in the order specified by ``entities``, overwriting if overlapped.

            * ``roadmap``: the roadmap.
            * ``waypoints``: the ego waypoints, need ``ego_waypoints`` from environment.
            * ``background_waypoints``: the intended waypoints of background vehicles.
            * ``ego_vehicle``: the ego vehicle.
            * ``background_vehicles``: the background vehicles.
            * ``fov_lines``: the field of view lines.

* ``display``: the configuration for :py:class:`car_dreamer.toolkit.EnvMonitorOpenCV`.

    * ``enabled`` (default: ``True``): whether to enable the visualization.
    * ``render_keys`` (default: ``[camera, birdeye_wpt]``): the keys of the observation data to render.
    * ``image_size`` (default: ``512``): the size of the rendered image.

* ``action``: the configuration for action space.

    * ``discrete`` (default: ``True``): whether to use discrete action space.
      If ``True``, ``len(discrete_acc) * len(discrete_steer)`` is the size of the action space.
    * ``discrete_acc`` (default: ``[-2.0, 0.0, 2.0]``): the discrete values for acceleration.
    * ``discrete_steer`` (default: ``[-0.6, -0.2, 0.0, 0.2, 0.6]``): the discrete values for steering.

Tasks and Environments
--------------------------

For task specific configurations in ``car_dreamer/configs/tasks.yaml``, two keys are important:

* ``env.name`` is the name of the environment class to instantiate appended by ``-v0``. Each environment, together with a set of configurations, defines a task. The high configurability of environment allows creating a new task or modifying an existing one without changing the source code. See below for the list of available :ref:`environments <environments>`.
* ``env.observation.enabled`` specifies the list of observation items to enable. It decides which content is contained in the observation data. See :ref:`above <config-observation>` for possible items.

.. _environments:

Here is a list of available environments and their supported tasks and configuration parameters:

.. autoclass:: car_dreamer.CarlaWptEnv
   :show-inheritance:

.. autoclass:: car_dreamer.CarlaFourLaneEnv
   :show-inheritance:

.. autoclass:: car_dreamer.CarlaNavigationEnv
   :show-inheritance:

.. autoclass:: car_dreamer.CarlaOvertakeEnv
   :show-inheritance:

.. autoclass:: car_dreamer.CarlaWptFixedEnv
   :show-inheritance:

.. autoclass:: car_dreamer.CarlaLaneMergeEnv
   :show-inheritance:

.. autoclass:: car_dreamer.CarlaLeftTurnEnv
   :show-inheritance:

.. autoclass:: car_dreamer.CarlaRightTurnEnv
   :show-inheritance:

.. autoclass:: car_dreamer.CarlaRoundaboutEnv
   :show-inheritance:

.. autoclass:: car_dreamer.CarlaStopSignEnv
   :show-inheritance:

.. autoclass:: car_dreamer.CarlaTrafficLightsEnv
   :show-inheritance:

.. note::
   All configurations mentioned here can also be changed by command line arguments. For example, to change the port number of the CARLA server, you can pass ``--env.world.carla_port <number>``. See :py:func:`car_dreamer.create_task` for more details.
