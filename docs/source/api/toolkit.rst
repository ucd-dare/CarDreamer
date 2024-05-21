:py:mod:`car_dreamer.toolkit`
================================

.. py:module:: car_dreamer.toolkit

Here are the API calls provided by our task development suite. See :doc:`../customization` for guides on how to customize your own task.

.. autoclass:: WorldManager
   :members:

.. autoclass:: Observer
   :members:

   The output of the Observer can be configured through ``env.observation.enabled``. See :ref:`observation configuration <config-observation>` for possible choices.

.. _planner:

.. autoclass:: BasePlanner
   :members:

.. autoclass:: RandomPlanner
   :members:

.. autoclass:: FixedPathPlanner
   :members:

.. autoclass:: FixedEndingPlanner
   :members:

.. automodule:: car_dreamer.toolkit.carla_manager.utils
   :members:
