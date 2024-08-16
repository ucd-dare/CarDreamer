Quick Start
===========

.. _installation:

Installation
------------

First, clone the repository.

.. code-block:: console

   git clone https://github.com/ucd-dare/CarDreamer
   cd CarDreamer

Download `Carla release <https://github.com/carla-simulator/carla/releases>`_ of version ``0.9.15`` as we experiemented with this version. Set the following environment variables:

.. code-block:: console

   export CARLA_ROOT="</path/to/carla>"
   export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}

Then, install the package using flit. The ``--symlink`` flag is used to create a symlink to the package in the Python environment, so that changes to the package are immediately available without reinstallation. (``--pth-file`` also works, as an alternative to ``--symlink``.)

.. code-block:: console

   conda create python=3.10 --name cardreamer
   conda activate cardreamer
   pip install flit
   flit install --symlink

Creating a task
---------------------

To create a driving task, for example ``carla_four_lane``, you should first start Carla at port 2000. This is the default port used by the package, but can be changed in the configuration (See :ref:`port configuration <config-port>`).

.. code-block:: console

   $CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -carla-port=2000 -benchmark -fps=10

Then, call the python function :py:func:`car_dreamer.create_task` with the name of the task and optionally, the command line arguments.

.. code-block:: python

   import car_dreamer
   task, _ = car_dreamer.create_task('carla_four_lane', argv)

.. seealso::

   Function :py:func:`car_dreamer.create_task`

   A completed list of :doc:`tasks and configurations <./tasks>`.

Visualization
-------------

After creating the task, the visualization is automatically started if ``env.display.enable`` is set to ``True`` in ``car_dreamer/configs/common.yaml``. The visualization server runs on port 9000 by default  (See :ref:`port configuration <config-port>`). You can run the following demo and access to ``http://localhost:9000`` for visualization.

.. code-block:: python

   import car_dreamer
   import time

   task, _ = car_dreamer.create_task('carla_four_lane')
   task.reset()
   while True:
      _, _, is_terminal, _ = task.step(12)  # 12 is the one-hot action index of going straight and accelerating with default settings
      if is_terminal:
         task.reset()
      time.sleep(0.1)  # prevents from running too fast to visualize
