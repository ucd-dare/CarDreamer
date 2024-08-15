Welcome to use ``car_dreamer``!
===================================

``car_dreamer`` aims at facilitating the training and evaluation of reinforcement learning and world model algorithms in the highly realistic simulator `CARLA <http://carla.org/>`_.

.. image:: https://github.com/Alexwangziyu/CarDreamer/blob/master/.assets/cardreamer_architecture.png
   :alt: CarDreamer Architecture
   :align: center
   :width: 600px

``car_dreamer`` provides a collection of well-defined urban driving tasks. Each task encompasses a driving environment and a task goal. The tasks vary from a single skill such as lane following or left turning, to random roaming in mixed road conditions which may encounter crossroads, roundabouts, and stop signs. All the tasks support the `Gym <https://gym.openai.com/>`_ interface so that users can easily plug in their own algorithms.

Furthermore, ``car_dreamer`` includes a task development suite that simplifies the customization of driving tasks. It provides a number of API calls to minimize users' efforts in spawning and managing the vehicles, planning routes, and obtaining diverse observation data for RL algorithms. A visualization server automatically grasps the observation data, displaying the videos and the statistics (e.g., terminal conditions, rewards, other information of user interest) through an HTTP server. This eases task and algorithm engineering and debugging.

.. note::
   This document details the installation and customization of ``car_dreamer``. For information regarding training and evaluation using specific RL and world model algorithms, please refer to `CarDreamer <https://github.com/ucd-dare/CarDreamer>`_ and the `arXiv paper <https://arxiv.org/abs/2405.09111>`_.

Contents
--------

.. toctree::
   :maxdepth: 3

   quick_start
   tasks
   customization
   api
