Welcome to use ``car_dreamer``!
===================================

Package ``car_dreamer`` is a collection of tasks aimed at facilitating RL algorithm designing, especially world model based ones. Each task is a certain driving environment in Carla simulator, varying from a single skill such as lane following or left turning, to random roaming in mixed road conditions which may encounter crossroads, roundabouts, and stop signs. They expose the same gym interface for backbone RL algorithm use.

Furthermore, ``car_dreamer`` includes a task development suite for those who want to customize their own tasks. It provides a number of API calls to minimize users' efforts in spawning the vehicles, planning the routes, and obtaining observation data as RL algorithm inputs. And an integrated traning visualization server automatically grasps the observation data, displaying the videos and plotting the statistics through an HTTP server. This eases algorithm designing and hyper-parameter tuning.

.. note::
   This document only describes the installation and customization of this package, but it should be used with an RL backend. For examples of training a world model using our tasks, please refer to `CarDreamer <https://github.com/labdare/car-dreamer>`_.

Contents
--------

.. toctree::
   :maxdepth: 3

   usage
   tasks
   customization
   api
