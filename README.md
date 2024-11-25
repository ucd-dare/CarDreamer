# 🌍 Learn to Drive in "Dreams": CarDreamer 🚗

<div align="center">
    <a href="https://huggingface.co/ucd-dare/CarDreamer/tree/main">
        <img src="https://img.icons8.com/?size=32&id=sop9ROXku5bb" alt="HuggingFace Checkpoints" />
        HuggingFace Checkpoints
    </a>
    &nbsp;|&nbsp;
    <a href="https://car-dreamer.readthedocs.io/en/latest/">
        <img src="https://img.icons8.com/nolan/32/api.png" alt="CarDreamer API Documents" />
        CarDreamer API Documents
    </a>
    &nbsp;|&nbsp;
    <a href="https://ieeexplore.ieee.org/document/10714437">
        <img src="https://img.icons8.com/?size=32&id=48326&format=png" alt="IEEE IoT" />
        Paper
    </a>
    &nbsp;|&nbsp;
    <a href="https://ucd-dare.github.io/cardreamer.github.io/">
        <img src="https://img.icons8.com/?size=32&id=X-WB1cntO5xU&format=png&color=000000" alt="Project Page" />
        Project Page
    </a>
</div>

______________________________________________________________________

Unleash the power of **imagination** and **generalization** of world models for self-driving cars.

> \[!NOTE\]
>
> - **October 2024:** CarDreamer has been accepted by [IEEE IoT](https://ieeexplore.ieee.org/document/10714437)!
> - **August 2024:** Support transmission error in intention sharing.
> - **August 2024:** Created a right turn random task.
> - **July 2024:** Created a stop-sign task and a traffic-light task.
> - **July 2024:** Uploaded all the task checkpoints to [HuggingFace](https://huggingface.co/ucd-dare/CarDreamer/tree/main)
> - **May 2024** Released [arXiv preprint](https://arxiv.org/abs/2405.09111).

## **Can world model based self-supervised reinforcement learning train autonomous driving agents via imagination of traffic dynamics? The answer is YES!**

Integrating the high-fidelity CARLA simulator with world models, we are able to train a world model that not only learns complex environment dynamics but also have an agent interact with the neural network "simulator" to learn to drive.

Simply put, in CarDreamer the agent can learn to drive in a "dream world" from scratch, mastering maneuvers like overtaking and right turns, and avoiding collisions in heavy traffic—all within an imagined world!

CarDreamer offers **customizable observability**, **multi-modal observation spaces**, and **intention-sharing** capabilities. Our paper presents a systematic analysis of the impact of different inputs on agent performance.

Dive into our demos to see the agent skillfully navigating challenges and ensuring safe and efficient travel.

## 📚 Open-Source World Model-Based Autonomous Driving Platform

**Explore** world model based autonomous driving with CarDreamer, an open-source platform designed for the **development** and **evaluation** of **world model** based autonomous driving.

- 🏙️ **Built-in Urban Driving Tasks**: flexible and customizable observation modality, observability, intention sharing; optimized rewards
- 🔧 **Task Development Suite**: create your own urban driving tasks with ease
- 🌍 **Model Backbones**: integrated state-of-the-art world models

**Documentation:** [CarDreamer API Documents](https://car-dreamer.readthedocs.io/en/latest/).

**Looking for more technical details? Check our report here! [Paper link](https://arxiv.org/abs/2405.09111)**

## :sun_with_face: Built-in Task Demos

> \[!TIP\]
> A world model is learnt to model traffic dynamics; then a driving agent is trained on world model's imagination! The driving agent masters diverse driving skills including lane merge, left turn, and right turn, to random roaming purely **from scratch**.

We train DreamerV3 agents on our built-in tasks with a single 4090. Depending on the observation spaces, the memory overhead ranges from 10GB-20GB alongwith 3GB reserved for CARLA.

| Right turn hard | Roundabout | Left turn hard | Lane merge | Overtake |
| :-------------: | :--------: | :------------: | :--------: | :------: |
| ![Right turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_hard.gif) | ![Roundabout](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/roundabout.gif) | ![Left turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/left_turn_hard.gif) | ![Lane merge](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/lane_merge.gif) | ![Overtake](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/overtake.gif) |

| Right turn hard | Roundabout | Left turn hard | Lane merge | Overtake |
| :-------------: | :--------: | :------------: | :--------: | :---------------: |
| ![Right turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/camera/right_turn_hard.gif) | ![Roundabout](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/camera/roundabout.gif) | ![Left turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/camera/left_turn_hard.gif) | ![Lane merge](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/camera/lane_merge.gif) | ![Right turn simple](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/camera/overtake.gif) |

| Traffic Light | Stop Sign |
| :-----------: | :-------: |
| ![Traffic Light](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/tl_right.gif) | ![Stop Sign](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/stop%20sign.gif) |

## :blossom: The Power of Intention Sharing

> \[!TIP\]
> **Human drivers use turn signals to inform their intentions** of turning left or right. **Autonomous vehicles can do the same!**

Let's see how CarDreamer agents communicate and leverage intentions. Our experiment has demonstrated that through sharing intention, the policy learning is much easier! Specifically, a policy without knowing other agents' intentions can be conservative in our crossroad tasks; while intention sharing allows the agents to find the proper timing to cut in the traffic flow.

| Sharing waypoints vs. Without sharing waypoints | Sharing waypoints vs. Without sharing waypoints |
| :---------------------------------------------: | :---------------------------------------------: |
| **Right turn hard**                                  | **Left turn hard**                                  |
| ![Right turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_hard.gif)     ![Right turn hard no waypoint](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_raw_fail.gif) | ![Left turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/left_turn_hard.gif)    <img src="https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/left_turn_raw.gif" style="width: 100%"> |

| Full observability vs. Partial observability |
| :------------------------------------------: |
| **Right turn hard**                               |
| ![Right turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_hard.gif)     ![Right turn hard FOV](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_fov.gif) |

## 📋 Experiments

### Task Performance

The following table shows the overall performance metrics over different CarDreamer built-in tasks.

| Task Performance Metrics |
| :-----------------------: |
| ![Task Performance](https://ucd-dare.github.io/cardreamer.github.io/static/images/tables/task_performance.png) |

### Observability

CarDreamer enables the customization of different levels of observability. The table below highlights performance metrics under different observability settings, including full observability, field-of-view (FOV), and recursive field-of-view (SFOV). These settings allow agents to operate with varying degrees of environmental awareness, impacting their ability to plan and execute maneuvers effectively.

| Observability Performance Metrics |
| :-------------------------------: |
| ![Observability Performance](https://ucd-dare.github.io/cardreamer.github.io/static/images/tables/observability_performance.png) |

### Intention Sharing

CarDreamer enhances autonomous vehicle planning by allowing vehicles to share their driving intentions, akin to how human drivers use turn signals. This feature facilitates smoother interactions between agents. Additionally, CarDreamer offers the ability to introduce and customize transmission errors in intention sharing, allowing for a more realistic simulation of communication imperfections. The table below presents performance results under various intention sharing and transmission error configurations.

| Intention Sharing and Transmission Errors |
| :--------------------------------------: |
| ![Transmission Error Intention](https://ucd-dare.github.io/cardreamer.github.io/static/images/tables/transmission_error_intention.png) |

## 📋 Prerequisites

### CarDreamer Dependencies

To install CarDreamer tasks or the development suite, clone the repository:

```bash
git clone https://github.com/ucd-dare/CarDreamer
cd CarDreamer
```

Download [CARLA release](https://github.com/carla-simulator/carla/releases) of version `0.9.15` as we experiemented with this version. Set the following environment variables:

```bash
export CARLA_ROOT="</path/to/carla>"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}
```

Install the package using flit. The `--symlink` flag is used to create a symlink to the package in the Python environment, so that changes to the package are immediately available without reinstallation. (`--pth-file` also works, as an alternative to `--symlink`.)

```bash
conda create python=3.10 --name cardreamer
conda activate cardreamer
pip install flit
flit install --symlink
```

### Model Dependencies

The model backbones are decoupled from CarDreamer tasks or the development sutie. Users can install model dependencies on their own demands. To install DreamerV2 and DreamerV3, check out the guidelines in separate folders [DreamerV3](https://github.com/ucd-dare/CarDreamer/tree/master/dreamerv3) or [DreamerV2](https://github.com/ucd-dare/CarDreamer/tree/master/dreamerv2). The experiments in our paper were conducted using DreamerV3, the current state-of-the-art world models.

## :gear: Quick Start

### :mechanical_arm: Training

Find `README.md` in the corresponding directory of the algorithm you want to use and follow the instructions to install dependencies for that model and start training. We suggest starting with DreamerV3 as it is showing better performance across our experiments. To train DreamerV3 agents, use

```bash
bash train_dm3.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/carla_four_lane
```

The command will launch CARLA at 2000 port, load task a built-in task named `carla_four_lane`, and start the visualization tool at port 9000 (2000+7000) which can be accessed through `http://localhost:9000/`. You can append flags to the command to overwrite yaml configurations.

### Creating Tasks

The section explains how to create CarDreamer tasks in a standalone mode without loading our integrated models. This can be helpful if you want to train and evaluate your own models other than our integrated DreamerV2 and DreamerV3 on CarDreamer tasks.

CarDreamer offers a range of built-in task classes, which you can explore in the [CarDreamer Docs: Tasks and Configurations](https://car-dreamer.readthedocs.io/en/latest/tasks.html#tasks-and-environments).

Each task class can be instantiated with various configurations. For instance, the right-turn task can be set up with simple, medium, or hard settings. These settings are defined in YAML blocks within [tasks.yaml](https://github.com/ucd-dare/CarDreamer/blob/master/car_dreamer/configs/tasks.yaml). The task creation API retrieves the given identifier (e.g., `carla_four_lane_hard`) from these YAML task blocks and injects the settings into the task class to create a gym task instance.

```python
# Create a gym environment with default task configurations
import car_dreamer
task, task_configs = car_dreamer.create_task('carla_four_lane_hard')

# Or load default environment configurations without instantiation
task_configs = car_dreamer.load_task_configs('carla_right_turn_hard')
```

To create your own driving tasks using the development suite, refer to [CarDreamer Docs: Customization](https://car-dreamer.readthedocs.io/en/latest/customization.html).

### Observation Customization

`CarDreamer` employs an `Observer-Handler` architecture to manage complex **multi-modal** observation spaces. Each handler defines its own observation space and lifecycle for stepping, resetting, or fetching information, similar to a gym environment. The agent communicates with the environment through an observer that manages these handlers.

Users can enable built-in observation handlers such as BEV, camera, LiDAR, and spectator in task configurations. Check out [common.yaml](https://github.com/ucd-dare/CarDreamer/blob/master/car_dreamer/configs/common.yaml) for all available built-in handlers. Additionally, users can customize observation handlers and settings to suit their specific needs.

#### Handler Implementation

To implement new handlers for different observation sources and modalities (e.g., text, velocity, locations, or even more complex data), `CarDreamer` provides two methods:

1. Register a callback as a [SimpleHandler](https://github.com/ucd-dare/CarDreamer/blob/master/car_dreamer/toolkit/observer/handlers/simple_handler.py) to fetch data at each step.
1. For observations requiring complex workflows that cannot be conveyed by a `SimpleHandler`, create an handler maintaining the full lifecycle of that observation, similar to our built-in message, BEV, spectator handlers.

For more details on defining new observation sources, see [CarDreamer Docs: Defining a new observation source](https://car-dreamer.readthedocs.io/en/latest/customization.html#defining-a-new-observation-source).

#### Observation Handler Configurations

Each handler can access yaml configurations for further customization. For example, a BEV handler setting can be defined as:

```yaml
birdeye_view:
   # Specify the handler name used to produce `birdeye_view` observation
   handler: birdeye
   # The observation key
   key: birdeye_view
   # Define what to render in the birdeye view
   entities: [roadmap, waypoints, background_waypoints, fov_lines, ego_vehicle, background_vehicles]
   # ... other settings used by the BEV handler
```

The handler field specifies which handler implementation is used to manage that observation key. Then, users can simply enable this observation in the task settings.

```yaml
your_task_name:
  env:
    observation.enabled: [camera, collision, spectator, birdeye_view]
```

#### Environment & Observer Communications

One might need transfer information from the environements to a handler to compute their observations. E.g., a BEV handler might need a location to render the destination spot. These environment information can be accessed either through [WorldManager](https://car-dreamer.readthedocs.io/en/latest/api/toolkit.html#car_dreamer.toolkit.WorldManager) APIs, or through environment state management.

A `WorldManager` instance is passed in the handler during its initialization. The environment states are defined by an environment's `get_state()` API, and passed as parameters to handler's `get_observation()`.

```python
class MyHandler(BaseHandler):
    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        self._world = world

def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
    # Get the waypoints through environment states
    waypoints = env_state.get("waypoints")
    # Get actors through the world manager API
    actors = self._world.actors
    # ...

class MyEnv(CarlaBaseEnv):
    # ...
    def get_state(self):
        return {
            # Expose the waypoints through get_state()
            'waypoints': self.waypoints,
        }
```

## :computer: Visualization Tool

We stream observations, rewards, terminal conditions, and custom metrics to a web browser for each training session in real-time, making it easier to engineer rewards and debug.

<table style="margin-left: auto; margin-right: auto;">
  <tr>
    <td class="center-text">Visualization Server</td>
  </tr>
  <tr>
    <td><img src="https://ucd-dare.github.io/cardreamer.github.io/static/images/visualization.png" style="width: 100%"></td>
  </tr>
</table>

## :hammer: System

...

To easily customize your own driving tasks, and observation spaces, etc., please refer to our [CarDreamer API Documents](https://car-dreamer.readthedocs.io/en/latest/).

![CarDreamer](https://ucd-dare.github.io/cardreamer.github.io/static/images/CarDreamerSystem.png)

# :star2: Citation

If you find this repository useful, please cite this paper:

**[IEEE IoT paper link](https://ieeexplore.ieee.org/document/10714437)**
**[ArXiv paper link](https://arxiv.org/abs/2405.09111)**

```
@ARTICLE{10714437,
  author={Gao, Dechen and Cai, Shuangyu and Zhou, Hanchu and Wang, Hang and Soltani, Iman and Zhang, Junshan},
  journal={IEEE Internet of Things Journal},
  title={CarDreamer: Open-Source Learning Platform for World Model Based Autonomous Driving},
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Autonomous Driving;Reinforcement Learning;World Model},
  doi={10.1109/JIOT.2024.3479088}}

@article{CarDreamer2024,
  title = {{CarDreamer: Open-Source Learning Platform for World Model based Autonomous Driving}},
  author = {Dechen Gao, Shuangyu Cai, Hanchu Zhou, Hang Wang, Iman Soltani, Junshan Zhang},
  journal = {arXiv preprint arXiv:2405.09111},
  year = {2024},
  month = {May}
}
```

# Suppliment Material

## World model imagination

<p align="center">
  Birdeye view imagination
</p>
<img src="https://ucd-dare.github.io/cardreamer.github.io/static/gifs/right_turn_hard_pre_bev.gif">
<p align="center">
  Camera view imagination
</p>
<img src="https://ucd-dare.github.io/cardreamer.github.io/static/gifs/right_turn_hard_pre_camera.gif">
<p align="center">
  LiDAR view imagination
</p>
<img src="https://ucd-dare.github.io/cardreamer.github.io/static/gifs/right_turn_hard_pre_lidar.gif">

# 👥 Contributors

Special thanks to the community for your valuable contributions and support in making CarDreamer better for everyone!

<!-- readme: contributors -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/HanchuZhou">
                    <img src="https://avatars.githubusercontent.com/u/99316745?v=4" width="100;" alt="HanchuZhou"/>
                    <br />
                    <sub><b>Hanchu Zhou</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/gaodechen">
                    <img src="https://avatars.githubusercontent.com/u/2103562?v=4" width="100;" alt="gaodechen"/>
                    <br />
                    <sub><b>GaoDechen</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/tonycaisy">
                    <img src="https://avatars.githubusercontent.com/u/92793139?v=4" width="100;" alt="tonycaisy"/>
                    <br />
                    <sub><b>Shuangyu Cai</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/junshanzhangJZ2080">
                    <img src="https://avatars.githubusercontent.com/u/111560343?v=4" width="100;" alt="junshanzhangJZ2080"/>
                    <br />
                    <sub><b>junshanzhangJZ2080</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/HeyGF">
                    <img src="https://avatars.githubusercontent.com/u/23623353?v=4" width="100;" alt="HeyGF"/>
                    <br />
                    <sub><b>Gaofeng Dong</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ustcmike">
                    <img src="https://avatars.githubusercontent.com/u/32145615?v=4" width="100;" alt="ustcmike"/>
                    <br />
                    <sub><b>ucdmike</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/JungDongwon">
                    <img src="https://avatars.githubusercontent.com/u/28348839?v=4" width="100;" alt="JungDongwon"/>
                    <br />
                    <sub><b>Jung Dongwon</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/MICH3LL3D">
                    <img src="https://avatars.githubusercontent.com/u/99102209?v=4" width="100;" alt="MICH3LL3D"/>
                    <br />
                    <sub><b>MICH3LL3D</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/andrewcwlee">
                    <img src="https://avatars.githubusercontent.com/u/31760595?v=4" width="100;" alt="andrewcwlee"/>
                    <br />
                    <sub><b>Andrew Lee</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/IanGuangleiZhu">
                    <img src="https://avatars.githubusercontent.com/u/91163000?v=4" width="100;" alt="IanGuangleiZhu"/>
                    <br />
                    <sub><b>IanGuangleiZhu</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/liamjxu">
                    <img src="https://avatars.githubusercontent.com/u/48697394?v=4" width="100;" alt="liamjxu"/>
                    <br />
                    <sub><b>liamjxu</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/TracyYXChen">
                    <img src="https://avatars.githubusercontent.com/u/31624007?v=4" width="100;" alt="TracyYXChen"/>
                    <br />
                    <sub><b>TracyYXChen</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/swsamleo">
                    <img src="https://avatars.githubusercontent.com/u/12550596?v=4" width="100;" alt="swsamleo"/>
                    <br />
                    <sub><b>Wei Shao</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: contributors -end -->

### How to Contribute?

The contributor list is automatically generated based on the commit history. Please use `pre-commit` to automatically check and format changes.

```bash
# Setup pre-commit tool
pip install pre-commit
pre-commit install
# Run pre-commit
pre-commit run --all-files
```

### Credits

`CarDreamer` builds on the several projects within the autonomous driving and machine learning communities.

- [gym-carla](https://github.com/cjy1992/gym-carla)
- [DreamerV2](https://github.com/danijar/director)
- [DreamerV3](https://github.com/danijar/dreamerv3)
- [CuriousReplay](https://github.com/AutonomousAgentsLab/curiousreplay)
