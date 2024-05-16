# 🌍 Learn to Drive in "Dreams": CarDreamer 🚗

## **Can neural networks imagine traffic dynamics for training autonomous driving agents? The answer is YES!**

Using the high-fidelity CARLA simulator, we are able to train a world model that not only learns complex environment dynamics but also have an agent interact with the neural network "simulator" to learn to drive.

This means the agent learns to drive in a "dream world" from scratch, mastering maneuvers like overtaking and right turns, and avoiding collisions in heavy traffic—all within an imagined world!

Dive into our demos to see the agent skillfully navigating challenges and ensuring safe and efficient travel.

## 📚 Open-Source World Model-Based Autonomous Driving Platform

**Explore** world model based autonomous driving with CarDreamer, an open-source platform designed for the **development** and **evaluation** of **world model** based autonomous driving.

* 🏙️ **Built-in Urban Driving Tasks**: flexible and customizable observation modality and observability; optimized rewards
* 🔧 **Task Development Suite**: create your own urban driving tasks with ease
* 🌍 **Model Backbones**: integrated state-of-the-art world models

**Documentation:** [CarDreamer API Documents](https://car-dreamer.readthedocs.io/en/latest/).


<table style="margin-left: auto; margin-right: auto;">
  <tr>
    <td class="center-text">Right turn hard</td>
    <td class="center-text">Roundabout</td>
    <td class="center-text">Left turn hard</td>
    <td class="center-text">Lane merge</td>
    <td class="center-text">Overtake</td>
  </tr>
  <tr>
    <td><img src="./.assets/right_turn_hard.gif" style="width: 100%"></td>
    <td><img src="./.assets/roundabout.gif" style="width: 100%"></td>
    <td><img src="./.assets/left_turn_hard.gif" style="width: 100%"></td>
    <td><img src="./.assets/lane_merge.gif" style="width: 100%"></td>
    <td><img src="./.assets/overtake.gif" style="width: 100%"></td>
  </tr>
</table>

<table style="margin-left: auto; margin-right: auto;">
  <tr>
    <td class="center-text">Right turn hard</td>
    <td class="center-text">Roundabout</td>
    <td class="center-text">Left turn hard</td>
    <td class="center-text">Lane merge</td>
    <td class="center-text">Right turn simple</td>
  </tr>
  <tr>
    <td><img src="./.assets/right_turn_hard_camera.gif" style="width: 100%"></td>
    <td><img src="./.assets/roundabout_camera.gif" style="width: 100%"></td>
    <td><img src="./.assets/left_turn_hard_camera.gif" style="width: 100%"></td>
    <td><img src="./.assets/lane_merge_camera.gif" style="width: 100%"></td>
    <td><img src="./.assets/right_turn_simple_camera.gif" style="width: 100%"></td>
  </tr>
</table>

![CarDreamer](.assets/architecture.png)



# 📋 Prerequisites

Clone the repository:

```bash
git clone https://github.com/labdare/CarDreamer
cd CarDreamer
```

Download [CARLA release](https://github.com/carla-simulator/carla/releases) of version ``0.9.15`` as we experiemented with this version. Set the following environment variables:

```bash
export CARLA_ROOT="</path/to/carla>"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}
```

Install the package using flit. The ``--symlink`` flag is used to create a symlink to the package in the Python environment, so that changes to the package are immediately available without reinstallation. (``--pth-file`` also works, as an alternative to ``--symlink``.)

```bash
pip install flit
flit install --symlink
```

## Training

Find ``README.md`` in the corresponding directory of the algorithm you want to use and follow the instructions.

## Citation

If you find this repository useful, please cite this paper:
```
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
  Birdeye view training
</p>
<img src="./.assets/right_turn_hard_pre_bev.gif">
<p align="center">
  Camera view training
</p>
<img src="./.assets/right_turn_hard_pre_camera.gif">
<p align="center">
  LiDAR view training
</p>
<img src="./.assets/right_turn_hard_pre_lidar.gif">


# 👥 Contributors

### Credits

`CarDreamer` builds on the several amazing projects within the autonomous driving and machine learning communities.

- [gym-carla](https://github.com/cjy1992/gym-carla)
- [DreamerV2](https://github.com/danijar/director)
- [DreamerV3](https://github.com/danijar/dreamerv3)
- [CuriousReplay](https://github.com/AutonomousAgentsLab/curiousreplay)

<!-- readme: contributors -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/tonycaisy">
                    <img src="https://avatars.githubusercontent.com/u/92793139?v=4" width="100;" alt="tonycaisy"/>
                    <br />
                    <sub><b>Shuangyu Cai</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ustcmike">
                    <img src="https://avatars.githubusercontent.com/u/32145615?v=4" width="100;" alt="ustcmike"/>
                    <br />
                    <sub><b>ucdmike</b></sub>
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
                <a href="https://github.com/HanchuZhou">
                    <img src="https://avatars.githubusercontent.com/u/99316745?v=4" width="100;" alt="HanchuZhou"/>
                    <br />
                    <sub><b>Hanchu Zhou</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: contributors -end -->
