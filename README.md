# 🌍🚗 CarDreamer: Open-Source World Model-Based Autonomous Driving Platform 🚗🌍

**Explore** world model based autonomous driving with CarDreamer, an open-source platform designed for the **development** and **evaluation** of **world model** based autonomous driving.

* 🏙️ **Built-in Urban Driving Tasks**: flexible and customizable observation modality and observability; optimized rewards
* 🔧 **Task Development Suite**: create your own urban driving tasks with ease
* 🌍 **Model Backbones**: integrated state-of-the-art world models

**Documentation:** [CarDreamer API Documents](https://car-dreamer.readthedocs.io/en/latest/).

**Technical Report** ArXiv Preprint (TODO)

![CarDreamer](.assets/banner.png)

## Overveiw

`CarDreamer` is a platform designed for world model based autonomous driving, featuring a set of urban driving tasks within the realistic CARLA simulator. Tasks range from basic maneuvers, such as lane following, to complex navigation in varied road conditions. All tasks are integrated with [OpenAI Gym](https://gymnasium.farama.org/) interfaces, enabling straightforward evaluation of algorithms.

The platform includes decoupled data handlers and an observer to manage multi-modal observations, allowing users to easily customize modality and observability. The development suite aims at facilitating creation of new urban driving tasks.

## Prerequisites

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

If you find this repository useful, please cite this paper (TODO)
```
@article{CarDreamer2024,
  title={CarDreamer:},
  author={},
  journal={},
  year={}
}
```

## 👥 Contributors

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
                    <sub><b>tonycaisy</b></sub>
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