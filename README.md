# CarDreamer

![CarDreamer](.assets/banner.png)

This repository is composed of two parts: a package ``car_dreamer`` and several backbone RL algorithms such as DreamerV2 and DreamerV3.

Package ``car_dreamer`` is a collection of tasks aimed at facilitating RL algorithm designing, especially world model based ones. Each task is a certain driving environment in Carla simulator, varying from a single skill such as lane following or left turning, to random roaming in mixed road conditions which may encounter crossroads, roundabouts, and stop signs. They expose the same gym interface for backbone RL algorithm use.

Furthermore, ``car_dreamer`` includes a task development suite for those who want to customize their own tasks. It provides a number of API calls to minimize users' efforts in spawning the vehicles and obtaining observation data as RL algorithm inputs. And an integrated traning visualization server automatically grasps the observation data, displaying the videos and plotting the statistics through an HTTP server. This eases reward engineering, algorithm designing and hyper-parameter tuning.

Detailed guides on how to use this package can be found in this [document](https://car-dreamer.readthedocs.io/en/latest/).

# Prerequisites

Clone the repository:

```bash
git clone https://github.com/labdare/car-dreamer
cd car-dreamer
```

Download [Carla release](https://github.com/carla-simulator/carla/releases) of version ``0.9.15`` as we experiemented with this version. Set the following environment variables:

```bash
export CARLA_ROOT="</path/to/carla>"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}
```

Install the package using flit. The ``--symlink`` flag is used to create a symlink to the package in the Python environment, so that changes to the package are immediately available without reinstallation. (``--pth-file`` also works, as an alternative to ``--symlink``.)

```bash
pip install flit
flit install --symlink
```

# Training

Find ``README.md`` in the corresponding directory of the algorithm you want to use and follow the instructions.

# Citation
TODO
If you find this repository useful, please cite this [paper](https://localhost):
```
@article{CarDreamer2022,
  title={CarDreamer:},
  author={},
  journal={},
  year={}
}
```

Credits:
- [gym-carla](https://github.com/cjy1992/gym-carla)
- [DreamerV2](https://github.com/danijar/director)
- [DreamerV3](https://github.com/danijar/dreamerv3)

# 👥 Contributors

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