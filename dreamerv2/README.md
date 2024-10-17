# Guide for Training DreamerV2 on CarDreaner

This guide assumes you have installed `car_dreamer`. If not, please follow the instructions in the [main README](../README.md).

First, install the required dependencies for DreamerV2:

```bash
cd dreamerv2
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
```

Set up CARLA and environment variables:

```bash
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export CUSOLVER_PATH=$(dirname $(python -c "import nvidia.cusolver;print(nvidia.cusolver.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CUSOLVER_PATH/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Training

Execute the training script with desired configurations:

```bash
cd ..
bash train_dm2.sh 2000 0 --task carla_four_lane --dreamerv2.logdir ./logdir/carla_four_lane
```

`2000` is the port number of the CARLA server. The script will automatically start the server so you don't need to start it manually.
`0` is the GPU number.
`--task` is the name of the task and `--dreamerv2.logdir` is the directory to save the training logs. For a complete list of tasks and their configurations, please refer to the [documentation](https://car-dreamer.readthedocs.io/en/latest/tasks.html).

## Visualization

Online data monitoring can be accessed on website on `http://localhost:9000/`, where the port number should be changed to `<carla-port> + 7000` if you don't use the default port number `2000` for CARLA server.

Offline data logging can be accessed through TensorBoard:

```bash
tensorboard --logdir ./logdir/carla_four_lane
```

Go to `http://localhost:6006/` in your browser to see the output.
