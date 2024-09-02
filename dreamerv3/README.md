# Guide for Training DreamerV3 on CarDreamer

This guide assumes you have installed `car_dreamer`. If not, please follow the instructions in the [main README](../README.md).

First, install the required dependencies for DreamerV3:

```bash
cd dreamerv3
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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
bash train_dm3.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/carla_four_lane
```

`2000` is the port number of the CARLA server. The script will automatically start the server so you don't need to start it manually.
`0` is the GPU number.
`--task` is the name of the task and `--dreamerv3.logdir` is the directory to save the training logs. For a complete list of tasks and their configurations, please refer to the [documentation](https://car-dreamer.readthedocs.io/en/latest/tasks.html).

## Visualization

Online data monitoring can be accessed on website on `http://localhost:9000/`, where the port number should be changed to `<carla-port> + 7000` if you don't use the default port number `2000` for CARLA server.

Offline data logging can be accessed through TensorBoard or WandB:

```bash
tensorboard --logdir ./logdir/carla_four_lane
```

Go to `http://localhost:6006/` in your browser to see the output.

To use `wandb` for visualization, add the WandB logger to `dreamerv3/train.py`:

```python
logger = embodied.Logger(step, [
    # ...
    embodied.logger.WandBOutput(logdir.name, config),
])
```

Once you log in `wandb`, put your project and entity name in `dreamerv3/embodied/logger.py`:

```python
class WandBOutput:

  def __init__(self, run_name, config, pattern=r'.*'):
    self._pattern = re.compile(pattern)
    import wandb
    wandb.init(
        project="project_name",
        name=run_name,
        entity='entity_name',
        config=dict(config),
    )
    self._wandb = wandb
```

# Evaluation

Run the following commands to evaluate the trained model where the third argument is the path to the checkpoint:

```bash
bash eval_dm3.sh 2000 0 ./logdir/carla_four_lane/checkpoint.ckpt --task carla_four_lane --dreamerv3.logdir ./logdir/eval_carla_four_lane
```

After running for some episodes, you can visualize the evaluation results using TensorBoard or WandB as described above. Furthermore, you can get the evaluation metrics by running the following command:

```bash
python dreamerv3/eval_stats.py --logdir ./logdir/eval_carla_four_lane
```
