TODO

Create a conda environment and install the required packages:
```bash
conda create --name dreamer python==3.10
conda activate dreamer
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Set up CARLA and environment variables:
```bash
export CARLA_ROOT="</path/to/carla>"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export CUSOLVER_PATH=$(dirname $(python -c "import nvidia.cusolver;print(nvidia.cusolver.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CUSOLVER_PATH/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

# Training

Execute the training script with desired configurations:
```bash
cd ..
bash train.sh 2000 0 --configs carla_navigation large --logdir ./logdir/carla_navigation
```
```carla_navigation``` is the name of the task. We provide the following tasks: ```carla_navigation```, ```carla_four_lane```, ```carla_text```.
The description of the tasks can be found in the corresponding files in ```gym-carla/gym_carla/envs/```.
```small``` is the default model size that has been tested on CarDreamer scenarios.

# Visualization

We provide online monitoring for real-time interaction videos and statistics (rewards, terminal, other information of interests) as well as the offline training logging.

Online data monitoring can be accessed through our web client.

Offline data logging can be accessed through TensorBoard or WandB that have been integrated with DreamerV2 and DreamerV3 models:
```bash
tensorboard --logdir ./logdir/carla_navigation
```
Go to ```http://localhost:6006/``` in your browser to see the output.

To use ```wandb``` for visualization, add the WandB logger to ```train.py```:
```python
logger = core.Logger(step, [
        # ...
        core.logger.WandBOutput(logdir.name, config),
    ])
```
Once you log in ```wandb```, put your project and entity name in logger:
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

```bash
bash eval_dm3.sh 1027 0 ~/logdir/path_to_your_checkpoint.ckpt --scenario scenario_name --dreamerv3.logdir ~/logdir/path_to_your_eval_logdir
```
