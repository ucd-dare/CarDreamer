#!/bin/bash

# Check if a port argument is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <carla_port> <gpu_device> [additional_training_parameters]"
    exit 1
fi

# Configuration
CARLA_PORT=$1
GPU_DEVICE=$2
LOG_FILE="log_${CARLA_PORT}.log"
CARLA_SERVER_COMMAND="$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT -benchmark -fps=10"
TRAINING_SCRIPT="dreamerv3/train.py"
COMMON_PARAMS="--env.world.carla_port $CARLA_PORT --dreamerv3.jax.policy_devices $GPU_DEVICE --dreamerv3.jax.train_devices $GPU_DEVICE"
ADDITIONAL_PARAMS="${@:3}"  # Capture all additional parameters passed to the script
TRAINING_COMMAND="python -u $TRAINING_SCRIPT $COMMON_PARAMS $ADDITIONAL_PARAMS"

# Clear log file before starting
> $LOG_FILE

# Function to log messages with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
}

# Function to start or restart CARLA
launch_carla() {
    # Check if CARLA is running
    if ! pgrep -f "CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT -benchmark -fps=10" > /dev/null; then
        log_with_timestamp "CARLA server is not running on port $CARLA_PORT. Starting or restarting..."
        # Kill any existing CARLA processes on the same port
        fuser -k ${CARLA_PORT}/tcp
        # Start CARLA
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE $CARLA_SERVER_COMMAND &
        # Wait for CARLA to fully start
        while ! nc -z localhost $CARLA_PORT; do
            log_with_timestamp "Waiting for CARLA server to start on port $CARLA_PORT..."
            sleep 1  # delay to prevent excessive resource usage
        done
        log_with_timestamp "CARLA server is up and running on port $CARLA_PORT."
    fi
}

# Function to start the training script
start_training() {
    launch_carla
    # Start the training script
    $TRAINING_COMMAND >> $LOG_FILE 2>&1 &
    TRAINING_PID=$!
    # Log the information about the log file
    log_with_timestamp "Training session started successfully. Logs are being written to: $LOG_FILE"
    echo -e "\033[1;32mTraining session started successfully. Logs are being written to: $LOG_FILE\033[0m"
}

# Function to clean up processes on exit
cleanup() {
    log_with_timestamp "Cleaning up and exiting..."
    # Kill CARLA process
    fuser -k ${CARLA_PORT}/tcp
    # Kill the specific training process using its PID
    kill -TERM $TRAINING_PID >/dev/null 2>&1
    wait $TRAINING_PID >/dev/null 2>&1
    exit
}

# Trap EXIT signal to call the cleanup function
trap cleanup SIGINT

# Initial start
log_with_timestamp "Starting training on port $CARLA_PORT..."
log_with_timestamp "Training command: $TRAINING_COMMAND"
start_training

while true; do
    # Check if the training script is still running
    if ! pgrep -f "$TRAINING_SCRIPT" > /dev/null; then
        log_with_timestamp "Training script crashed on port $CARLA_PORT. Restarting..."
        start_training
    fi
    # Check if CARLA server needs to be restarted
    launch_carla
    # Check every minute
    sleep 60
done
