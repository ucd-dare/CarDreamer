#!/bin/bash

# Check if a port argument is provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <carla_port> <gpu_device> <checkpoint_path> [additional_eval_parameters]"
    exit 1
fi

# Configuration
CARLA_PORT=$1
GPU_DEVICE=$2
CHECKPOINT_PATH=$3
LOG_FILE="eval_log_${CARLA_PORT}.log"
CARLA_SERVER_COMMAND="$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT -benchmark -fps=10"
EVAL_SCRIPT="dreamerv3/eval.py"
COMMON_PARAMS="--env.world.carla_port $CARLA_PORT --dreamerv3.jax.policy_devices $GPU_DEVICE --dreamerv3.run.from_checkpoint $CHECKPOINT_PATH"
ADDITIONAL_PARAMS="${@:4}"  # Capture all additional parameters passed to the script
EVAL_COMMAND="python -u $EVAL_SCRIPT $COMMON_PARAMS $ADDITIONAL_PARAMS"

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

# Function to start the eval script
start_eval() {
    launch_carla
    # Start the eval script
    $EVAL_COMMAND >> $LOG_FILE 2>&1 &
    EVAL_PID=$!
    # Log the information about the log file
    log_with_timestamp "Eval session started successfully. Logs are being written to: $LOG_FILE"
    echo -e "\033[1;32mEval session started successfully. Logs are being written to: $LOG_FILE\033[0m"
}

# Function to clean up processes on exit
cleanup() {
    log_with_timestamp "Cleaning up and exiting..."
    # Kill CARLA process
    fuser -k ${CARLA_PORT}/tcp
    # Kill the specific eval process using its PID
    kill -TERM $EVAL_PID >/dev/null 2>&1
    wait $EVAL_PID >/dev/null 2>&1
    exit
}

# Trap EXIT signal to call the cleanup function
trap cleanup SIGINT

# Initial start
log_with_timestamp "Starting eval on port $CARLA_PORT..."
log_with_timestamp "Eval command: $EVAL_COMMAND"
start_eval

while true; do
    # Check if the eval script is still running
    if ! pgrep -f "$EVAL_SCRIPT" > /dev/null; then
        log_with_timestamp "Eval script crashed on port $CARLA_PORT. Restarting..."
        start_eval
    fi
    # Check if CARLA server needs to be restarted
    launch_carla
    # Check every minute
    sleep 60
done
