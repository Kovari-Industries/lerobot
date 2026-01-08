#!/bin/bash
set -e

# SageMaker automatically overrides the Docker CMD with "train" when launching a training job.
if [ "$1" = "train" ]; then
    shift
fi

# If arguments are provided directly (e.g. via Docker CMD or manuall run), use them.
if [ $# -gt 0 ]; then
    echo "Arguments provided, running: python src/lerobot/scripts/lerobot_train.py $@"
    exec python src/lerobot/scripts/lerobot_train.py "$@"
fi

# If no arguments provided, check for SageMaker hyperparameters.json
HP_FILE="/opt/ml/input/config/hyperparameters.json"

# Ensure /opt/ml/model is writable by the current user
if [ -d "/opt/ml/model" ]; then
    sudo chown -R user_lerobot:user_lerobot /opt/ml/model
fi

if [ -f "$HP_FILE" ]; then
    echo "Reading configuration from $HP_FILE..."
    
    # Parse config_path and output_dir from JSON using python
    # Note: SageMaker might store values as strings, even if we passed them as other types.
    CONFIG_ARGS=$(python -c "
import json
import os

hp_file = '$HP_FILE'
if os.path.exists(hp_file):
    with open(hp_file, 'r') as f:
        hp = json.load(f)
        
    config_path = hp.get('config_path', 'configs/v126.yaml')
    output_dir = hp.get('output_dir', '/opt/ml/model')
    
    print(f'--config_path {config_path} --output_dir {output_dir}')
else:
    # Fallback default if parsing fails for some reason
    print('--config_path configs/v126.yaml --output_dir /opt/ml/model')
")
    echo "Constructed arguments: $CONFIG_ARGS"
    exec python src/lerobot/scripts/lerobot_train.py $CONFIG_ARGS

else
    # Fallback to hardcoded default if not running in SageMaker or file missing
    echo "No arguments and no SageMaker config found. Using default."
    exec python src/lerobot/scripts/lerobot_train.py --config_path configs/v126.yaml --output_dir /opt/ml/model
fi
