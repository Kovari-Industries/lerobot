import sagemaker
from sagemaker.estimator import Estimator
import argparse
import sys
import os

def launch_training_job():
    parser = argparse.ArgumentParser(description="Launch LeRobot Training on SageMaker")
    parser.add_argument("--config", type=str, default="configs/v126.yaml", help="Path to config file (relative to repo root)")
    args = parser.parse_args()

    print("Initializing SageMaker Session...")
    session = sagemaker.Session()
    role = "arn:aws:iam::815254799754:role/dev-sagemaker-jobs"
    print(f"Using Role: {role}")
    
    # ECR Image Uri
    image_uri = "815254799754.dkr.ecr.us-west-2.amazonaws.com/dev-lerobot-training:latest"
    
    # Instance configuration
    instance_type = "ml.g5.2xlarge"
    instance_count = 1
    
    print(f"Creating Estimator for image: {image_uri}")
    # We pass the config path as a hyperparameter. 
    # Our entrypoint script will read this and start training with it.
    hyperparameters = {
        "config_path": args.config,
        "output_dir": "/opt/ml/model"
    }
    
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        max_run=86400 * 5, # 5 days max
        output_path=f"s3://{session.default_bucket()}/lerobot-training",
        sagemaker_session=session,
        hyperparameters=hyperparameters,
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "WANDB_PROJECT": "lerobot",
        }
    )
    
    print(f"Launching training job on {instance_type} with config: {args.config}")
    estimator.fit(
        inputs=None, 
        wait=False,
        job_name=f"lerobot-coffee-{sagemaker.utils.unique_name_from_base('train')}"
    )
    
    print(f"Job launched successfully! Training outputs will be synced to s3://{session.default_bucket()}/lerobot-training/[job-name]/output/model.tar.gz")

if __name__ == "__main__":
    launch_training_job()