import sagemaker
from sagemaker.estimator import Estimator
import boto3

def launch_training_job():
    print("Initializing SageMaker Session...")
    # Explicitly set the region to match your ECR and S3 infrastructure
    region = "us-west-2"
    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    
    # Hardcoded role since we are launching from a local machine
    role = "arn:aws:iam::815254799754:role/dev-sagemaker-jobs"
    print(f"Using Role: {role}")
    
    # ECR Image Uri
    image_uri = "815254799754.dkr.ecr.us-west-2.amazonaws.com/dev-lerobot-training:latest"
    
    # Instance configuration
    # ml.g5.2xlarge = 1x NVIDIA A10G (24GB VRAM)
    instance_type = "ml.g5.2xlarge"
    instance_count = 1
    
    print(f"Creating Estimator for image: {image_uri}")
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        max_run=86400 * 5, 
        output_path=f"s3://{session.default_bucket()}/lerobot-training",
        sagemaker_session=session
    )
    
    print(f"Launching training job on {instance_type}...")
    # inputs={} is used if you aren't using SageMaker's native data channels
    # (e.g. if your container pulls data directly from Hugging Face or S3 internally)
    estimator.fit(
        inputs=None, 
        wait=False,
        job_name=f"lerobot-coffee-{sagemaker.utils.unique_name_from_base('train')}"
    )
    
    print(f"Job launched successfully!")
    print(f"View progress in Console: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs")

if __name__ == "__main__":
    launch_training_job()