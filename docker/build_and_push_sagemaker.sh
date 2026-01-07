#!/bin/bash
set -e

# Configuration
REPO_NAME="dev-lerobot-training"
ACCOUNT_ID="815254799754"
REGION="us-west-2"
TAG="latest"

FULL_IMAGE_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}"

echo "Login to AWS ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "Building Docker image..."
# Build using the sagemaker dockerfile from the root of the repo
# Assumes script is run from project root or checks directory
if [ -f "docker/Dockerfile.sagemaker" ]; then
    DOCKERFILE="docker/Dockerfile.sagemaker"
    CONTEXT="."
elif [ -f "Dockerfile.sagemaker" ]; then
    DOCKERFILE="Dockerfile.sagemaker"
    CONTEXT=".."
else
    echo "Error: Could not find docker/Dockerfile.sagemaker. Run this from the repository root."
    exit 1
fi

docker build -f ${DOCKERFILE} -t ${REPO_NAME} ${CONTEXT}

echo "Tagging image..."
docker tag ${REPO_NAME} ${FULL_IMAGE_NAME}

echo "Pushing image to ECR..."
docker push ${FULL_IMAGE_NAME}

echo "Done! Image pushed to: ${FULL_IMAGE_NAME}"
