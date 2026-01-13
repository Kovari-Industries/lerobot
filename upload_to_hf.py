#!/usr/bin/env python3
"""
Script to upload LeRobot dataset to Hugging Face Hub.
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path

# Configuration
DATASET_PATH = Path.home() / "Documents/Kovari/lerobot_dataset"
REPO_NAME = "v133_coffee_pod_train_no_quaternions"  # Change this to your desired repo name
USERNAME = "pans06"  # Your HuggingFace username

def upload_dataset():
    """Upload dataset to Hugging Face Hub."""

    repo_id = f"{USERNAME}/{REPO_NAME}"

    print(f"Uploading dataset to: {repo_id}")
    print(f"Dataset path: {DATASET_PATH}")

    # Initialize HF API
    api = HfApi()

    # Create repository (will skip if already exists)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True
        )
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Upload the entire dataset folder
    print("\nUploading files...")
    api.upload_folder(
        folder_path=str(DATASET_PATH),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=[".cache/*", "*.pyc", "__pycache__/*"]
    )

    print(f"\n✓ Upload complete!")
    print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    print("=" * 60)
    print("Hugging Face Dataset Upload")
    print("=" * 60)
    print(f"\nRepo ID: {USERNAME}/{REPO_NAME}")
    print(f"Dataset: {DATASET_PATH}")
    print("\nThis will upload your dataset to Hugging Face Hub.")
    print("Make sure you're logged in with: huggingface-cli login")
    print("=" * 60)

    response = input("\nProceed with upload? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        upload_dataset()
    else:
        print("Upload cancelled.")
