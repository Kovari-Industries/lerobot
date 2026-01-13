#!/usr/bin/env python3
"""
Script to encode videos for LeRobot dataset to make images viewable on HuggingFace.
"""
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Path to your dataset
dataset_path = Path.home() / "Documents/Kovari/lerobot_dataset"

print(f"Loading dataset from: {dataset_path}")
print("This will create video files from the parquet data...")

# Load the dataset
dataset = LeRobotDataset(str(dataset_path))

# Encode videos
print("\nEncoding videos (this may take a few minutes)...")
dataset.consolidate(run_compute_stats=False)

print("\n✓ Videos encoded successfully!")
print(f"Check {dataset_path}/videos/ for the video files")
