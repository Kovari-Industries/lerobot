#!/usr/bin/env python3
"""
Script to download, process, and upload LeRobot dataset.
1. Downloads dataset from Hugging Face Hub
2. Transforms observation.state and action from [x, y, z, qx, qy, qz, qw, gripper_width]
   to [x, y, z, gripper_width].
3. Uploads processed dataset back to Hugging Face Hub
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download, HfApi, create_repo
import shutil

# Configuration
SOURCE_REPO_ID = "dageorge1111/v133_coffee_pod_train"  # Source dataset repo
TARGET_REPO_ID = "pans06/v133_coffee_pod_train_no_quaternions"  # Target dataset repo (change username/name as needed)
DATASET_PATH = Path.home() / "Documents/Kovari/lerobot_dataset"
DATA_DIR = DATASET_PATH / "data"
META_DIR = DATASET_PATH / "meta"

# Indices to keep: x(0), y(1), z(2), gripper_width(7)
INDICES_TO_KEEP = [0, 1, 2, 7]

def download_dataset_from_hf(repo_id, local_dir):
    """Download dataset from Hugging Face Hub."""
    print("=" * 60)
    print("STEP 1: Downloading Dataset from Hugging Face")
    print("=" * 60)
    print(f"Source repo: {repo_id}")
    print(f"Destination: {local_dir}")

    # Remove existing dataset if it exists
    if local_dir.exists():
        print(f"\nRemoving existing dataset at {local_dir}")
        shutil.rmtree(local_dir)

    # Download the dataset
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir)
    )

    print(f"\n✓ Dataset downloaded successfully\n")
    return True

def process_parquet_file(file_path):
    """Process a single parquet file to remove quaternion columns."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    # Read table to preserve metadata
    table = pq.read_table(file_path)
    df = table.to_pandas()

    # Process observation.state column
    if 'observation.state' in df.columns:
        obs_state = np.stack(df['observation.state'].values)
        obs_state_new = obs_state[:, INDICES_TO_KEEP]
        df['observation.state'] = list(obs_state_new)

    # Process action column
    if 'action' in df.columns:
        action = np.stack(df['action'].values)
        action_new = action[:, INDICES_TO_KEEP]
        df['action'] = list(action_new)

    # Convert back to table PRESERVING the original schema metadata
    new_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)

    # Preserve the huggingface metadata (critical for image display!)
    if table.schema.metadata:
        new_table = new_table.replace_schema_metadata(table.schema.metadata)

    # Save with metadata
    pq.write_table(new_table, file_path)
    return True

def update_info_json():
    """Update meta/info.json with new feature dimensions."""
    info_path = META_DIR / "info.json"

    with open(info_path, 'r') as f:
        info = json.load(f)

    # Update observation.state
    info['features']['observation.state']['shape'] = [4]
    info['features']['observation.state']['names'] = ["x", "y", "z", "gripper_width"]

    # Update action
    info['features']['action']['shape'] = [4]
    info['features']['action']['names'] = ["x", "y", "z", "gripper_width"]

    # Save updated info
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

    print(f"Updated {info_path}")

def update_stats_json():
    """Update meta/stats.json by removing quaternion statistics."""
    stats_path = META_DIR / "stats.json"

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    # Update observation.state stats - keep indices 0, 1, 2, 7
    for stat_key in ['min', 'max', 'mean', 'std']:
        if stat_key in stats['observation.state']:
            old_vals = stats['observation.state'][stat_key]
            stats['observation.state'][stat_key] = [old_vals[i] for i in INDICES_TO_KEEP]

    # Update action stats - keep indices 0, 1, 2, 7
    for stat_key in ['min', 'max', 'mean', 'std']:
        if stat_key in stats['action']:
            old_vals = stats['action'][stat_key]
            stats['action'][stat_key] = [old_vals[i] for i in INDICES_TO_KEEP]

    # Save updated stats
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Updated {stats_path}")

def upload_dataset_to_hf(repo_id, local_dir):
    """Upload processed dataset to Hugging Face Hub."""
    print("=" * 60)
    print("STEP 3: Uploading Dataset to Hugging Face")
    print("=" * 60)
    print(f"Target repo: {repo_id}")
    print(f"Dataset path: {local_dir}")

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
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=[".cache/*", "*.pyc", "__pycache__/*"]
    )

    print(f"\n✓ Upload complete!")

    # Create version tag (required for LeRobot training)
    print("\nCreating version tag...")
    info_path = Path(local_dir) / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)

    tag = info.get('codebase_version', 'v3.0')
    try:
        api.create_tag(
            repo_id=repo_id,
            tag=tag,
            repo_type="dataset"
        )
        print(f"✓ Created tag: {tag}")
    except Exception as e:
        print(f"Note: Tag creation - {e}")
        print("  (This is OK if tag already exists)")

    print(f"\nDataset URL: https://huggingface.co/datasets/{repo_id}")
    return True

def main():
    """Main function to download, process, and upload dataset."""
    print("\n" + "=" * 60)
    print("LeRobot Dataset Processor")
    print("=" * 60)
    print(f"Source: {SOURCE_REPO_ID}")
    print(f"Target: {TARGET_REPO_ID}")
    print(f"Local path: {DATASET_PATH}")
    print("=" * 60 + "\n")

    # Step 1: Download dataset from HF
    download_dataset_from_hf(SOURCE_REPO_ID, DATASET_PATH)

    # Step 2: Process dataset
    print("=" * 60)
    print("STEP 2: Processing Dataset (Removing Quaternions)")
    print("=" * 60)
    print(f"Processing dataset at: {DATASET_PATH}")

    # Find all parquet files
    parquet_files = list(DATA_DIR.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files to process\n")

    # Process each parquet file
    for file_path in tqdm(parquet_files, desc="Processing parquet files"):
        process_parquet_file(file_path)

    print("\nUpdating metadata files...")
    update_info_json()
    update_stats_json()

    print("\n✓ Dataset transformation complete!")
    print(f"  - Removed quaternion components (qx, qy, qz, qw)")
    print(f"  - New format: [x, y, z, gripper_width]\n")

    # Step 3: Upload processed dataset to HF
    upload_dataset_to_hf(TARGET_REPO_ID, DATASET_PATH)

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
