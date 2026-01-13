#!/usr/bin/env python3
"""
Fixed script to remove quaternion components from LeRobot dataset.
Preserves parquet schema for proper HuggingFace display.
"""

import os
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_REPO_ID = "dageorge1111/v133_coffee_pod_train"
TARGET_REPO_ID = "pans06/v133_coffee_pod_train_no_quaternions"
DATASET_PATH = Path.home() / "Documents/Kovari/lerobot_dataset"
DATA_DIR = DATASET_PATH / "data"
META_DIR = DATASET_PATH / "meta"

# Indices to keep: x(0), y(1), z(2), gripper_width(7)
INDICES_TO_KEEP = [0, 1, 2, 7]

def process_parquet_file(file_path):
    """Process a single parquet file preserving schema."""
    # Read with pyarrow to preserve schema
    table = pq.read_table(file_path)
    df = table.to_pandas()

    # Process observation.state column
    if 'observation.state' in df.columns:
        obs_state = np.stack(df['observation.state'].values)
        obs_state_new = obs_state[:, INDICES_TO_KEEP]
        # Convert back to list of arrays (preserves type)
        df['observation.state'] = [row for row in obs_state_new]

    # Process action column
    if 'action' in df.columns:
        action = np.stack(df['action'].values)
        action_new = action[:, INDICES_TO_KEEP]
        df['action'] = [row for row in action_new]

    # Save with pyarrow preserving schema
    # Get the original schema
    schema = table.schema

    # Update schema for the modified columns
    new_fields = []
    for field in schema:
        if field.name == 'observation.state' or field.name == 'action':
            # Keep the same type but the data is now smaller
            new_fields.append(field)
        else:
            new_fields.append(field)

    new_schema = pa.schema(new_fields)

    # Convert back to table and write
    new_table = pa.Table.from_pandas(df, schema=new_schema, preserve_index=False)
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

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

    print(f"Updated {info_path}")

def update_stats_json():
    """Update meta/stats.json by removing quaternion statistics."""
    stats_path = META_DIR / "stats.json"

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    # Update observation.state stats
    for stat_key in ['min', 'max', 'mean', 'std']:
        if stat_key in stats['observation.state']:
            old_vals = stats['observation.state'][stat_key]
            stats['observation.state'][stat_key] = [old_vals[i] for i in INDICES_TO_KEEP]

    # Update action stats
    for stat_key in ['min', 'max', 'mean', 'std']:
        if stat_key in stats['action']:
            old_vals = stats['action'][stat_key]
            stats['action'][stat_key] = [old_vals[i] for i in INDICES_TO_KEEP]

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Updated {stats_path}")

def main():
    """Main function to process entire dataset."""
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
    print(f"  - New format: [x, y, z, gripper_width]")
    print(f"  - Schema preserved for HuggingFace compatibility")

if __name__ == "__main__":
    main()
