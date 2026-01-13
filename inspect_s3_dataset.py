#!/usr/bin/env python3
"""
Script to download and inspect the S3 parquet dataset structure.
Run this first to understand the data format before conversion.
"""

import subprocess
import os
import json
from pathlib import Path

# Configuration
S3_URI = "s3://dev-data-46eb/datasets/embodiment=evo-v1/v133_larger_img_size/"
LOCAL_DIR = "./temp_dataset_inspect"

def download_from_s3():
    """Download the dataset from S3"""
    print(f"Downloading from {S3_URI} to {LOCAL_DIR}...")
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # Use aws s3 sync to download
    cmd = ["aws", "s3", "sync", S3_URI, LOCAL_DIR]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error downloading: {result.stderr}")
        return False

    print("Download complete!")
    return True

def inspect_structure():
    """Inspect the downloaded directory structure"""
    print("\n" + "="*60)
    print("DIRECTORY STRUCTURE:")
    print("="*60)

    for root, dirs, files in os.walk(LOCAL_DIR):
        level = root.replace(LOCAL_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Limit files shown
            print(f'{subindent}{file}')
        if len(files) > 10:
            print(f'{subindent}... and {len(files) - 10} more files')

def inspect_parquet_files():
    """Inspect parquet file schemas"""
    import pyarrow.parquet as pq

    print("\n" + "="*60)
    print("PARQUET FILE SCHEMAS:")
    print("="*60)

    parquet_files = list(Path(LOCAL_DIR).rglob("*.parquet"))

    for pq_file in parquet_files[:3]:  # Inspect first 3 parquet files
        print(f"\n--- {pq_file.relative_to(LOCAL_DIR)} ---")
        try:
            table = pq.read_table(pq_file)
            print(f"Columns: {table.column_names}")
            print(f"Num rows: {table.num_rows}")
            print("\nSchema:")
            for field in table.schema:
                print(f"  {field.name}: {field.type}")

            # Show first row as sample
            print("\nFirst row sample:")
            df = table.to_pandas()
            for col in df.columns:
                val = df[col].iloc[0]
                if hasattr(val, '__len__') and not isinstance(val, str):
                    print(f"  {col}: {type(val).__name__} with len={len(val)}")
                else:
                    print(f"  {col}: {val}")
        except Exception as e:
            print(f"Error reading {pq_file}: {e}")

def inspect_json_files():
    """Inspect any JSON metadata files"""
    print("\n" + "="*60)
    print("JSON METADATA FILES:")
    print("="*60)

    json_files = list(Path(LOCAL_DIR).rglob("*.json"))

    for json_file in json_files:
        print(f"\n--- {json_file.relative_to(LOCAL_DIR)} ---")
        try:
            with open(json_file) as f:
                data = json.load(f)
            print(json.dumps(data, indent=2)[:2000])  # First 2000 chars
            if len(json.dumps(data)) > 2000:
                print("... (truncated)")
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

def inspect_video_files():
    """Check for video files"""
    print("\n" + "="*60)
    print("VIDEO FILES:")
    print("="*60)

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(LOCAL_DIR).rglob(f"*{ext}"))

    if video_files:
        print(f"Found {len(video_files)} video files:")
        for vf in video_files[:5]:
            print(f"  {vf.relative_to(LOCAL_DIR)}")
        if len(video_files) > 5:
            print(f"  ... and {len(video_files) - 5} more")
    else:
        print("No video files found (images may be embedded in parquet)")

if __name__ == "__main__":
    if download_from_s3():
        inspect_structure()
        inspect_json_files()
        inspect_parquet_files()
        inspect_video_files()

        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("Based on the output above, we can create a conversion script.")
        print(f"Data downloaded to: {os.path.abspath(LOCAL_DIR)}")
