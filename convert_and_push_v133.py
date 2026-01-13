#!/usr/bin/env python3
"""
Convert v2.1 dataset from S3 to LeRobot v3.0 format and push to HuggingFace.

Usage:
    python convert_and_push_v133.py --repo-id dageorge1111/v133_larger_img_size
"""

import argparse
import shutil
from pathlib import Path

# The conversion script expects a specific directory structure
# We need to set up the local path correctly

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="pans06/v133_larger_img_size",
                        help="HuggingFace repo ID to push to")
    parser.add_argument("--local-dir", type=str, default="./temp_dataset_inspect",
                        help="Local directory containing the v2.1 dataset")
    parser.add_argument("--push-to-hub", action="store_true", default=True,
                        help="Push to HuggingFace after conversion")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    repo_id = args.repo_id

    # The converter expects the dataset at ~/.cache/huggingface/lerobot/{repo_id}
    # OR we can use --root to specify a custom location

    # Create target directory structure
    user, dataset_name = repo_id.split("/")
    target_root = Path("./converted_datasets")
    target_dir = target_root / repo_id

    if target_dir.exists():
        print(f"Removing existing {target_dir}")
        shutil.rmtree(target_dir)

    # Copy the v2.1 data to the expected location
    print(f"Copying {local_dir} to {target_dir}")
    shutil.copytree(local_dir, target_dir)

    # Now run the conversion (without push, we'll push manually)
    print(f"\nConverting dataset to v3.0 format...")
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

    convert_dataset(
        repo_id=repo_id,
        root=str(target_root),
        push_to_hub=False,  # Don't push yet, we need to create repo first
        force_conversion=True,
    )

    if args.push_to_hub:
        print(f"\nPushing to HuggingFace...")
        from huggingface_hub import HfApi

        api = HfApi()

        # Create the repo if it doesn't exist
        try:
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
            print(f"Created/verified repo: {repo_id}")
        except Exception as e:
            print(f"Note: {e}")

        # Upload the entire converted dataset folder
        print(f"Uploading from {target_dir}...")
        api.upload_folder(
            folder_path=str(target_dir),
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Tag with v3.0
        try:
            api.create_tag(repo_id, tag="v3.0", repo_type="dataset")
            print("Tagged with v3.0")
        except Exception as e:
            print(f"Tag note: {e}")

        print(f"\nDone! Dataset pushed to: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
