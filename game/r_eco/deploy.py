#!/usr/bin/env python3
"""Deploy Hearts to a Hugging Face Space.

Usage:
    python deploy.py              # space name defaults to "hearts"
    python deploy.py --name my-hearts
"""

import argparse
import glob
import os
import sys


SOURCE_FILES = [
    "server.py",
    "app.py",
    "hearts_env.py",
    "obs_encoder.py",
    "ppo.py",
    "vec_env.py",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    "static/index.html",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="hearts", help="Space repo name (default: hearts)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()  # uses credentials from `huggingface-cli login`
    username = api.whoami()["name"]
    repo_id = f"{username}/{args.name}"
    base = os.path.dirname(os.path.abspath(__file__))

    print(f"Deploying to {repo_id} ...")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    for rel in SOURCE_FILES:
        path = os.path.join(base, rel)
        if not os.path.exists(path):
            print(f"  skip (not found): {rel}")
            continue
        print(f"  uploading: {rel}")
        api.upload_file(path_or_fileobj=path, path_in_repo=rel,
                        repo_id=repo_id, repo_type="space")

    pkt_files = glob.glob(os.path.join(base, "model", "*.pkt"))
    if pkt_files:
        newest = max(pkt_files, key=os.path.getmtime)
        dest = "model/" + os.path.basename(newest)
        print(f"  uploading model: {os.path.basename(newest)}")
        api.upload_file(path_or_fileobj=newest, path_in_repo=dest,
                        repo_id=repo_id, repo_type="space")
    else:
        print("  warning: no .pkt model files found, trained agent will not be available")

    print(f"\nDone! https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
