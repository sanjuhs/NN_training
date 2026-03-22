#!/usr/bin/env python3
"""
Download one or more public Hugging Face video datasets to a local directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".webm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public Hugging Face dataset snapshots for video processing."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset repo id such as sanjuhs/ml_video_dataset. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory where downloaded snapshots should be stored.",
    )
    return parser.parse_args()


def repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def snapshot_download_with_retry(repo_id: str, local_dir: Path) -> str:
    try:
        return snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:
        cache_dir = local_dir / ".cache"
        if not cache_dir.exists():
            raise

        print(f"Download resume failed for {repo_id}: {exc}")
        print(f"Removing stale local Hugging Face cache under {cache_dir} and retrying once...")
        shutil.rmtree(cache_dir, ignore_errors=True)
        return snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {"datasets": []}
    for repo_id in args.dataset:
        local_dir = args.output_root / repo_slug(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Downloading {repo_id} ===")
        snapshot_path = snapshot_download_with_retry(repo_id, local_dir)

        files = [path for path in local_dir.rglob("*") if path.is_file()]
        video_files = [path for path in files if path.suffix.lower() in VIDEO_EXTENSIONS]
        record = {
            "repo_id": repo_id,
            "local_dir": str(local_dir),
            "snapshot_path": snapshot_path,
            "file_count": len(files),
            "video_count": len(video_files),
            "video_files": [str(path.relative_to(local_dir)) for path in sorted(video_files)],
        }
        manifest["datasets"].append(record)

        print(f"Stored at: {local_dir}")
        print(f"Files: {len(files)}")
        print(f"Videos: {len(video_files)}")

    manifest_path = args.output_root / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
