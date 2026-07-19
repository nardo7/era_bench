#!/usr/bin/env python3
"""Create missing per-demo MP4 byte datasets from RGB frames in an ERA Bench HDF5 dataset."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import re

import h5py
import imageio.v2 as imageio
import numpy as np


DEFAULT_DATASET = Path("datasets/era_bench.hdf5")
DEFAULT_FPS = 30
DEFAULT_CAMERA = "table_cam"


def _demo_sort_key(name: str) -> tuple[int, str]:
    """Sort demo_2 before demo_10 while still accepting other group names."""
    match = re.fullmatch(r"demo_(\d+)", name)
    return (int(match.group(1)), name) if match else (2**63 - 1, name)


def _as_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    """Convert an HDF5 frame array to imageio-compatible uint8 RGB frames."""
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim != 4:
        raise ValueError(f"expected frames with shape (T, H, W, C), got {frames.shape}")
    if frames.shape[-1] not in (3, 4):
        raise ValueError(f"expected RGB or RGBA frames, got {frames.shape}")

    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0.0, 1.0) * 255.0 if np.issubdtype(frames.dtype, np.floating) else frames
        frames = np.asarray(frames, dtype=np.uint8)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    return frames


def _encode_video(frames: np.ndarray, fps: int) -> np.ndarray:
    """Encode RGB frames using the same settings as ``EpisodeVideoRecorder``."""
    buffer = io.BytesIO()
    writer = imageio.get_writer(
        buffer,
        format="mp4",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=28,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def create_videos(
    dataset_path: Path,
    camera: str = DEFAULT_CAMERA,
    fps: int = DEFAULT_FPS,
) -> tuple[int, int]:
    """Store missing demo videos in the HDF5 file and return ``(created, skipped)`` counts."""
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {dataset_path}")

    created = 0
    skipped = 0

    with h5py.File(dataset_path, "r+") as dataset:
        data_group = dataset.get("data")
        if not isinstance(data_group, h5py.Group):
            raise ValueError(f"dataset has no /data group: {dataset_path}")

        demo_names = sorted(
            (name for name in data_group if isinstance(data_group[name], h5py.Group)),
            key=_demo_sort_key,
        )
        if not demo_names:
            raise ValueError(f"dataset contains no demo groups under /data: {dataset_path}")

        for demo_name in demo_names:
            demo_group = data_group[demo_name]
            if "video" in demo_group:
                print(f"[skip] /data/{demo_name}/video already exists")
                skipped += 1
                continue

            frames_dataset = demo_group.get(f"obs/{camera}")
            if not isinstance(frames_dataset, h5py.Dataset):
                print(f"[skip] /data/{demo_name}/obs/{camera} is missing")
                skipped += 1
                continue

            frames = _as_uint8_rgb(frames_dataset[()])
            if frames.shape[0] == 0:
                print(f"[skip] /data/{demo_name}/obs/{camera} contains no frames")
                skipped += 1
                continue

            print(f"[write] /data/{demo_name}/video ({frames.shape[0]} frames, {frames.shape[1:3]} pixels)")
            mp4_bytes = _encode_video(frames, fps)
            demo_group.create_dataset("video", data=mp4_bytes[None, :], compression="gzip")
            created += 1

    return created, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"ERA Bench HDF5 dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=f"Camera group under each demo's obs group (default: {DEFAULT_CAMERA})",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"Video FPS (default: {DEFAULT_FPS})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created, skipped = create_videos(args.dataset, camera=args.camera, fps=args.fps)
    print(f"Done: stored {created} video(s) in the dataset, skipped {skipped} existing/missing episode(s).")


if __name__ == "__main__":
    main()
