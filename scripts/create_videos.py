#!/usr/bin/env python3
"""Create missing per-demo MP4 files from RGB frames in an ERA Bench HDF5 dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import re
import tempfile

import h5py
import imageio.v2 as imageio
import numpy as np


DEFAULT_DATASET = Path("datasets/era_bench.hdf5")
DEFAULT_FPS = 30
DEFAULT_CAMERA = "table_cam"
DATA_TYPE_SUFFIXES = {
    "rgb": "",
    "depth": "_depth",
    "segmentation": "_semantic_segmentation",
}


def _demo_sort_key(name: str) -> tuple[int, str]:
    """Sort demo_2 before demo_10 while still accepting other group names."""
    match = re.fullmatch(r"demo_(\d+)", name)
    return (int(match.group(1)), name) if match else (2**63 - 1, name)


def _as_uint8_color_frames(frames: np.ndarray) -> np.ndarray:
    """Convert RGB or colorized semantic frames to imageio-compatible RGB."""
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


def _as_uint8_depth_frames(frames: np.ndarray) -> np.ndarray:
    """Normalize finite depth values across an episode and convert them to grayscale RGB."""
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim == 4 and frames.shape[-1] == 1:
        frames = frames[..., 0]
    if frames.ndim != 3:
        raise ValueError(f"expected depth frames with shape (T, H, W, 1), got {frames.shape}")

    finite = np.isfinite(frames)
    if not np.any(finite):
        raise ValueError("depth frames contain no finite values")

    finite_values = frames[finite]
    min_depth = finite_values.min()
    max_depth = finite_values.max()
    normalized = np.zeros(frames.shape, dtype=np.uint8)
    if max_depth > min_depth:
        normalized[finite] = np.clip(
            (frames[finite] - min_depth) / (max_depth - min_depth) * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
    return np.repeat(normalized[..., None], 3, axis=-1)


def _as_uint8_video_frames(frames: np.ndarray, data_type: str) -> np.ndarray:
    """Convert frames for the selected camera data type into uint8 RGB video frames."""
    if data_type == "depth":
        return _as_uint8_depth_frames(frames)
    return _as_uint8_color_frames(frames)


def _task_id(demo_group: h5py.Group) -> str | None:
    """Decode the recorder's ASCII uint8 task ID representation."""
    task_id_dataset = demo_group.get("task_info/task_id")
    if not isinstance(task_id_dataset, h5py.Dataset):
        return None
    task_id_bytes = np.asarray(task_id_dataset[()], dtype=np.uint8).reshape(-1).tobytes()
    return task_id_bytes.decode("ascii").rstrip("\x00")


def _task_pair(demo_group: h5py.Group) -> int | None:
    """Read the task-local pair index stored by the task recorder."""
    task_pair_dataset = demo_group.get("task_info/task_pair")
    if not isinstance(task_pair_dataset, h5py.Dataset):
        return None
    return int(np.asarray(task_pair_dataset[()]).reshape(-1)[-1])


def _output_path(output_dir: Path, demo_name: str, camera: str, data_type: str) -> Path:
    """Keep the original RGB/table-camera file name while avoiding camera/type collisions."""
    if camera == DEFAULT_CAMERA and data_type == "rgb":
        return output_dir / f"{demo_name}.mp4"
    safe_camera = re.sub(r"[^A-Za-z0-9_.-]+", "_", camera)
    return output_dir / f"{demo_name}_{safe_camera}_{data_type}.mp4"


def _write_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    """Write RGB frames using the same settings as ``EpisodeVideoRecorder``."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        writer = imageio.get_writer(
            temporary_path,
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
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def create_videos(
    dataset_path: Path,
    output_dir: Path,
    camera: str = DEFAULT_CAMERA,
    data_types: Sequence[str] = ("rgb",),
    task: str | None = None,
    pair: int | None = None,
    fps: int = DEFAULT_FPS,
) -> tuple[int, int]:
    """Create missing demo videos and return ``(created, skipped)`` counts."""
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    if not data_types:
        raise ValueError("at least one data type must be provided")
    unknown_data_types = set(data_types).difference(DATA_TYPE_SUFFIXES)
    if unknown_data_types:
        raise ValueError(f"unknown data type(s): {', '.join(sorted(unknown_data_types))}")
    if pair is not None and task is None:
        raise ValueError("--pair requires --task because pair indices are task-local")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {dataset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0
    selected = 0

    with h5py.File(dataset_path, "r") as dataset:
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
            if task is not None:
                demo_task = _task_id(demo_group)
                demo_pair = _task_pair(demo_group)
                if demo_task is None or demo_pair is None:
                    print(f"[skip] /data/{demo_name} has no task_info/task_id or task_info/task_pair")
                    skipped += 1
                    continue
                if demo_task != task or (pair is not None and demo_pair != pair):
                    continue
                selected += 1

            for data_type in data_types:
                output_path = _output_path(output_dir, demo_name, camera, data_type)
                if output_path.exists():
                    print(f"[skip] {output_path} already exists")
                    skipped += 1
                    continue

                frames_key = f"obs/{camera}{DATA_TYPE_SUFFIXES[data_type]}"
                frames_dataset = demo_group.get(frames_key)
                if not isinstance(frames_dataset, h5py.Dataset):
                    print(f"[skip] /data/{demo_name}/{frames_key} is missing")
                    skipped += 1
                    continue

                frames = _as_uint8_video_frames(frames_dataset[()], data_type)
                if frames.shape[0] == 0:
                    print(f"[skip] /data/{demo_name}/{frames_key} contains no frames")
                    skipped += 1
                    continue

                print(f"[write] {output_path} ({frames.shape[0]} frames, {frames.shape[1:3]} pixels)")
                _write_video(frames, output_path, fps)
                created += 1

    if task is not None and selected == 0:
        pair_description = f", pair {pair}" if pair is not None else ""
        raise ValueError(f"no recorded demo found for task {task}{pair_description}")

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
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for demo_<n>.mp4 files (default: <dataset-dir>/<dataset-stem>_videos)",
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=f"Camera name under each demo's obs group (default: {DEFAULT_CAMERA})",
    )
    parser.add_argument(
        "--data-type",
        "--type",
        dest="data_type_groups",
        choices=DATA_TYPE_SUFFIXES,
        action="append",
        nargs="+",
        metavar="TYPE",
        help="Camera data type(s) to render; may be repeated (default: rgb)",
    )
    parser.add_argument(
        "--task",
        help="Task ID to render; includes all of its recorded pairs unless --pair is also set",
    )
    parser.add_argument("--pair", type=int, help="Task-local pair index; requires --task")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"Video FPS (default: {DEFAULT_FPS})")
    args = parser.parse_args()
    if args.pair is not None and args.task is None:
        parser.error("--pair requires --task because pair indices are task-local")
    data_type_groups = args.data_type_groups or [["rgb"]]
    args.data_types = list(dict.fromkeys(data_type for group in data_type_groups for data_type in group))
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.dataset.parent / f"{args.dataset.stem}_videos"
    created, skipped = create_videos(
        args.dataset,
        output_dir,
        camera=args.camera,
        data_types=args.data_types,
        task=args.task,
        pair=args.pair,
        fps=args.fps,
    )
    print(f"Done: created {created} video(s), skipped {skipped} existing/missing episode(s).")


if __name__ == "__main__":
    main()
