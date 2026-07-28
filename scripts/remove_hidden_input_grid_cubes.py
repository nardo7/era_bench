#!/usr/bin/env python3
"""Copy an ERA Bench dataset without hidden input-grid reference cubes.

Unused input-grid cube slots are reset to local ``z = -10``.  The recorder
stores those slots in both the initial state and every recorded ``states``
step, even though they are not visible in the scene.  This script removes the
corresponding ``rigid_object/input_grid_cube_*`` groups from both locations
when their recorded state poses are below ``--hidden-z-max``.  All non-input-grid
objects and all other HDF5 data are copied unchanged.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import tempfile

import h5py
import numpy as np


INPUT_GRID_CUBE_PREFIX = "input_grid_cube_"
ROOT_POSE_DATASET = "root_pose"
DEFAULT_HIDDEN_Z_MAX = -1.0


def _demo_sort_key(name: str) -> tuple[int, str]:
    """Sort ``demo_2`` before ``demo_10`` while accepting other group names."""
    match = re.fullmatch(r"demo_(\d+)", name)
    return (int(match.group(1)), name) if match else (2**63 - 1, name)


def is_hidden_input_grid_cube(cube_group: h5py.Group, hidden_z_max: float) -> bool:
    """Return whether the cube's recorded state pose is below the threshold.

    Input-grid cubes are kinematic and remain fixed throughout an episode, so
    reading the first state avoids loading their full per-step pose datasets.
    """
    root_pose = cube_group.get(ROOT_POSE_DATASET)
    if not isinstance(root_pose, h5py.Dataset):
        return False
    if root_pose.ndim == 0 or root_pose.shape[0] == 0 or root_pose.shape[-1] < 3:
        raise ValueError(f"{cube_group.name}/{ROOT_POSE_DATASET} must end in an xyz pose, got {root_pose.shape}")

    z_position = root_pose[0, ..., 2]
    return bool(np.all(np.isfinite(z_position)) and np.all(z_position <= hidden_z_max))


def clean_hidden_input_grid_cubes(
    dataset: h5py.File, hidden_z_max: float, dry_run: bool
) -> tuple[int, int, int]:
    """Scan state objects once and immediately remove hidden cubes when enabled."""
    data_group = dataset.get("data")
    if not isinstance(data_group, h5py.Group):
        raise ValueError(f"dataset has no /data group: {dataset.filename}")

    demo_names = sorted(data_group, key=_demo_sort_key)
    demos_with_removals = 0
    hidden_cube_count = 0
    total_cube_count = 0
    for demo_index, demo_name in enumerate(demo_names, start=1):
        demo_group = data_group[demo_name]
        if not isinstance(demo_group, h5py.Group):
            continue
        state_objects = demo_group.get("states/rigid_object")
        if not isinstance(state_objects, h5py.Group):
            continue

        cube_names = sorted(name for name in state_objects if name.startswith(INPUT_GRID_CUBE_PREFIX))
        demo_hidden_count = 0
        for cube_name in cube_names:
            cube_group = state_objects[cube_name]
            if not isinstance(cube_group, h5py.Group):
                continue
            total_cube_count += 1
            if not is_hidden_input_grid_cube(cube_group, hidden_z_max):
                continue

            demo_hidden_count += 1
            hidden_cube_count += 1
            if not dry_run:
                initial_objects = demo_group.get("initial_state/rigid_object")
                if isinstance(initial_objects, h5py.Group) and cube_name in initial_objects:
                    del initial_objects[cube_name]
                del state_objects[cube_name]

        if demo_hidden_count:
            demos_with_removals += 1
        print(
            f"[scan {demo_index}/{len(demo_names)}] {demo_name}: "
            f"{demo_hidden_count} hidden input-grid cube(s) out of {len(cube_names)}"
        )

    return demos_with_removals, hidden_cube_count, total_cube_count


def compact_hdf5_file(dataset_path: Path) -> None:
    """Repack the dataset so space from deleted HDF5 objects is released."""
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{dataset_path.stem}_repacked_", suffix=".hdf5", dir=dataset_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    size_before = dataset_path.stat().st_size
    print(f"[compact] Repacking {dataset_path}; this may take a while...", flush=True)
    try:
        with h5py.File(dataset_path, "r") as source, h5py.File(temporary_path, "w") as compacted:
            for attribute_name, value in source.attrs.items():
                compacted.attrs[attribute_name] = value
            for name in source:
                if name != "data":
                    source.copy(name, compacted)

            source_data = source["data"]
            compacted_data = compacted.create_group("data")
            for attribute_name, value in source_data.attrs.items():
                compacted_data.attrs[attribute_name] = value
            demo_names = sorted(source_data, key=_demo_sort_key)
            for demo_index, demo_name in enumerate(demo_names, start=1):
                source_data.copy(demo_name, compacted_data)
                print(f"[compact {demo_index}/{len(demo_names)}] {demo_name}", flush=True)
        shutil.copymode(dataset_path, temporary_path)
        os.replace(temporary_path, dataset_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    size_after = dataset_path.stat().st_size
    print(f"[compact] {size_before / 1024**3:.2f} GiB -> {size_after / 1024**3:.2f} GiB")


def remove_hidden_input_grid_cubes(
    input_path: Path, output_path: Path, hidden_z_max: float, dry_run: bool = False
) -> tuple[int, int, int]:
    """Copy ``input_path`` to ``output_path`` and remove hidden cube groups.

    Returns ``(demo_count_with_removals, hidden_cube_count, total_cube_count)``.
    The source is never modified.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {input_path}")
    dataset_path = input_path
    mode = "r"
    if not dry_run:
        if input_path.resolve() == output_path.resolve():
            raise ValueError("--output must differ from --input; the source dataset is never modified")
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite existing output dataset: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[copy] {input_path} -> {output_path}")
        shutil.copy2(input_path, output_path)
        dataset_path = output_path
        mode = "r+"

    with h5py.File(dataset_path, mode) as dataset:
        result = clean_hidden_input_grid_cubes(dataset, hidden_z_max, dry_run)
    if not dry_run:
        compact_hdf5_file(dataset_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source ERA Bench HDF5 dataset")
    parser.add_argument("--output", type=Path, required=True, help="New cleaned HDF5 dataset")
    parser.add_argument(
        "--hidden-z-max",
        type=float,
        default=DEFAULT_HIDDEN_Z_MAX,
        help=(
            "Remove input-grid cubes when their recorded state root-pose z is at or below this value "
            f"(default: {DEFAULT_HIDDEN_Z_MAX:g}; reset-hidden cubes are at z=-10)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Report removals without creating an output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demos, hidden_cubes, total_cubes = remove_hidden_input_grid_cubes(
        args.input, args.output, args.hidden_z_max, dry_run=args.dry_run
    )
    action = "Would remove" if args.dry_run else "Removed"
    print(
        f"{action} {hidden_cubes} hidden input-grid cube(s) out of {total_cubes} "
        f"input-grid cubes across {demos} demonstration(s)."
    )


if __name__ == "__main__":
    main()
