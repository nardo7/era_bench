#!/usr/bin/env python3
"""Copy an ERA Bench dataset without hidden input-grid reference cubes.

Unused input-grid cube slots are reset to local ``z = -10``.  The recorder
stores those slots in every demonstration's initial state, even though they
are not visible in the scene.  This script removes only the corresponding
``initial_state/rigid_object/input_grid_cube_*`` groups whose recorded poses
are below ``--hidden-z-max``.  All non-input-grid objects and all other HDF5
data are copied unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

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
    """Return whether every recorded root-pose position is below the threshold.

    Requiring every pose to be hidden prevents a malformed or time-varying
    recording from losing a cube that was visible at any recorded instant.
    """
    root_pose = cube_group.get(ROOT_POSE_DATASET)
    if not isinstance(root_pose, h5py.Dataset):
        return False
    if root_pose.ndim == 0 or root_pose.shape[-1] < 3:
        raise ValueError(f"{cube_group.name}/{ROOT_POSE_DATASET} must end in an xyz pose, got {root_pose.shape}")

    z_positions = root_pose[..., 2]
    return bool(np.all(np.isfinite(z_positions)) and np.all(z_positions <= hidden_z_max))


def find_hidden_input_grid_cubes(dataset: h5py.File, hidden_z_max: float) -> dict[str, list[str]]:
    """Return hidden input-grid cube paths grouped by demonstration name."""
    data_group = dataset.get("data")
    if not isinstance(data_group, h5py.Group):
        raise ValueError(f"dataset has no /data group: {dataset.filename}")

    hidden_by_demo: dict[str, list[str]] = {}
    for demo_name in sorted(data_group, key=_demo_sort_key):
        demo_group = data_group[demo_name]
        if not isinstance(demo_group, h5py.Group):
            continue
        rigid_objects = demo_group.get("initial_state/rigid_object")
        if not isinstance(rigid_objects, h5py.Group):
            continue

        hidden_names = [
            object_name
            for object_name, object_group in rigid_objects.items()
            if object_name.startswith(INPUT_GRID_CUBE_PREFIX)
            and isinstance(object_group, h5py.Group)
            and is_hidden_input_grid_cube(object_group, hidden_z_max)
        ]
        if hidden_names:
            hidden_by_demo[demo_name] = sorted(hidden_names)
    return hidden_by_demo


def remove_hidden_input_grid_cubes(
    input_path: Path, output_path: Path, hidden_z_max: float, dry_run: bool = False
) -> tuple[int, int]:
    """Copy ``input_path`` to ``output_path`` and remove hidden cube groups.

    Returns ``(demo_count_with_removals, removed_cube_count)``.  The source is
    never modified; use an explicit output path for a recoverable cleanup.
    """
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--output must differ from --input; the source dataset is never modified")
    if not input_path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output dataset: {output_path}")

    with h5py.File(input_path, "r") as source:
        hidden_by_demo = find_hidden_input_grid_cubes(source, hidden_z_max)

    demo_count = len(hidden_by_demo)
    removed_count = sum(len(names) for names in hidden_by_demo.values())
    for demo_name, object_names in hidden_by_demo.items():
        print(f"[remove] /data/{demo_name}/initial_state/rigid_object: {len(object_names)} hidden input-grid cube(s)")

    if dry_run:
        return demo_count, removed_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    with h5py.File(output_path, "r+") as output:
        for demo_name, object_names in hidden_by_demo.items():
            rigid_objects = output[f"data/{demo_name}/initial_state/rigid_object"]
            for object_name in object_names:
                del rigid_objects[object_name]
    return demo_count, removed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source ERA Bench HDF5 dataset")
    parser.add_argument("--output", type=Path, required=True, help="New cleaned HDF5 dataset")
    parser.add_argument(
        "--hidden-z-max",
        type=float,
        default=DEFAULT_HIDDEN_Z_MAX,
        help=(
            "Remove input-grid cubes only when all recorded root-pose z values are at or below this value "
            f"(default: {DEFAULT_HIDDEN_Z_MAX:g}; reset-hidden cubes are at z=-10)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Report removals without creating an output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demos, cubes = remove_hidden_input_grid_cubes(
        args.input, args.output, args.hidden_z_max, dry_run=args.dry_run
    )
    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {cubes} hidden input-grid cube(s) across {demos} demonstration(s).")


if __name__ == "__main__":
    main()
