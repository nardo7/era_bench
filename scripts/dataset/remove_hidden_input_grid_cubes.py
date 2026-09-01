#!/usr/bin/env python3
"""Create a compact ERA Bench dataset without hidden cubes or duplicate pairs.

Each retained source episode is copied once into a temporary output file. The
rewrite removes hidden input-grid reference cubes from copied state groups and
retains only the earliest source episode for each ``(task_id, task_pair)``.
The input dataset is never modified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
import tempfile

import h5py
import numpy as np


INPUT_GRID_CUBE_PREFIX = "input_grid_cube_"
ROOT_POSE_DATASET = "root_pose"
DEFAULT_HIDDEN_Z_MAX = -1.0
SOURCE_DEMO_PATTERN = re.compile(r"demo_(\d+)")
TASK_VIEW_MARKER = "era_bench_task_view_v1"


@dataclass(frozen=True)
class CleanupStats:
    input_demo_count: int
    retained_demo_count: int
    duplicate_demo_count: int
    demos_with_hidden_cubes: int
    hidden_cube_count: int
    total_input_grid_cube_count: int


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = SOURCE_DEMO_PATTERN.fullmatch(name)
    return (int(match.group(1)), name) if match else (2**63 - 1, name)


def _copy_attributes(source: h5py.File | h5py.Group, destination: h5py.File | h5py.Group) -> None:
    for attribute_name, value in source.attrs.items():
        destination.attrs[attribute_name] = value


def _object_address(group: h5py.Group) -> int:
    return int(h5py.h5o.get_info(group.id).addr)


def _decode_task_id(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset[-1]).reshape(-1)
    try:
        return bytes(values.tolist()).decode("ascii")
    except (TypeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid task ID encoding at {dataset.name}") from error


def _decode_task_pair(dataset: h5py.Dataset) -> int:
    values = np.asarray(dataset[-1]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected one task-pair value at {dataset.name}, found {values.size}")
    return int(values[0])


def _task_key(demo_group: h5py.Group) -> tuple[str, int]:
    try:
        task_info = demo_group["task_info"]
        task_id = task_info["task_id"]
        task_pair = task_info["task_pair"]
    except KeyError as error:
        raise ValueError(f"Missing task metadata in {demo_group.name}: {error}") from error
    if not isinstance(task_id, h5py.Dataset) or not isinstance(task_pair, h5py.Dataset):
        raise TypeError(f"Task metadata in {demo_group.name} must be datasets")
    return _decode_task_id(task_id), _decode_task_pair(task_pair)


def is_hidden_input_grid_cube(cube_group: h5py.Group, hidden_z_max: float) -> bool:
    root_pose = cube_group.get(ROOT_POSE_DATASET)
    if not isinstance(root_pose, h5py.Dataset):
        return False
    if root_pose.ndim == 0 or root_pose.shape[0] == 0 or root_pose.shape[-1] < 3:
        raise ValueError(f"{cube_group.name}/{ROOT_POSE_DATASET} must end in an xyz pose, got {root_pose.shape}")
    z_position = root_pose[0, ..., 2]
    return bool(np.all(np.isfinite(z_position)) and np.all(z_position <= hidden_z_max))


def _hidden_cube_names(demo_group: h5py.Group, hidden_z_max: float) -> tuple[set[str], int]:
    state_objects = demo_group.get("states/rigid_object")
    if not isinstance(state_objects, h5py.Group):
        return set(), 0
    cube_names = sorted(name for name in state_objects if name.startswith(INPUT_GRID_CUBE_PREFIX))
    hidden_names = {
        cube_name
        for cube_name in cube_names
        if isinstance(state_objects[cube_name], h5py.Group)
        and is_hidden_input_grid_cube(state_objects[cube_name], hidden_z_max)
    }
    return hidden_names, len(cube_names)


def _copy_state_root_without_hidden_cubes(
    source_root: h5py.Group,
    destination_demo: h5py.Group,
    root_name: str,
    hidden_cube_names: set[str],
) -> None:
    destination_root = destination_demo.create_group(root_name)
    _copy_attributes(source_root, destination_root)
    for child_name in source_root:
        if child_name != "rigid_object":
            source_root.copy(child_name, destination_root)
            continue
        source_objects = source_root[child_name]
        if not isinstance(source_objects, h5py.Group):
            source_root.copy(child_name, destination_root)
            continue
        destination_objects = destination_root.create_group(child_name)
        _copy_attributes(source_objects, destination_objects)
        for object_name in source_objects:
            if object_name not in hidden_cube_names:
                source_objects.copy(object_name, destination_objects)


def _copy_demo_without_hidden_cubes(
    source_demo: h5py.Group,
    destination_data: h5py.Group,
    destination_name: str,
    hidden_z_max: float,
) -> tuple[int, int]:
    hidden_cube_names, total_cube_count = _hidden_cube_names(source_demo, hidden_z_max)
    destination_demo = destination_data.create_group(destination_name)
    _copy_attributes(source_demo, destination_demo)
    for child_name in source_demo:
        source_child = source_demo[child_name]
        if child_name in {"states", "initial_state"} and isinstance(source_child, h5py.Group):
            _copy_state_root_without_hidden_cubes(source_child, destination_demo, child_name, hidden_cube_names)
        else:
            source_demo.copy(child_name, destination_demo)
    return len(hidden_cube_names), total_cube_count


def _copy_task_view(
    source_view: h5py.Group,
    destination_data: h5py.Group,
    source_address_to_destination_demo: dict[int, str],
    task_pair_to_destination_demo: dict[tuple[str, int], str],
    destination_rules: h5py.Group | None,
) -> None:
    """Copy view metadata and recreate episode/rule hard links in the output."""
    task_id = source_view.name.rsplit("/", 1)[-1]
    destination_view = destination_data.create_group(task_id)
    _copy_attributes(source_view, destination_view)
    for child_name in source_view:
        source_child = source_view[child_name]
        if child_name == "rule" and destination_rules is not None and task_id in destination_rules:
            destination_view[child_name] = destination_rules[task_id]
        elif child_name == "arc_grids":
            source_view.copy(child_name, destination_view)
        elif child_name.startswith(("demo_", "test_")) and isinstance(source_child, h5py.Group):
            destination_demo_name = source_address_to_destination_demo.get(_object_address(source_child))
            if destination_demo_name is None:
                destination_demo_name = task_pair_to_destination_demo[_task_key(source_child)]
            destination_view[child_name] = destination_data[destination_demo_name]
        else:
            source_view.copy(child_name, destination_view)


def _inspect_source(input_path: Path, hidden_z_max: float) -> CleanupStats:
    """Read-only preview of the records that the rewrite will retain."""
    with h5py.File(input_path, "r") as source:
        source_data = source.get("data")
        if not isinstance(source_data, h5py.Group):
            raise ValueError(f"dataset has no /data group: {input_path}")
        source_demo_names = sorted(
            (name for name in source_data if SOURCE_DEMO_PATTERN.fullmatch(name)), key=_demo_sort_key
        )
        seen_task_pairs: set[tuple[str, int]] = set()
        retained_demo_count = 0
        duplicate_demo_count = 0
        demos_with_hidden_cubes = 0
        hidden_cube_count = 0
        total_cube_count = 0
        for source_name in source_demo_names:
            source_demo = source_data[source_name]
            if not isinstance(source_demo, h5py.Group):
                raise TypeError(f"Expected /data/{source_name} to be a group")
            task_key = _task_key(source_demo)
            if task_key in seen_task_pairs:
                duplicate_demo_count += 1
                continue
            seen_task_pairs.add(task_key)
            hidden_cube_names, cube_count = _hidden_cube_names(source_demo, hidden_z_max)
            retained_demo_count += 1
            demos_with_hidden_cubes += int(bool(hidden_cube_names))
            hidden_cube_count += len(hidden_cube_names)
            total_cube_count += cube_count
    return CleanupStats(
        input_demo_count=len(source_demo_names),
        retained_demo_count=retained_demo_count,
        duplicate_demo_count=duplicate_demo_count,
        demos_with_hidden_cubes=demos_with_hidden_cubes,
        hidden_cube_count=hidden_cube_count,
        total_input_grid_cube_count=total_cube_count,
    )


def _validate_output(temporary_path: Path, expected_stats: CleanupStats, hidden_z_max: float) -> None:
    """Check the temporary file before it is atomically published."""
    observed_stats = _inspect_source(temporary_path, hidden_z_max)
    expected_output_stats = CleanupStats(
        input_demo_count=expected_stats.retained_demo_count,
        retained_demo_count=expected_stats.retained_demo_count,
        duplicate_demo_count=0,
        demos_with_hidden_cubes=0,
        hidden_cube_count=0,
        total_input_grid_cube_count=expected_stats.total_input_grid_cube_count - expected_stats.hidden_cube_count,
    )
    if observed_stats != expected_output_stats:
        raise ValueError(
            "temporary rewrite validation failed: expected "
            f"{expected_output_stats}, found {observed_stats}"
        )
    with h5py.File(temporary_path, "r") as dataset:
        data_group = dataset["data"]
        expected_demo_names = [f"demo_{index}" for index in range(expected_stats.retained_demo_count)]
        actual_demo_names = sorted(
            (name for name in data_group if SOURCE_DEMO_PATTERN.fullmatch(name)), key=_demo_sort_key
        )
        if actual_demo_names != expected_demo_names:
            raise ValueError(
                "temporary rewrite validation failed: raw demo names are not contiguous: "
                f"{actual_demo_names}"
            )
        if int(data_group.attrs.get("total", -1)) != expected_stats.retained_demo_count:
            raise ValueError("temporary rewrite validation failed: /data total does not match retained demos")

        raw_demo_addresses = {_object_address(data_group[name]) for name in actual_demo_names}
        for name in data_group:
            child = data_group[name]
            if not isinstance(child, h5py.Group) or child.attrs.get("view_type") != TASK_VIEW_MARKER:
                continue
            for member_name in child:
                if member_name.startswith(("demo_", "test_")):
                    member = child[member_name]
                    if not isinstance(member, h5py.Group) or _object_address(member) not in raw_demo_addresses:
                        raise ValueError(
                            f"temporary rewrite validation failed: {child.name}/{member_name} is not a raw-demo link"
                        )


def _copy_dataset_once(input_path: Path, temporary_path: Path, hidden_z_max: float) -> CleanupStats:
    with h5py.File(input_path, "r") as source, h5py.File(temporary_path, "w") as destination:
        _copy_attributes(source, destination)
        for root_name in source:
            if root_name != "data":
                source.copy(root_name, destination)

        source_data = source.get("data")
        if not isinstance(source_data, h5py.Group):
            raise ValueError(f"dataset has no /data group: {input_path}")
        destination_data = destination.create_group("data")
        _copy_attributes(source_data, destination_data)
        destination_rules = destination.get("rules")
        if not isinstance(destination_rules, h5py.Group):
            destination_rules = None

        source_demo_names = sorted(
            (name for name in source_data if SOURCE_DEMO_PATTERN.fullmatch(name)),
            key=_demo_sort_key,
        )
        source_address_to_destination_demo: dict[int, str] = {}
        task_pair_to_destination_demo: dict[tuple[str, int], str] = {}
        duplicate_sources: dict[tuple[str, int], list[str]] = defaultdict(list)
        demos_with_hidden_cubes = 0
        hidden_cube_count = 0
        total_cube_count = 0

        for source_index, source_name in enumerate(source_demo_names, start=1):
            source_demo = source_data[source_name]
            if not isinstance(source_demo, h5py.Group):
                raise TypeError(f"Expected /data/{source_name} to be a group")
            task_key = _task_key(source_demo)
            if task_key in task_pair_to_destination_demo:
                duplicate_sources[task_key].append(source_name)
                print(f"[rewrite {source_index}/{len(source_demo_names)}] skip duplicate {source_name}: {task_key}")
                continue

            destination_name = f"demo_{len(task_pair_to_destination_demo)}"
            hidden_count, cube_count = _copy_demo_without_hidden_cubes(
                source_demo, destination_data, destination_name, hidden_z_max
            )
            source_address_to_destination_demo[_object_address(source_demo)] = destination_name
            task_pair_to_destination_demo[task_key] = destination_name
            demos_with_hidden_cubes += int(hidden_count > 0)
            hidden_cube_count += hidden_count
            total_cube_count += cube_count
            print(
                f"[rewrite {source_index}/{len(source_demo_names)}] {source_name} -> {destination_name}: "
                f"{hidden_count} hidden input-grid cube(s) out of {cube_count}",
                flush=True,
            )

        destination_data.attrs["total"] = len(task_pair_to_destination_demo)
        for source_name in sorted(name for name in source_data if name not in source_demo_names):
            source_child = source_data[source_name]
            if isinstance(source_child, h5py.Group) and source_child.attrs.get("view_type") == TASK_VIEW_MARKER:
                _copy_task_view(
                    source_child,
                    destination_data,
                    source_address_to_destination_demo,
                    task_pair_to_destination_demo,
                    destination_rules,
                )
            else:
                source_data.copy(source_name, destination_data)

    duplicate_count = sum(len(sources) for sources in duplicate_sources.values())
    if duplicate_count:
        print(
            f"[rewrite] Kept the earliest source episode for each task pair and skipped "
            f"{duplicate_count} duplicate episode(s) across {len(duplicate_sources)} task pair(s)."
        )
    return CleanupStats(
        input_demo_count=len(source_demo_names),
        retained_demo_count=len(task_pair_to_destination_demo),
        duplicate_demo_count=duplicate_count,
        demos_with_hidden_cubes=demos_with_hidden_cubes,
        hidden_cube_count=hidden_cube_count,
        total_input_grid_cube_count=total_cube_count,
    )


def remove_hidden_input_grid_cubes(
    input_path: Path, output_path: Path, hidden_z_max: float, dry_run: bool = False
) -> CleanupStats:
    """Create one compact output rewrite with both requested cleanups applied."""
    if not input_path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--output must differ from --input; the source dataset is never modified")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output dataset: {output_path}")
    if dry_run:
        stats = _inspect_source(input_path, hidden_z_max)
        print(
            f"[dry-run] Would retain {stats.retained_demo_count}/{stats.input_demo_count} source episode(s), "
            f"skip {stats.duplicate_demo_count} duplicate episode(s), and remove {stats.hidden_cube_count} "
            f"hidden input-grid cube(s)."
        )
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}_", suffix=".tmp.hdf5", dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        print(f"[rewrite] {input_path} -> {output_path}", flush=True)
        stats = _copy_dataset_once(input_path, temporary_path, hidden_z_max)
        _validate_output(temporary_path, stats, hidden_z_max)
        os.chmod(temporary_path, stat.S_IMODE(input_path.stat().st_mode))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return stats


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report retained duplicates and hidden cubes without creating an output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = remove_hidden_input_grid_cubes(args.input, args.output, args.hidden_z_max, dry_run=args.dry_run)
    if not args.dry_run:
        print(
            f"Retained {stats.retained_demo_count}/{stats.input_demo_count} source episode(s), skipped "
            f"{stats.duplicate_demo_count} duplicate episode(s), and removed {stats.hidden_cube_count} hidden "
            f"input-grid cube(s) out of {stats.total_input_grid_cube_count} across "
            f"{stats.demos_with_hidden_cubes} retained demonstration(s)."
        )


if __name__ == "__main__":
    main()
