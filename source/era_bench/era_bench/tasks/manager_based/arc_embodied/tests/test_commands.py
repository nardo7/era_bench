from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import torch
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from era_bench.tasks.manager_based.arc_embodied.mdp.commands import (
    COMMAND_HEADER_FEATURES,
    MAX_OBJECTS_PER_GRID,
    OBJECT_FEATURES,
    LoadTaskCommand,
    TaskStatus,
)
from era_bench.tasks.manager_based.common.dataset_handler import AppendHDF5DatasetFileHandler


def test_get_features_returns_unpadded_features_for_the_requested_environment() -> None:
    """Input and output features occupy separate fixed-width regions in each command row."""
    command_term = object.__new__(LoadTaskCommand)
    command_term._debug_vis_handle = None
    command_term._command = torch.zeros(
        (2, COMMAND_HEADER_FEATURES + 2 * MAX_OBJECTS_PER_GRID * OBJECT_FEATURES),
        dtype=torch.float32,
    )

    input_offset = COMMAND_HEADER_FEATURES
    output_offset = input_offset + MAX_OBJECTS_PER_GRID * OBJECT_FEATURES

    # Environment 0 deliberately uses different values to verify row selection.
    command_term._command[0, :2] = torch.tensor([1.0, 1.0])
    command_term._command[0, input_offset : input_offset + OBJECT_FEATURES] = torch.tensor([1.0, 2.0, 3.0])
    command_term._command[0, output_offset : output_offset + OBJECT_FEATURES] = torch.tensor([4.0, 5.0, 6.0])

    # Environment 1 has two input objects and one output object. Sentinel values
    # after each count must not be returned.
    command_term._command[1, :2] = torch.tensor([2.0, 1.0])
    command_term._command[1, input_offset : input_offset + 3 * OBJECT_FEATURES] = torch.tensor(
        [10.0, 11.0, 1.0, 20.0, 21.0, 2.0, 99.0, 99.0, 99.0]
    )
    command_term._command[1, output_offset : output_offset + 2 * OBJECT_FEATURES] = torch.tensor(
        [30.0, 31.0, 3.0, 88.0, 88.0, 88.0]
    )

    input_features, output_features = command_term.get_features(env_id=1)

    torch.testing.assert_close(input_features, torch.tensor([[10.0, 11.0, 1.0], [20.0, 21.0, 2.0]]))
    torch.testing.assert_close(output_features, torch.tensor([[30.0, 31.0, 3.0]]))


def _make_command_term(
    *,
    status: TaskStatus = TaskStatus.IN_PROGRESS,
    pair_position: int = 0,
    retries_used: int = 0,
    retries: int = 3,
) -> LoadTaskCommand:
    """Build the smallest command-term state needed to exercise pair progression."""
    command_term = object.__new__(LoadTaskCommand)
    command_term._debug_vis_handle = None
    command_term._env = SimpleNamespace(device="cpu", num_envs=1)
    command_term.cfg = SimpleNamespace(retries=retries)
    command_term._prepared_for_reset = torch.zeros(1, dtype=torch.bool)
    command_term.current_tasks = ["task-a"]
    command_term._current_task_pair_position = [pair_position]
    command_term._task_pair_queue = {"task-a": [3, 7], "task-b": [2]}
    command_term.current_task_pair = [command_term._task_pair_queue["task-a"][pair_position]]
    command_term.tasks_status = {"task-a": status, "task-b": TaskStatus.NOT_STARTED}
    command_term.retries_per_task = {"task-a": retries_used, "task-b": 0}
    command_term._task_queue = ["task-a", "task-b"]
    return command_term


def test_resample_leaves_an_in_progress_pair_unchanged() -> None:
    command_term = _make_command_term()

    command_term._resample_command([0])

    assert command_term.current_tasks == ["task-a"]
    assert command_term.current_task_pair == [3]
    assert command_term._current_task_pair_position == [0]


def test_failed_pair_is_retried_until_retries_are_exhausted() -> None:
    command_term = _make_command_term(status=TaskStatus.FAILED, retries_used=1, retries=3)

    command_term._resample_command([0])

    assert command_term.current_tasks == ["task-a"]
    assert command_term.current_task_pair == [3]
    assert command_term._current_task_pair_position == [0]
    assert command_term.retries_per_task["task-a"] == 2
    assert command_term.tasks_status["task-a"] is TaskStatus.IN_PROGRESS


def test_exhausted_failed_pair_skips_remaining_pairs_and_advances_task() -> None:
    command_term = _make_command_term(status=TaskStatus.FAILED, retries_used=3, retries=3)

    command_term._resample_command([0])

    assert command_term.current_tasks == ["task-b"]
    assert command_term.current_task_pair == [2]
    assert command_term._current_task_pair_position == [0]
    assert command_term.tasks_status["task-a"] is TaskStatus.FAILED
    assert command_term.tasks_status["task-b"] is TaskStatus.IN_PROGRESS


def test_completed_pair_advances_to_the_next_pair_not_the_next_task() -> None:
    command_term = _make_command_term(status=TaskStatus.COMPLETED)

    command_term._resample_command([0])

    assert command_term.current_tasks == ["task-a"]
    assert command_term.current_task_pair == [7]
    assert command_term._current_task_pair_position == [1]
    assert command_term.tasks_status["task-a"] is TaskStatus.IN_PROGRESS
    assert command_term.tasks_status["task-b"] is TaskStatus.NOT_STARTED


def test_completed_final_pair_advances_to_the_next_task() -> None:
    command_term = _make_command_term(status=TaskStatus.COMPLETED, pair_position=1)

    command_term._resample_command([0])

    assert command_term.current_tasks == ["task-b"]
    assert command_term.current_task_pair == [2]
    assert command_term._current_task_pair_position == [0]
    assert command_term.tasks_status["task-a"] is TaskStatus.COMPLETED
    assert command_term.tasks_status["task-b"] is TaskStatus.IN_PROGRESS


def _write_task_dataset(dataset_path, records: list[tuple[str, int]]) -> None:
    with h5py.File(dataset_path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        for demo_id, (task_id, task_pair) in enumerate(records):
            demo_group = data_group.create_group(f"demo_{demo_id}")
            task_info_group = demo_group.create_group("task_info")
            task_info_group.create_dataset(
                "task_id", data=np.asarray([[*task_id.encode("ascii")]], dtype=np.uint8)
            )
            task_info_group.create_dataset("task_pair", data=np.asarray([[task_pair]], dtype=np.int64))


def test_dataset_handler_reads_nested_task_metadata(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.hdf5"
    _write_task_dataset(dataset_path, [("task-a", 0), ("task-a", 1), ("task-b", 0)])

    handler = AppendHDF5DatasetFileHandler()
    handler.open(str(dataset_path), mode="r+")
    try:
        assert handler.get_executed_task_ids_with_pairs() == {"task-a": [0, 1], "task-b": [0]}
    finally:
        handler.close()


def test_initialize_from_dataset_marks_only_fully_recorded_tasks_completed(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.hdf5"
    _write_task_dataset(dataset_path, [("task-a", 0), ("task-a", 1), ("task-b", 0)])

    handler = AppendHDF5DatasetFileHandler()
    handler.open(str(dataset_path), mode="r+")
    try:
        command_term = object.__new__(LoadTaskCommand)
        command_term._debug_vis_handle = None
        command_term._env = SimpleNamespace(
            recorder_manager=SimpleNamespace(_dataset_file_handler=handler)
        )
        command_term._tasks_by_id = {
            "task-a": SimpleNamespace(train=[object(), object()], test=[]),
            "task-b": SimpleNamespace(train=[object(), object()], test=[]),
        }
        command_term.tasks_status = {
            "task-a": TaskStatus.NOT_STARTED,
            "task-b": TaskStatus.NOT_STARTED,
        }
        command_term.current_tasks = ["task-a"]
        command_term._current_task_pair_position = [0]
        command_term.current_task_pair = [0]

        command_term._initialize_from_dataset()

        assert command_term.tasks_status["task-a"] is TaskStatus.COMPLETED
        assert command_term.tasks_status["task-b"] is TaskStatus.NOT_STARTED
        assert command_term._current_task_pair_position == [1]
        assert command_term.current_task_pair == [1]
    finally:
        handler.close()


def test_failed_final_pair_preserves_failed_task_status() -> None:
    command_term = _make_command_term(status=TaskStatus.FAILED, pair_position=1, retries_used=3, retries=3)

    command_term._resample_command([0])

    assert command_term.current_tasks == ["task-b"]
    assert command_term.tasks_status["task-a"] is TaskStatus.FAILED
    assert command_term.tasks_status["task-b"] is TaskStatus.IN_PROGRESS


def test_resample_processes_all_environments_after_exhausted_failures() -> None:
    command_term = object.__new__(LoadTaskCommand)
    command_term._debug_vis_handle = None
    command_term._env = SimpleNamespace(device="cpu", num_envs=2)
    command_term.cfg = SimpleNamespace(retries=0)
    command_term._prepared_for_reset = torch.zeros(2, dtype=torch.bool)
    command_term.current_tasks = ["task-a", "task-b"]
    command_term.current_task_pair = [0, 0]
    command_term._current_task_pair_position = [0, 0]
    command_term._task_pair_queue = {
        "task-a": [0],
        "task-b": [0],
        "task-c": [0],
        "task-d": [0],
    }
    command_term._task_queue = ["task-a", "task-b", "task-c", "task-d"]
    command_term.tasks_status = {
        "task-a": TaskStatus.FAILED,
        "task-b": TaskStatus.FAILED,
        "task-c": TaskStatus.NOT_STARTED,
        "task-d": TaskStatus.NOT_STARTED,
    }
    command_term.retries_per_task = {task_id: 0 for task_id in command_term._task_queue}

    command_term._resample_command([0, 1])

    assert command_term.current_tasks == ["task-c", "task-d"]
    assert command_term.tasks_status["task-a"] is TaskStatus.FAILED
    assert command_term.tasks_status["task-b"] is TaskStatus.FAILED
    assert command_term.tasks_status["task-c"] is TaskStatus.IN_PROGRESS
    assert command_term.tasks_status["task-d"] is TaskStatus.IN_PROGRESS
