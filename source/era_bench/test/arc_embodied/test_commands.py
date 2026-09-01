from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import torch
from arc_utils import GridPair

from isaaclab.sim.utils import get_current_stage
from isaacsim.core.utils import prims as prim_utils
from pxr import UsdGeom
from era_bench.tasks.manager_based.arc_embodied.env.scene_cfg import EraBenchSceneCfg, GridArea
from era_bench.tasks.manager_based.arc_embodied.mdp.commands import (
    COMMAND_HEADER_FEATURES,
    MAX_OBJECTS_PER_GRID,
    OBJECT_FEATURES,
    LoadTaskCommand,
    TaskStatus,
)
from era_bench.tasks.manager_based.arc_embodied.mdp import events as arc_events
from era_bench.tasks.manager_based.arc_embodied.mdp.recorders import (
    ARCHighLevelActionsRecorderTerm,
    ARCTaskTestRecorderTerm,
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


def test_current_pair_test_membership_uses_the_task_train_boundary() -> None:
    command_term = object.__new__(LoadTaskCommand)
    command_term._debug_vis_handle = None
    command_term._tasks_by_id = {
        "task-a": SimpleNamespace(train=[object(), object()], test=[object(), object()]),
    }
    command_term.current_tasks = ["task-a", "task-a"]
    command_term.current_task_pair = [1, 2]

    assert command_term.is_current_pair_test(0) is False
    assert command_term.is_current_pair_test(1) is True


def test_task_test_recorder_marks_the_completed_pair_before_reset_advances_it() -> None:
    recorder = object.__new__(ARCTaskTestRecorderTerm)
    recorder._env = SimpleNamespace(num_envs=2, device="cpu")
    recorder._task_command = SimpleNamespace(is_current_pair_test=lambda env_id: env_id == 1)
    recorder._has_completed_initial_reset = False

    assert recorder.record_pre_reset([0, 1]) == (None, None)

    recorder.record_post_reset([0, 1])
    key, is_test = recorder.record_pre_reset([0, 1])

    assert key == "task_info/is_test"
    torch.testing.assert_close(is_test, torch.tensor([False, True]))


def test_reserve_grid_background_covers_the_full_reserve_stack_footprint() -> None:
    scene_cfg = object.__new__(EraBenchSceneCfg)

    assert scene_cfg.reserve_grid_background_name == "reserve_grid_background"
    assert scene_cfg.get_reserve_grid_background_center_local() == (0.51, 0.0, 0.00025)
    assert scene_cfg.get_reserve_grid_background_size() == (0.48, 0.26, 0.0005)


def test_grid_visuals_follow_raw_grid_shape_and_hide_unused_lines(monkeypatch) -> None:
    scene_cfg = SimpleNamespace(
        max_grid_size=8,
        cubes_size=(0.04, 0.04, 0.04),
        grid_cell_size=0.06,
        grid_visual_height=0.0005,
        get_grid_center_local=lambda area: (0.0, -0.5, 0.0),
        get_grid_background_name=lambda area: f"{area.value}_background",
        get_grid_row_line_name=lambda area, index: f"{area.value}_row_{index}",
        get_grid_column_line_name=lambda area, index: f"{area.value}_column_{index}",
    )
    env = SimpleNamespace(cfg=SimpleNamespace(scene=scene_cfg))
    transforms: list[tuple] = []
    visibility: list[tuple[str, bool, int]] = []
    monkeypatch.setattr(
        arc_events,
        "_set_visual_transform",
        lambda *args: transforms.append(args[2:]),
    )
    monkeypatch.setattr(
        arc_events,
        "_set_visual_visibility",
        lambda _env, env_id, asset_name, visible: visibility.append((asset_name, visible, env_id)),
    )

    arc_events._setup_grid_visuals(env, env_id=0, grid_area=GridArea.INPUT, grid=[[0] * 4 for _ in range(3)])

    assert transforms[0] == ("input_background", (-0.02, -0.52, 0.00025), (3.0, 4.0, 1.0))
    assert transforms[1] == ("input_row_0", (-0.11, -0.52, 0.00075), (1.0, 4.0, 1.0))
    assert transforms[2] == ("input_column_0", (-0.02, -0.64, 0.00075), (3.0, 1.0, 1.0))
    assert ("input_background", True, 0) in visibility
    assert ("input_row_3", True, 0) in visibility
    assert ("input_row_4", False, 0) in visibility
    assert ("input_column_4", True, 0) in visibility
    assert ("input_column_5", False, 0) in visibility


def test_grid_visual_transform_updates_a_live_usd_prim() -> None:
    asset_name = "grid_visual_transform_test"
    prim_path = f"/World/envs/env_0/{asset_name}"
    if prim_utils.is_prim_path_valid(prim_path):
        prim_utils.delete_prim(prim_path)
    prim_utils.create_prim(prim_path, prim_type="Xform")

    arc_events._set_visual_transform(
        SimpleNamespace(),
        env_id=0,
        asset_name=asset_name,
        position=(-0.02, -0.52, 0.00025),
        scale=(3.0, 4.0, 1.0),
    )

    prim = get_current_stage().GetPrimAtPath(prim_path)
    assert tuple(prim.GetAttribute("xformOp:translate").Get()) == (-0.02, -0.52, 0.00025)
    assert tuple(prim.GetAttribute("xformOp:scale").Get()) == (3.0, 4.0, 1.0)

    arc_events._set_visual_visibility(SimpleNamespace(), env_id=0, asset_name=asset_name, visible=False)
    assert UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible

    arc_events._set_visual_visibility(SimpleNamespace(), env_id=0, asset_name=asset_name, visible=True)
    assert UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == UsdGeom.Tokens.inherited


def test_high_level_actions_use_phase_specific_location_fields() -> None:
    scene_cfg = SimpleNamespace(
        reserve_cubes=["reserve_cube_00"],
        reserve_cubes_amount_per_stack=1,
        output_grid_center_local=(0.0, 0.5, 0.0),
        cubes_size=(1.0, 1.0, 1.0),
        separation_distance=1.0,
        get_reserve_cube_arc_color=lambda *_: 2,
    )
    scene = type("Scene", (dict,), {})()
    scene.env_origins = torch.zeros((1, 3))
    scene["reserve_cube_00"] = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[-0.5, 1.0, 0.5]])))
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        cfg=SimpleNamespace(scene=scene_cfg),
        scene=scene,
    )
    recorder = object.__new__(ARCHighLevelActionsRecorderTerm)
    recorder._env = env
    recorder._task_command = SimpleNamespace(current_grid_pairs=[GridPair(input=[[0]], output=[[0, 2]])])
    recorder._object_grasped = {0: [torch.tensor([False]), torch.tensor([True]), torch.tensor([False]), torch.tensor([False])]}
    recorder._object_placed = {0: [torch.tensor([False]), torch.tensor([False]), torch.tensor([False]), torch.tensor([True])]}

    key, timeline = recorder.record_pre_reset([0])

    assert key == "high_level_actions/action_timeline"
    torch.testing.assert_close(
        timeline,
        torch.tensor([[[1, 2, 0, -1, -1], [1, 2, 0, -1, -1], [2, 2, 1, 0, 1], [2, 2, 1, 0, 1]]]),
    )


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
    command_term.successful_task_pairs = {
        "task-a": set(command_term._task_pair_queue["task-a"][: pair_position + 1])
        if status is TaskStatus.COMPLETED
        else set(),
        "task-b": set(),
    }
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
    command_term = _make_command_term()
    command_term.set_task_completed(env_id=0)

    command_term._resample_command([0])

    assert command_term.current_tasks == ["task-a"]
    assert command_term.current_task_pair == [7]
    assert command_term._current_task_pair_position == [1]
    assert command_term.successful_task_pairs["task-a"] == {3}
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


def test_initialize_from_dataset_tracks_unique_pairs_and_resumes_the_first_missing_pair(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.hdf5"
    _write_task_dataset(dataset_path, [("task-a", 0), ("task-a", 0), ("task-a", 1), ("task-b", 0)])

    handler = AppendHDF5DatasetFileHandler()
    handler.open(str(dataset_path), mode="r+")
    try:
        command_term = object.__new__(LoadTaskCommand)
        command_term._debug_vis_handle = None
        command_term._env = SimpleNamespace(
            recorder_manager=SimpleNamespace(_dataset_file_handler=handler), num_envs=1
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
        command_term._task_queue = ["task-a", "task-b"]
        command_term._task_pair_queue = {"task-a": [0, 1], "task-b": [0, 1]}
        command_term.successful_task_pairs = {"task-a": set(), "task-b": set()}
        command_term.retries_per_task = {"task-a": 0, "task-b": 0}

        command_term._initialize_from_dataset()

        assert command_term.tasks_status["task-a"] is TaskStatus.COMPLETED
        assert command_term.tasks_status["task-b"] is TaskStatus.IN_PROGRESS
        assert command_term.successful_task_pairs == {"task-a": {0, 1}, "task-b": {0}}
        assert command_term.current_tasks == ["task-b"]
        assert command_term._current_task_pair_position == [1]
        assert command_term.current_task_pair == [1]
    finally:
        handler.close()


def test_initialize_from_dataset_does_not_complete_a_task_with_duplicate_pairs(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.hdf5"
    # The raw record count matches the pair count, but pair 1 was never successful.
    _write_task_dataset(dataset_path, [("task-a", 0), ("task-a", 0)])

    handler = AppendHDF5DatasetFileHandler()
    handler.open(str(dataset_path), mode="r+")
    try:
        command_term = object.__new__(LoadTaskCommand)
        command_term._debug_vis_handle = None
        command_term._env = SimpleNamespace(
            recorder_manager=SimpleNamespace(_dataset_file_handler=handler), num_envs=1
        )
        command_term._tasks_by_id = {"task-a": SimpleNamespace(train=[object(), object()], test=[])}
        command_term._task_queue = ["task-a"]
        command_term._task_pair_queue = {"task-a": [0, 1]}
        command_term.tasks_status = {"task-a": TaskStatus.NOT_STARTED}
        command_term.successful_task_pairs = {"task-a": set()}
        command_term.retries_per_task = {"task-a": 0}
        command_term.current_tasks = ["task-a"]
        command_term._current_task_pair_position = [0]
        command_term.current_task_pair = [0]

        command_term._initialize_from_dataset()

        assert command_term.successful_task_pairs == {"task-a": {0}}
        assert command_term.tasks_status["task-a"] is TaskStatus.IN_PROGRESS
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
    command_term.successful_task_pairs = {task_id: set() for task_id in command_term._task_queue}
    command_term.retries_per_task = {task_id: 0 for task_id in command_term._task_queue}

    command_term._resample_command([0, 1])

    assert command_term.current_tasks == ["task-c", "task-d"]
    assert command_term.tasks_status["task-a"] is TaskStatus.FAILED
    assert command_term.tasks_status["task-b"] is TaskStatus.FAILED
    assert command_term.tasks_status["task-c"] is TaskStatus.IN_PROGRESS
    assert command_term.tasks_status["task-d"] is TaskStatus.IN_PROGRESS


def test_initial_assignment_never_assigns_one_task_to_multiple_environments(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.hdf5"
    _write_task_dataset(dataset_path, [("task-a", 0), ("task-a", 1), ("task-b", 0)])

    handler = AppendHDF5DatasetFileHandler()
    handler.open(str(dataset_path), mode="r+")
    try:
        command_term = object.__new__(LoadTaskCommand)
        command_term._debug_vis_handle = None
        command_term._env = SimpleNamespace(
            recorder_manager=SimpleNamespace(_dataset_file_handler=handler), num_envs=2
        )
        command_term._tasks_by_id = {
            task_id: SimpleNamespace(train=[object(), object()], test=[])
            for task_id in ("task-a", "task-b", "task-c", "task-d")
        }
        command_term._task_queue = ["task-a", "task-b", "task-c", "task-d"]
        command_term._task_pair_queue = {task_id: [0, 1] for task_id in command_term._task_queue}
        command_term.tasks_status = {task_id: TaskStatus.NOT_STARTED for task_id in command_term._task_queue}
        command_term.successful_task_pairs = {task_id: set() for task_id in command_term._task_queue}
        command_term.retries_per_task = {task_id: 0 for task_id in command_term._task_queue}
        command_term.current_tasks = ["task-a", "task-b"]
        command_term._current_task_pair_position = [0, 0]
        command_term.current_task_pair = [0, 0]

        command_term._initialize_from_dataset()

        assert command_term.current_tasks == ["task-b", "task-c"]
        assert len(command_term.current_tasks) == len(set(command_term.current_tasks))
        assert command_term.current_task_pair == [1, 0]
        assert command_term.tasks_status["task-a"] is TaskStatus.COMPLETED
        assert command_term.tasks_status["task-b"] is TaskStatus.IN_PROGRESS
        assert command_term.tasks_status["task-c"] is TaskStatus.IN_PROGRESS
    finally:
        handler.close()
