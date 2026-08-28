from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(package_path)]
    sys.modules[name] = module


def _load_module(name: str, module_path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PACKAGE_ROOT = Path(__file__).resolve().parents[5] / "era_bench"
_ensure_package("era_bench", PACKAGE_ROOT)
_ensure_package("era_bench.tasks", PACKAGE_ROOT / "tasks")
_ensure_package("era_bench.tasks.manager_based", PACKAGE_ROOT / "tasks" / "manager_based")
_ensure_package(
    "era_bench.tasks.manager_based.common",
    PACKAGE_ROOT / "tasks" / "manager_based" / "common",
)
_ensure_package(
    "era_bench.tasks.manager_based.arc_embodied",
    PACKAGE_ROOT / "tasks" / "manager_based" / "arc_embodied",
)

_load_module(
    "era_bench.tasks.manager_based.common.progress_watchdog",
    PACKAGE_ROOT / "tasks" / "manager_based" / "common" / "progress_watchdog.py",
)
_load_module(
    "era_bench.tasks.manager_based.arc_embodied.scene_cfg",
    PACKAGE_ROOT / "tasks" / "manager_based" / "arc_embodied" / "scene_cfg.py",
)
success_positions_module = _load_module(
    "era_bench.tasks.manager_based.arc_embodied.success_positions",
    PACKAGE_ROOT / "tasks" / "manager_based" / "arc_embodied" / "success_positions.py",
)
terminations_module = _load_module(
    "era_bench.tasks.manager_based.arc_embodied.mdp.terminations",
    PACKAGE_ROOT / "tasks" / "manager_based" / "arc_embodied" / "mdp" / "terminations.py",
)

no_progress = terminations_module.no_progress
are_objects_in_success_positions = terminations_module.are_objects_in_success_positions
time_out = terminations_module.time_out
SuccessPositionBuffer = success_positions_module.SuccessPositionBuffer


class _FakeEnv:
    def __init__(self, distances: list[float]) -> None:
        self.num_envs = 1
        self.device = "cpu"
        self.scene = _FakeScene(self)
        self.command_manager = _FakeCommandManager()
        self.success_positions = {"blue": [torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)]}
        self._distances = list(distances)
        self._step = 0

    def get_reserve_cubes_positions_by_color(self):
        distance = self._distances[min(self._step, len(self._distances) - 1)]
        self._step += 1
        return {"blue": torch.tensor([[[distance, 0.0, 0.0]]], dtype=torch.float32)}


class _FakeScene:
    def __init__(self, env: _FakeEnv) -> None:
        self.cfg = object()
        self.env_origins = torch.zeros((env.num_envs, 3), dtype=torch.float32)
        self._frame_data = _FakeFrameData(env)

    def __getitem__(self, name: str):
        assert name == "ee_frame"
        return SimpleNamespace(data=self._frame_data)


class _FakeFrameData:
    def __init__(self, env: _FakeEnv) -> None:
        self._env = env

    @property
    def target_pos_w(self) -> torch.Tensor:
        distance = self._env._distances[min(self._env._step, len(self._env._distances) - 1)]
        return torch.tensor([[[distance, 0.0, 0.0]]], dtype=torch.float32)

    @property
    def target_quat_w(self) -> torch.Tensor:
        self._env._step += 1
        return torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)


class _FakeCommandManager:
    def __init__(self, output_grid: list[list[int]] | None = None) -> None:
        self.failed_env_ids: list[int] = []
        self.completed_env_ids: list[int] = []
        self.current_grid_pairs = [SimpleNamespace(output=output_grid or [[1]])]

    def get_term(self, name: str):
        assert name == "load_task"
        return self

    def set_task_failed(self, env_id: int) -> None:
        self.failed_env_ids.append(env_id)

    def set_task_completed(self, env_id: int) -> None:
        self.completed_env_ids.append(env_id)


class _FakeSuccessEnv:
    def __init__(
        self,
        target_position: tuple[float, float, float] = (0.0, 0.5, 0.0),
        tcp_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        output_grid: list[list[int]] | None = None,
    ) -> None:
        self.num_envs = 1
        self.device = "cpu"
        self.scene = _FakeSuccessScene(tcp_position)
        self.command_manager = _FakeCommandManager(output_grid)
        self.success_positions = SuccessPositionBuffer(num_envs=1, max_targets=4, device="cpu")
        self._target_position = torch.tensor(target_position, dtype=torch.float32)
        self.success_positions.replace(0, self._target_position.unsqueeze(0), torch.tensor([1]))
        self.success_positions_reached = torch.zeros((1, 4), dtype=torch.bool)
        self.reserve_cube_stacks = SimpleNamespace(color_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

    def get_live_reserve_cube_stack_poses(self):
        poses = torch.zeros((1, 4, 1, 7), dtype=torch.float32)
        poses[0, 0, 0, :3] = self._target_position
        return poses


class _FakeSuccessScene:
    def __init__(self, tcp_position: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.cfg = SimpleNamespace(
            cubes_size=(0.04, 0.04, 0.04),
            separation_distance=0.02,
            output_grid_center_local=(0.0, 0.5, 0.0),
            table_cam=SimpleNamespace(offset=SimpleNamespace(pos=(1.8, 0.0, 1.7))),
        )
        self.env_origins = torch.zeros((1, 3), dtype=torch.float32)
        self._frame_data = SimpleNamespace(
            target_pos_w=torch.tensor(tcp_position, dtype=torch.float32).view(1, 1, 3),
            target_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        )

    def __getitem__(self, name: str):
        assert name == "ee_frame"
        return SimpleNamespace(data=self._frame_data)


class _FakeTimeoutEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.device = "cpu"
        self.command_manager = _FakeCommandManager()
        self.episode_length_buf = torch.tensor([2, 3], dtype=torch.int64)
        self.max_episode_length = 3


def test_no_progress_triggers_after_patience() -> None:
    env = _FakeEnv([0.10, 0.0998, 0.0997, 0.0996])
    term = no_progress(
        cfg=SimpleNamespace(
            params={
                "patience": 3,
                "min_improvement": 1.0e-3,
                "warmup_steps": 0,
            }
        ),
        env=env,
    )

    outputs = [term(env).item() for _ in range(4)]

    assert outputs == [False, False, True, True]
    assert env.command_manager.failed_env_ids == [0, 0]


def test_no_progress_resets_after_real_improvement() -> None:
    env = _FakeEnv([0.10, 0.0998, 0.0970, 0.0968, 0.0967, 0.0966])
    term = no_progress(
        cfg=SimpleNamespace(
            params={
                "patience": 3,
                "min_improvement": 1.0e-3,
                "warmup_steps": 0,
            }
        ),
        env=env,
    )

    outputs = [term(env).item() for _ in range(6)]

    assert outputs == [False, False, False, False, False, True]
    assert env.command_manager.failed_env_ids == [0]


def test_success_marks_current_task_completed() -> None:
    env = _FakeSuccessEnv()

    success = are_objects_in_success_positions(env, distance_threshold=0.015)

    assert success.tolist() == [True]
    assert env.command_manager.completed_env_ids == [0]


def test_success_waits_until_tcp_is_clear_of_the_table_camera_grid_line() -> None:
    env = _FakeSuccessEnv()

    success = are_objects_in_success_positions(env, distance_threshold=0.015)

    # The target buffer has three zero-padded slots.  They must not affect the
    # top-left grid corner used to form the table-camera boundary line.
    assert success.tolist() == [True]

    # This TCP has a low enough y coordinate for the removed front-edge check,
    # but remains on the output side of the table-camera boundary line.
    env.scene._frame_data.target_pos_w[0, 0, :2] = torch.tensor([1.7, 0.0])
    success = are_objects_in_success_positions(env, distance_threshold=0.015)

    assert success.tolist() == [False]


def test_success_uses_output_cell_zero_when_it_is_background() -> None:
    # The only output cube is at (1, 1).  Clearance must still be measured
    # from the empty (0, 0) cell, whose camera boundary is farther from the
    # populated target line at this TCP position.
    env = _FakeSuccessEnv(
        target_position=(0.01, 0.51, 0.02),
        tcp_position=(1.17, 0.0, 0.245913),
        output_grid=[[0, 0], [0, 1]],
    )

    success = are_objects_in_success_positions(env, distance_threshold=0.015)

    assert success.tolist() == [False]


def test_success_allows_demo_12_final_cube_from_its_reset_tcp_pose() -> None:
    # Demo 12: final output-cube position and the recorded reset TCP pose.
    env = _FakeSuccessEnv(
        target_position=(0.0, 0.5, 0.02),
        tcp_position=(0.458353, -0.0200575, 0.245913),
    )

    success = are_objects_in_success_positions(env, distance_threshold=0.015)

    assert success.tolist() == [True]
    assert env.command_manager.completed_env_ids == [0]


@pytest.mark.parametrize(
    "target_position",
    [
        (-0.25, 0.25, 0.02),
        (0.25, 0.25, 0.02),
        (-0.25, 0.75, 0.02),
        (0.25, 0.75, 0.02),
    ],
)
def test_success_allows_reset_tcp_pose_for_each_output_grid_corner(
    target_position: tuple[float, float, float],
) -> None:
    env = _FakeSuccessEnv(
        target_position=target_position,
        tcp_position=(0.458353, -0.0200575, 0.245913),
    )

    success = are_objects_in_success_positions(env, distance_threshold=0.015)

    assert success.tolist() == [True]
    assert env.command_manager.completed_env_ids == [0]


def test_time_out_marks_timed_out_tasks_failed() -> None:
    env = _FakeTimeoutEnv()

    timed_out = time_out(env)

    assert timed_out.tolist() == [False, True]
    assert env.command_manager.failed_env_ids == [1]


def teardown_module() -> None:
    simulation_app.close()
