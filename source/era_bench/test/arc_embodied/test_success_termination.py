import importlib.util
import pathlib
import sys
import types

import torch
from arc_utils import ARC_COLOR_TO_COLOR


def _ensure_pkg(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def _stub_dependencies() -> None:
    for pkg in [
        "era_bench",
        "era_bench.tasks",
        "era_bench.tasks.manager_based",
        "era_bench.tasks.manager_based.arc_embodied",
        "era_bench.tasks.manager_based.arc_embodied.env",
        "era_bench.tasks.manager_based.arc_embodied.mdp",
        "era_bench.tasks.manager_based.common",
        "isaaclab",
        "isaaclab.envs",
    ]:
        _ensure_pkg(pkg)

    env_mod = types.ModuleType("era_bench.tasks.manager_based.arc_embodied.env.environment")
    env_mod.EraBenchEnv = object
    sys.modules[env_mod.__name__] = env_mod

    geom_mod = types.ModuleType("era_bench.tasks.manager_based.common.geometry_utils")
    geom_mod.mirror_points_across_line = lambda points, line_pos, line_dir: points
    sys.modules[geom_mod.__name__] = geom_mod

    scene_cfg_mod = types.ModuleType("era_bench.tasks.manager_based.arc_embodied.env.scene_cfg")
    scene_cfg_mod.EraBenchSceneCfg = object
    sys.modules[scene_cfg_mod.__name__] = scene_cfg_mod

    assets_mod = types.ModuleType("isaaclab.assets")
    assets_mod.RigidObject = object
    sys.modules[assets_mod.__name__] = assets_mod

    managers_mod = types.ModuleType("isaaclab.managers")
    managers_mod.ManagerTermBase = object
    sys.modules[managers_mod.__name__] = managers_mod

    sensors_mod = types.ModuleType("isaaclab.sensors")
    sensors_mod.FrameTransformer = object
    sys.modules[sensors_mod.__name__] = sensors_mod

    utils_mod = types.ModuleType("era_bench.tasks.manager_based.common.utils")
    utils_mod.line_direction_xy_from_quat = lambda quat: quat[..., :2]
    sys.modules[utils_mod.__name__] = utils_mod

    commands_mod = types.ModuleType("era_bench.tasks.manager_based.arc_embodied.mdp.commands")
    commands_mod.LoadTaskCommand = object
    sys.modules[commands_mod.__name__] = commands_mod

    observations_mod = types.ModuleType("era_bench.tasks.manager_based.arc_embodied.mdp.observations")
    observations_mod.objects_grasped = lambda env: torch.zeros((env.num_envs, 4, 1), dtype=torch.bool)
    sys.modules[observations_mod.__name__] = observations_mod

    mdp_mod = types.ModuleType("isaaclab.envs.mdp")
    mdp_mod.time_out = lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    sys.modules[mdp_mod.__name__] = mdp_mod


def _load_module(name: str, relative_path: str):
    module_path = pathlib.Path(__file__).resolve().parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path.resolve())
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_stub_dependencies()
_buffer_module = _load_module(
    "era_bench.tasks.manager_based.arc_embodied.env.success_positions",
    "era_bench/tasks/manager_based/arc_embodied/env/success_positions.py",
)
_terminations = _load_module(
    "era_bench.tasks.manager_based.arc_embodied.mdp.terminations",
    "era_bench/tasks/manager_based/arc_embodied/mdp/terminations.py",
)
SuccessPositionBuffer = _buffer_module.SuccessPositionBuffer
ReserveCubeStackBuffer = _buffer_module.ReserveCubeStackBuffer
are_objects_in_success_positions = _terminations.are_objects_in_success_positions


class _DummyCommand:
    def __init__(self):
        self.completed_env_ids: list[int] = []
        self.current_grid_pairs = [types.SimpleNamespace(output=[[1]]), types.SimpleNamespace(output=[[1]])]

    def set_task_completed(self, env_id: int) -> None:
        self.completed_env_ids.append(env_id)


class _DummyCommandManager:
    def __init__(self):
        self.command = _DummyCommand()

    def get_term(self, name: str) -> _DummyCommand:
        assert name == "load_task"
        return self.command


class _DummyScene:
    def __init__(self, num_envs: int, tcp_positions_w: torch.Tensor):
        self.cfg = types.SimpleNamespace(
            cubes_size=(0.05, 0.05, 0.05),
            separation_distance=0.01,
            output_grid_center_local=(0.0, 0.0, 0.0),
            table_cam=types.SimpleNamespace(offset=types.SimpleNamespace(pos=(0.0, 10.0, 1.7))),
        )
        self.env_origins = torch.zeros((num_envs, 3))
        self._ee_frame = types.SimpleNamespace(data=types.SimpleNamespace(target_pos_w=tcp_positions_w.unsqueeze(1)))

    def __getitem__(self, name: str):
        assert name == "ee_frame"
        return self._ee_frame


class _DummyEnv:
    def __init__(self, success_positions, reserve_stack_poses, reserve_stack_colors, tcp_positions_w):
        self.num_envs = success_positions.num_envs
        self.device = torch.device("cpu")
        self.success_positions = success_positions
        self.success_positions_reached = torch.zeros_like(success_positions.active_targets)
        self._reserve_stack_poses = reserve_stack_poses
        self.reserve_cube_stacks = ReserveCubeStackBuffer(
            num_envs=self.num_envs,
            num_stacks=reserve_stack_poses.shape[1],
            cubes_per_stack=reserve_stack_poses.shape[2],
            device="cpu",
        )
        for env_id in range(self.num_envs):
            self.reserve_cube_stacks.replace(env_id, reserve_stack_poses[env_id], reserve_stack_colors[env_id])
        self.scene = _DummyScene(self.num_envs, tcp_positions_w)
        self.command_manager = _DummyCommandManager()

    def get_live_reserve_cube_stack_poses(self):
        return self._reserve_stack_poses


def _buffer_with(*entries: tuple[int, list[list[float]], list[int]]) -> SuccessPositionBuffer:
    buffer = SuccessPositionBuffer(num_envs=2, max_targets=4, device="cpu")
    for env_id, positions, color_ids in entries:
        buffer.replace(env_id, torch.tensor(positions), torch.tensor(color_ids))
    return buffer


def test_success_positions_are_environment_major_and_color_matched():
    blue, red = ARC_COLOR_TO_COLOR[1], ARC_COLOR_TO_COLOR[2]
    success_positions = _buffer_with(
        (0, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [2, 1]),
        (1, [[2.0, 0.0, 0.0]], [1]),
    )
    reserve_poses = torch.zeros((2, 4, 2, 7))
    reserve_poses[..., 3] = 1.0
    reserve_poses[0, 0, 0, :3] = torch.tensor([0.01, 0.0, 0.0])  # red
    reserve_poses[0, 1, 0, :3] = torch.tensor([1.01, 0.0, 0.0])  # blue
    reserve_poses[1, 2, 0, :3] = torch.tensor([2.01, 0.0, 0.0])  # blue
    colors = torch.tensor([[2, 1, 3, 4], [3, 4, 1, 2]])
    env = _DummyEnv(success_positions, reserve_poses, colors, torch.tensor([[5.0, -1.0, 0.0], [6.0, 1.0, 0.0]]))

    result = are_objects_in_success_positions(env, distance_threshold=0.05)

    assert torch.equal(result, torch.tensor([True, True]))
    assert env.command_manager.command.completed_env_ids == [0, 1]


def test_success_positions_fail_when_one_environment_target_is_far():
    blue = ARC_COLOR_TO_COLOR[1]
    success_positions = _buffer_with(
        (0, [[0.0, 0.0, 0.0]], [1]),
        (1, [[1.0, 0.0, 0.0]], [1]),
    )
    reserve_poses = torch.zeros((2, 4, 1, 7))
    reserve_poses[..., 3] = 1.0
    reserve_poses[0, 0, 0, :3] = torch.tensor([0.01, 0.0, 0.0])
    reserve_poses[1, 0, 0, :3] = torch.tensor([1.5, 0.0, 0.0])
    colors = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    env = _DummyEnv(success_positions, reserve_poses, colors, torch.tensor([[5.0, -1.0, 0.0], [6.0, -1.0, 0.0]]))

    result = are_objects_in_success_positions(env, distance_threshold=0.05)

    assert torch.equal(result, torch.tensor([True, False]))
    assert env.command_manager.command.completed_env_ids == [0]


def test_missing_reserve_color_fails_only_environments_that_require_it():
    success_positions = _buffer_with((1, [[0.0, 0.0, 0.0]], [3]))
    reserve_poses = torch.zeros((2, 4, 1, 7))
    reserve_poses[..., 3] = 1.0
    env = _DummyEnv(success_positions, reserve_poses, torch.tensor([[1, 2, 4, 5], [1, 2, 4, 5]]), torch.tensor([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]))

    result = are_objects_in_success_positions(env, distance_threshold=0.05)

    assert torch.equal(result, torch.tensor([True, False]))
    assert env.command_manager.command.completed_env_ids == [0]


def test_empty_success_positions_complete_immediately():
    success_positions = SuccessPositionBuffer(num_envs=2, max_targets=4, device="cpu")
    reserve_poses = torch.zeros((2, 4, 1, 7))
    reserve_poses[..., 3] = 1.0
    env = _DummyEnv(success_positions, reserve_poses, torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]), torch.tensor([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]))

    result = are_objects_in_success_positions(env, distance_threshold=0.05)

    assert torch.equal(result, torch.tensor([True, True]))
    assert env.command_manager.command.completed_env_ids == [0, 1]
