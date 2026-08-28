import importlib.util
import pathlib
import sys

import torch


def _load_buffer_module():
    module_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "era_bench"
        / "tasks"
        / "manager_based"
        / "arc_embodied"
        / "env"
        / "success_positions.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location("success_positions", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_buffer_module = _load_buffer_module()
SuccessPositionBuffer = _buffer_module.SuccessPositionBuffer
ReserveCubeStackBuffer = _buffer_module.ReserveCubeStackBuffer


def test_environment_view_keeps_all_targets_and_colors():
    buffer = SuccessPositionBuffer(num_envs=2, max_targets=4, device="cpu")
    buffer.replace(
        1,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([2, 1]),
    )

    env_targets = buffer[1]

    assert torch.equal(env_targets.positions_w, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert torch.equal(env_targets.color_ids, torch.tensor([2, 1]))
    assert torch.equal(buffer.counts, torch.tensor([0, 2]))


def test_replacing_one_environment_clears_its_stale_targets_only():
    buffer = SuccessPositionBuffer(num_envs=2, max_targets=4, device="cpu")
    buffer.replace(0, torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), torch.tensor([1, 2]))
    buffer.replace(1, torch.tensor([[3.0, 0.0, 0.0]]), torch.tensor([3]))

    buffer.replace(0, torch.tensor([[4.0, 0.0, 0.0]]), torch.tensor([4]))

    assert torch.equal(buffer[0].positions_w, torch.tensor([[4.0, 0.0, 0.0]]))
    assert torch.equal(buffer[0].color_ids, torch.tensor([4]))
    assert torch.equal(buffer[1].positions_w, torch.tensor([[3.0, 0.0, 0.0]]))
    assert torch.equal(buffer.counts, torch.tensor([1, 1]))


def test_reserve_stack_buffer_keeps_environment_specific_color_mappings():
    buffer = ReserveCubeStackBuffer(num_envs=2, num_stacks=4, cubes_per_stack=2, device="cpu")
    poses = torch.zeros((4, 2, 7))
    poses[..., 3] = 1.0
    buffer.replace(0, poses, torch.tensor([2, 1, 3, 4]))
    buffer.replace(1, poses + 1.0, torch.tensor([5, 6, 1, 2]))

    assert buffer[0].stack_index_for_color_id(1) == 1
    assert buffer[1].stack_index_for_color_id(1) == 2
    assert buffer.color_to_stack[0, 2].item() == 0
    assert buffer.color_to_stack[1, 6].item() == 1
    assert torch.equal(buffer[0].poses_w, poses)
