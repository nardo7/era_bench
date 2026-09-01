import importlib.util
import pathlib
import sys

import torch
import warp as wp
from arc_utils import ARC_COLOR_TO_COLOR


ARC_TASK_ROOT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "era_bench"
    / "tasks"
    / "manager_based"
    / "arc_embodied"
)


def _load_module(name: str, filename: str):
    module_path = ARC_TASK_ROOT / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_buffer_module = _load_module("success_positions", "env/success_positions.py")
_state_machine_module = _load_module("arc_state_machine", "control/state_machine.py")
wp.init()
SuccessPositionBuffer = _buffer_module.SuccessPositionBuffer
ReserveCubeStackBuffer = _buffer_module.ReserveCubeStackBuffer
ArcStateMachine = _state_machine_module.ArcStateMachine
ArcControlMode = _state_machine_module.ArcControlMode
ArcStates = _state_machine_module.ArcStates


def _reserve_stacks(num_envs: int, num_cubes: int) -> ReserveCubeStackBuffer:
    buffer = ReserveCubeStackBuffer(num_envs=num_envs, num_stacks=4, cubes_per_stack=num_cubes, device="cpu")
    poses = torch.zeros((4, num_cubes, 7))
    poses[..., 3] = 1.0
    buffer.replace(0, poses, torch.tensor([2, 1, 3, 4]))
    buffer.replace(1, poses, torch.tensor([3, 4, 1, 2]))
    return buffer


def test_state_machine_consumes_environment_major_success_positions():
    success_positions = SuccessPositionBuffer(num_envs=2, max_targets=4, device="cpu")
    success_positions.replace(0, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([2, 1]))
    success_positions.replace(1, torch.tensor([[7.0, 8.0, 9.0]]), torch.tensor([1]))

    state_machine = ArcStateMachine(
        dt=0.01,
        num_envs=2,
        success_positions=success_positions,
        reserve_cube_stacks=_reserve_stacks(2, 2),
        cube_sizes=(0.04, 0.04, 0.04),
        cubes_per_stack=1,
        device=torch.device("cpu"),
    )

    assert torch.equal(state_machine.dest_counts, torch.tensor([2, 1], dtype=torch.int32))
    assert torch.equal(state_machine.dest_poses[0, :2, :3], success_positions[0].positions_w)
    assert torch.equal(state_machine.dest_arc_color_ids[0, :2], torch.tensor([2, 1], dtype=torch.int32))
    assert torch.equal(state_machine.dest_arc_color_ids[1, :1], torch.tensor([1], dtype=torch.int32))

    success_positions.replace(0, torch.tensor([[10.0, 11.0, 12.0]]), torch.tensor([1]))
    state_machine.update_success_positions(success_positions, [0])

    assert state_machine.dest_counts[0].item() == 1
    assert torch.equal(state_machine.dest_poses[0, 0, :3], torch.tensor([10.0, 11.0, 12.0]))
    assert state_machine.dest_arc_color_ids[0, 0].item() == 1
    assert state_machine.dest_arc_color_ids[0, 1].item() == 0
    assert state_machine.reserve_cube_poses.shape == (2, 4, 2, 7)
    assert state_machine.get_current_cube_slot(0) == (1, 1)


def test_empty_output_holds_the_initial_rest_pose():
    success_positions = SuccessPositionBuffer(num_envs=2, max_targets=4, device="cpu")
    state_machine = ArcStateMachine(
        dt=0.01,
        num_envs=2,
        success_positions=success_positions,
        reserve_cube_stacks=_reserve_stacks(2, 2),
        cube_sizes=(0.04, 0.04, 0.04),
        cubes_per_stack=1,
        device=torch.device("cpu"),
    )
    initial_ee_pose = torch.tensor(
        [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]]
    )

    next_pose, state = state_machine.compute(initial_ee_pose)

    assert torch.equal(state, torch.zeros(2, dtype=torch.int32))
    assert torch.equal(next_pose[:, :7], initial_ee_pose)
    assert torch.equal(next_pose[:, 7], torch.ones(2))

    moved_ee_pose = initial_ee_pose.clone()
    moved_ee_pose[:, :3] += 10.0
    next_pose, state = state_machine.compute(moved_ee_pose)

    assert torch.equal(state, torch.zeros(2, dtype=torch.int32))
    assert torch.equal(next_pose[:, :7], initial_ee_pose)

    state_machine.reset_idx([0])
    next_pose, _ = state_machine.compute(moved_ee_pose)

    assert torch.equal(next_pose[0, :7], moved_ee_pose[0])
    assert torch.equal(next_pose[1, :7], initial_ee_pose[1])


def test_action_metadata_is_derived_from_state_and_current_cube_slot():
    success_positions = SuccessPositionBuffer(num_envs=2, max_targets=4, device="cpu")
    success_positions.replace(0, torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([2]))
    success_positions.replace(1, torch.tensor([[4.0, 5.0, 6.0]]), torch.tensor([1]))
    state_machine = ArcStateMachine(
        dt=0.01,
        num_envs=2,
        success_positions=success_positions,
        reserve_cube_stacks=_reserve_stacks(2, 2),
        cube_sizes=(0.04, 0.04, 0.04),
        cubes_per_stack=1,
        device=torch.device("cpu"),
    )
    state_machine.sm_state[:] = torch.tensor(
        [ArcStates.APPROACHING_ABOVE.python_value, ArcStates.APPROACHING_PLACING.python_value],
        dtype=torch.int32,
    )

    metadata = state_machine.compute_action_metadata()

    assert metadata[0].tolist() == [
        ArcControlMode.CUROBO.value,
        ArcStates.APPROACHING_ABOVE.python_value,
        -1,
        -1,
    ]
    assert metadata[1].tolist() == [
        ArcControlMode.DIFF_IK.value,
        ArcStates.APPROACHING_PLACING.python_value,
        2,
        1,
    ]


def test_transition_gate_defers_only_the_requested_environment():
    success_positions = SuccessPositionBuffer(num_envs=2, max_targets=4, device="cpu")
    success_positions.replace(0, torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([2]))
    success_positions.replace(1, torch.tensor([[4.0, 5.0, 6.0]]), torch.tensor([1]))
    state_machine = ArcStateMachine(
        dt=0.01,
        num_envs=2,
        success_positions=success_positions,
        reserve_cube_stacks=_reserve_stacks(2, 2),
        cube_sizes=(0.04, 0.04, 0.04),
        cubes_per_stack=1,
        device=torch.device("cpu"),
    )
    ee_pose = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    )

    state_machine.advance(6)
    _, state = state_machine.compute(ee_pose, transition_allowed=torch.tensor([False, True]))

    assert state.tolist() == [ArcStates.REST.python_value, ArcStates.APPROACHING_ABOVE.python_value]


def test_final_target_returns_to_rest_without_wrapping_to_the_first_target():
    success_positions = SuccessPositionBuffer(num_envs=2, max_targets=4, device="cpu")
    success_positions.replace(0, torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([2]))
    state_machine = ArcStateMachine(
        dt=0.01,
        num_envs=2,
        success_positions=success_positions,
        reserve_cube_stacks=_reserve_stacks(2, 2),
        cube_sizes=(0.04, 0.04, 0.04),
        cubes_per_stack=1,
        device=torch.device("cpu"),
    )
    rest_ee_pose = torch.tensor(
        [[7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    )

    # Simulate the last retreat reaching its destination.  Warp stores
    # quaternions in xyzw order, while the public API uses wxyz.
    state_machine.sm_state[0] = ArcStates.RETREATING_PLACING_ABOVE.python_value
    state_machine.sm_wait_time[0] = 0.1
    state_machine.rest_pose[0] = torch.tensor([7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0])
    state_machine.rest_pose_initialized[0] = 1
    state_machine.des_ee_pose[0] = state_machine.rest_pose[0]

    next_pose, state = state_machine.compute(rest_ee_pose)

    assert state[0].item() == ArcStates.REST.python_value
    assert state_machine.current_idx_objects[0].item() == 1
    assert torch.equal(next_pose[0, :7], rest_ee_pose[0])

    state_machine.advance(100)
    _, state = state_machine.compute(rest_ee_pose)

    assert state[0].item() == ArcStates.REST.python_value


def test_intermediate_target_leaves_rest_for_the_next_target():
    success_positions = SuccessPositionBuffer(num_envs=2, max_targets=4, device="cpu")
    success_positions.replace(
        0,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([2, 1]),
    )
    state_machine = ArcStateMachine(
        dt=0.01,
        num_envs=2,
        success_positions=success_positions,
        reserve_cube_stacks=_reserve_stacks(2, 2),
        cube_sizes=(0.04, 0.04, 0.04),
        cubes_per_stack=1,
        device=torch.device("cpu"),
    )
    rest_ee_pose = torch.tensor(
        [[7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    )

    # Complete the retreat after placing the first of two targets.
    state_machine.sm_state[0] = ArcStates.RETREATING_PLACING_ABOVE.python_value
    state_machine.sm_wait_time[0] = 0.1
    state_machine.rest_pose[0] = torch.tensor([7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0])
    state_machine.rest_pose_initialized[0] = 1
    state_machine.des_ee_pose[0] = state_machine.rest_pose[0]

    _, state = state_machine.compute(rest_ee_pose)

    assert state[0].item() == ArcStates.REST.python_value
    assert state_machine.current_idx_objects[0].item() == 1

    # REST must start the next pick instead of treating goal index 1 as the
    # completed sentinel for this two-target task.
    state_machine.advance(100)
    _, state = state_machine.compute(rest_ee_pose)

    assert state[0].item() == ArcStates.APPROACHING_ABOVE.python_value
    assert state_machine.current_idx_objects[0].item() == 1
