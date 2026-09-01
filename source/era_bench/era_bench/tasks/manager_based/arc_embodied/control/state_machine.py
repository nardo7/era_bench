from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, cast

import warp as wp
import torch
from arc_utils import ARC_COLOR_TO_COLOR, Colors as ArcColors

if TYPE_CHECKING:
    from ..env.success_positions import ReserveCubeStackBuffer, SuccessPositionBuffer


NUM_ARC_COLORS = max(ARC_COLOR_TO_COLOR) + 1


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    RELEASE = wp.constant(0.0)
    CLOSE = wp.constant(-1.0)


class ArcControlMode(Enum):
    """Arm controller selected by the ARC state machine."""

    DIFF_IK = 0
    CUROBO = 1


class ArcStates(Enum):
    """States for the Arc task."""

    _python_value: int
    _wait_time: float

    @property
    def python_value(self) -> int:
        return self._python_value

    @property
    def warp_value(self) -> int:
        return self.value

    @property
    def wait_time(self) -> float:
        return self._wait_time

    def __new__(cls, python_value: int, warp_value: int, wait_time: float = 0.1):
        obj = object.__new__(cls)
        obj._value_ = warp_value
        obj._python_value = python_value
        obj._wait_time = wait_time
        return obj

    REST = (0, wp.constant(0), wp.constant(0.1))
    APPROACHING = (1, wp.constant(1), wp.constant(0.1))
    GRASPING = (2, wp.constant(2), wp.constant(0.3))
    APPROACHING_PLACING = (3, wp.constant(3), wp.constant(0.5))
    PLACING = (4, wp.constant(4), wp.constant(0.3))
    RETREATING = (5, wp.constant(5), wp.constant(0.1))

    APPROACHING_ABOVE = (6, wp.constant(6), wp.constant(0.2))
    RETREATING_ABOVE = (7, wp.constant(7), wp.constant(0.1))

    APPROACHING_PLACING_ABOVE = (8, wp.constant(8), wp.constant(0.2))
    RETREATING_PLACING_ABOVE = (9, wp.constant(9), wp.constant(0.1))


# Warp kernels cannot consume Python Enum objects directly.
# Bridge enum members to plain module-level Warp constants for kernel use.
ARC_STATE_REST = ArcStates.REST.warp_value
ARC_STATE_APPROACHING = ArcStates.APPROACHING.warp_value
ARC_STATE_GRASPING = ArcStates.GRASPING.warp_value
ARC_STATE_APPROACHING_PLACING = ArcStates.APPROACHING_PLACING.warp_value
ARC_STATE_PLACING = ArcStates.PLACING.warp_value
ARC_STATE_RETREATING = ArcStates.RETREATING.warp_value
ARC_STATE_APPROACHING_ABOVE = ArcStates.APPROACHING_ABOVE.warp_value
ARC_STATE_RETREATING_ABOVE = ArcStates.RETREATING_ABOVE.warp_value
ARC_STATE_APPROACHING_PLACING_ABOVE = ArcStates.APPROACHING_PLACING_ABOVE.warp_value
ARC_STATE_RETREATING_PLACING_ABOVE = ArcStates.RETREATING_PLACING_ABOVE.warp_value


@wp.func
def get_next_reserve_cube(
    env_id: int,
    stack_idx: int,
    next_idx: int,
    reserve_cubes_positions: wp.array3d(dtype=wp.transform),
    num_reserve_cubes: int,
) -> wp.transform:
    """Destack the next reserve cube index from the stack."""
    if next_idx < 0:
        next_idx = num_reserve_cubes - 1
    if next_idx >= num_reserve_cubes:
        next_idx = 0
    return reserve_cubes_positions[env_id, stack_idx, next_idx]


@wp.func
def distance_below_threshold(current_pos: wp.vec3f, desired_pos: wp.vec3f, threshold: float, state: int) -> bool:
    threshold = threshold  # if state == ARC_STATE_APPROACHING or state == ARC_STATE_APPROACHING_PLACING else threshold * 2.0  # double threshold for above states
    return wp.length(current_pos - desired_pos) < threshold


@wp.func
def get_state_config(state: int) -> tuple[float, int]:
    """Get wait time and next state for current state."""
    if state == ARC_STATE_REST:  # REST
        return 0.05, ARC_STATE_APPROACHING_ABOVE
    elif state == ARC_STATE_APPROACHING_ABOVE:  # APPROACHING_ABOVE
        return 0.1, ARC_STATE_APPROACHING
    elif state == ARC_STATE_APPROACHING:  # APPROACHING
        return 0.3, ARC_STATE_GRASPING
    elif state == ARC_STATE_GRASPING:  # GRASPING
        return 0.3, ARC_STATE_RETREATING_ABOVE
    elif state == ARC_STATE_RETREATING_ABOVE:  # RETREATING_ABOVE
        return 0.05, ARC_STATE_APPROACHING_PLACING_ABOVE
    elif state == ARC_STATE_APPROACHING_PLACING_ABOVE:  # APPROACHING_PLACING_ABOVE
        return 0.1, ARC_STATE_APPROACHING_PLACING
    elif state == ARC_STATE_APPROACHING_PLACING:  # APPROACHING_PLACING
        return 0.3, ARC_STATE_PLACING
    elif state == ARC_STATE_PLACING:  # PLACING
        return 0.3, ARC_STATE_RETREATING_PLACING_ABOVE
    elif state == ARC_STATE_RETREATING_PLACING_ABOVE:  # RETREATING_PLACING_ABOVE
        return 0.05, ARC_STATE_REST
    else:  # RETREATING (state == ARC_STATE_RETREATING)
        return 0.05, ARC_STATE_REST


@wp.kernel
def arc_state_machine_kernel(
    sm_state: wp.array(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    sm_wait_time: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    transition_allowed: wp.array(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    ee_pose: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    rest_pose: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    rest_pose_initialized: wp.array(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    desired_goal_poses: wp.array2d(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    num_desired_goal_poses: wp.array(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    reserve_cube_poses_stacks: wp.array3d(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    reserve_cubes_per_stack: int,
    color_to_stack: wp.array2d(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    goal_arc_color_ids: wp.array2d(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    des_ee_pose: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    des_gripper_state: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    offset: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    retreat_offset: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    retreat_offset_reserve: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    retreat_from_position: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    current_goal_objects: wp.array(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    reserve_cube_indices: wp.array2d(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    position_threshold: float,
):
    tid = wp.tid()

    state = sm_state[tid]
    num_desired_goals = num_desired_goal_poses[tid]

    # A goal index equal to the target count is the completed sentinel.  It is
    # set only after the final placement retreat reaches REST, so the arm
    # returns to its captured rest pose and cannot wrap around to the first
    # output target while the success termination is waiting to fire.
    if current_goal_objects[tid] >= num_desired_goals:
        sm_state[tid] = ARC_STATE_REST
        sm_wait_time[tid] = 0.0
        if rest_pose_initialized[tid] == 0:
            rest_pose[tid] = ee_pose[tid]
            rest_pose_initialized[tid] = 1
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = rest_pose[tid]
        return

    if state == ARC_STATE_REST and rest_pose_initialized[tid] == 0:
        rest_pose[tid] = ee_pose[tid]
        rest_pose_initialized[tid] = 1

    current_goal_object_idx = current_goal_objects[tid]
    goal_color_id = goal_arc_color_ids[tid, current_goal_object_idx]
    stack_idx = color_to_stack[tid, goal_color_id]
    reserve_cube_idx = reserve_cube_indices[tid, stack_idx]
    # Get state configuration
    state_wait_time, next_state = get_state_config(state)

    # Check transition condition
    # The caller may temporarily defer a transition while its arm controller
    # completes a motion.  This remains controller-neutral: the state machine
    # only receives a per-environment permission bit.
    condition = transition_allowed[tid] != 0 and sm_wait_time[tid] > state_wait_time
    condition = (
        condition
        and distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
            state,
        )
        if state == ARC_STATE_APPROACHING_ABOVE
        or state == ARC_STATE_APPROACHING
        or state == ARC_STATE_APPROACHING_PLACING_ABOVE
        or state == ARC_STATE_APPROACHING_PLACING  # approach states require distance check
        or state == ARC_STATE_RETREATING_ABOVE
        or state == ARC_STATE_RETREATING_PLACING_ABOVE  # retreat states require distance check
        else condition
    )

    # Handle transition
    if condition:
        sm_state[tid] = next_state
        state = next_state  # update local variable for use in this step
        sm_wait_time[tid] = 0.0
        wp.printf("[INFO]: ArcStateMachine: Transitioning to state %d for env %d\n", next_state, tid)

        # save position to retreat from after grasping/placing ONLY DURING TRANSITION
        # Use the current end-effector pose (ee_pose) rather than the last desired
        # pose so that transitions which only change the gripper state do not
        # cause the arm to move toward a previously requested target. This keeps
        # the arm stable when the intent is to only close/open the gripper.
        if (
            state == ARC_STATE_GRASPING
            or state == ARC_STATE_PLACING
            or state == ARC_STATE_RETREATING_ABOVE
            or state == ARC_STATE_RETREATING_PLACING_ABOVE
        ):
            retreat_from_position[tid] = ee_pose[tid]

    # Keep the selected physical cube slot valid throughout PLACING: the
    # motion planner still needs that object as its attached collision body.
    # Pop the stack after the gripper releases the cube, but keep that goal
    # selected throughout the retreat. Once the retreat reaches REST, advance
    # to the next goal or to the completed sentinel after the final goal.
    if state == ARC_STATE_RETREATING_PLACING_ABOVE and condition:
        reserve_cube_indices[tid, stack_idx] = reserve_cube_idx - 1

    if state == ARC_STATE_REST and condition:
        current_goal_objects[tid] = current_goal_object_idx + 1

    if state == ARC_STATE_REST:
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = rest_pose[tid]

    elif state == ARC_STATE_APPROACHING_ABOVE:
        des_gripper_state[tid] = GripperState.OPEN
        object_position = wp.transform_get_translation(
            get_next_reserve_cube(tid, stack_idx, reserve_cube_idx, reserve_cube_poses_stacks, reserve_cubes_per_stack)
        )
        desired_grasp_orientation = wp.transform_get_rotation(desired_goal_poses[tid, current_goal_object_idx])
        approach_pose = wp.transform(object_position, desired_grasp_orientation)
        des_ee_pose[tid] = wp.transform_multiply(retreat_offset_reserve[tid], approach_pose)  # +0.2m above object

    elif state == ARC_STATE_APPROACHING:
        des_gripper_state[tid] = GripperState.OPEN
        object_position = wp.transform_get_translation(
            get_next_reserve_cube(tid, stack_idx, reserve_cube_idx, reserve_cube_poses_stacks, reserve_cubes_per_stack)
        )
        desired_grasp_orientation = wp.transform_get_rotation(desired_goal_poses[tid, current_goal_object_idx])
        approach_pose = wp.transform(object_position, desired_grasp_orientation)
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], approach_pose)  # +0.1m above object

    elif state == ARC_STATE_GRASPING:
        des_gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = retreat_from_position[tid]  # hold position after grasping

    elif state == ARC_STATE_RETREATING_ABOVE:
        des_gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = wp.transform_multiply(retreat_offset_reserve[tid], retreat_from_position[tid])  # lift +0.2m

    elif state == ARC_STATE_APPROACHING_PLACING_ABOVE:
        des_gripper_state[tid] = GripperState.CLOSE

        des_ee_pose[tid] = wp.transform_multiply(retreat_offset[tid], desired_goal_poses[tid, current_goal_object_idx])

    elif state == ARC_STATE_APPROACHING_PLACING:
        des_gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], desired_goal_poses[tid, current_goal_object_idx])

    elif state == ARC_STATE_PLACING:
        des_gripper_state[tid] = GripperState.RELEASE
        des_ee_pose[tid] = retreat_from_position[tid]  # hold position after placing

    elif state == ARC_STATE_RETREATING_PLACING_ABOVE:
        des_gripper_state[tid] = GripperState.RELEASE
        des_ee_pose[tid] = wp.transform_multiply(
            retreat_offset[tid], retreat_from_position[tid]
        )  # lift +0.2m after placing


class ArcStateMachine:

    @staticmethod
    def _validate_stack_pose_tensor_shape(name: str, tensor: torch.Tensor, num_envs: int) -> None:
        if tensor.ndim != 4:
            raise ValueError(f"Expected {name} to have shape (num_envs, stacks, cubes, 7), got ndim={tensor.ndim}")
        if tensor.shape[0] != num_envs:
            raise ValueError(f"Expected {name}.shape[0] == num_envs ({num_envs}), got {tensor.shape[0]}")
        if tensor.shape[3] != 7:
            raise ValueError(f"Expected {name}.shape[3] == 7, got {tensor.shape[3]}")

    @staticmethod
    def _log_shape(name: str, tensor: torch.Tensor) -> None:
        print(f"[INFO]: {name} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")

    @staticmethod
    def _convert_pose_tensor_for_warp(pose_tensor: torch.Tensor) -> torch.Tensor:
        return pose_tensor[:, :, :, [0, 1, 2, 4, 5, 6, 3]]  # convert from (w, x, y, z) to Warp ordering

    def __init__(
        self,
        dt: float,
        num_envs: int,
        success_positions: SuccessPositionBuffer,
        reserve_cube_stacks: ReserveCubeStackBuffer,
        cube_sizes: tuple[float, float, float],
        cubes_per_stack: int,
        device: torch.device = torch.device("cpu"),
        position_threshold: float = 0.01,
        debug_shapes: bool = False,
    ):

        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.debug_shapes = debug_shapes

        if success_positions.num_envs != self.num_envs:
            raise ValueError(
                f"Expected success positions for {self.num_envs} environments, got {success_positions.num_envs}."
            )
        if reserve_cube_stacks.num_envs != self.num_envs:
            raise ValueError("Reserve stack buffer has a different number of environments.")
        self._validate_stack_pose_tensor_shape(
            "reserve_cube_stacks.poses_w", reserve_cube_stacks.poses_w, self.num_envs
        )
        self.num_stacks = reserve_cube_stacks.num_stacks
        num_reserve_cubes = reserve_cube_stacks.cubes_per_stack
        self.num_reserve_cubes = num_reserve_cubes

        # Goals are environment-major and use Warp quaternion ordering
        # ``(x, y, z, qx, qy, qz, qw)``. ARC color IDs remain unchanged.
        self.dest_poses = torch.zeros(
            (self.num_envs, success_positions.max_targets, 7), dtype=torch.float32, device=device
        )
        self.dest_poses[:, :, 3] = 1.0
        self.dest_arc_color_ids = torch.zeros(
            (self.num_envs, success_positions.max_targets), dtype=torch.int32, device=device
        )
        self.dest_counts = torch.zeros((self.num_envs,), dtype=torch.int32, device=device)

        self.reserve_cube_poses = torch.zeros(
            (self.num_envs, self.num_stacks, num_reserve_cubes, 7), dtype=torch.float32, device=device
        )
        self.reserve_cube_poses[..., 6] = 1.0
        self.color_to_stack = torch.full((self.num_envs, NUM_ARC_COLORS), -1, dtype=torch.int32, device=device)
        self.initial_stack_idx = num_reserve_cubes - 1
        self.reserve_cube_indices = torch.full(
            (self.num_envs, self.num_stacks),
            self.initial_stack_idx,
            dtype=torch.int32,
            device=device,
        )

        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=device)
        self.sm_state = torch.full((self.num_envs,), ArcStates.REST.python_value, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=device)
        self.current_idx_objects = torch.full((self.num_envs,), 0, dtype=torch.int32, device=device)
        self.rest_pose = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=device)
        self.rest_pose[:, -1] = 1.0
        self.rest_pose_initialized = torch.zeros((self.num_envs,), dtype=torch.int32, device=device)

        self.update_reserve_cube_stacks(reserve_cube_stacks)
        self.update_success_positions(success_positions)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=device)
        self.retreat_offset = torch.zeros((self.num_envs, 7), device=device)
        self.retreat_offset_reserve = torch.zeros((self.num_envs, 7), device=device)
        self.retreat_from_position = torch.zeros((self.num_envs, 7), device=device)

        cube_stacks_max_height = cubes_per_stack * cube_sizes[-1]  # max height of stacked cubes
        print(f"[INFO]: Cube stacks max height: {cube_stacks_max_height}")

        self.offset[:, 2] = 0
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)
        self.retreat_offset[:, 2] = cube_stacks_max_height  # retreat above the cube stack
        self.retreat_offset[:, -1] = 1.0  # convert to (x, y, z, w) for warp kernel use
        self.retreat_offset_reserve[:, 2] = (
            cube_sizes[-1] + cube_stacks_max_height + 0.02
        )  # retreat 0.2m above the reserve cube stack
        self.retreat_offset_reserve[:, -1] = 1.0  # convert

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.rest_pose_wp = wp.from_torch(self.rest_pose, wp.transform)
        self.rest_pose_initialized_wp = wp.from_torch(self.rest_pose_initialized, wp.int32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
        self.retreat_offset_wp = wp.from_torch(self.retreat_offset, wp.transform)
        self.current_idx_objects_wp = wp.from_torch(self.current_idx_objects, wp.int32)
        self.dest_poses_wp = wp.from_torch(self.dest_poses, wp.transform)
        self.dest_counts_wp = wp.from_torch(self.dest_counts, wp.int32)
        self.dest_arc_color_ids_wp = wp.from_torch(self.dest_arc_color_ids, wp.int32)
        self.reserve_cubes_poses_wp = wp.from_torch(self.reserve_cube_poses, wp.transform)
        self.color_to_stack_wp = wp.from_torch(self.color_to_stack, wp.int32)
        self.reserve_cube_indices_wp = wp.from_torch(self.reserve_cube_indices, wp.int32)
        self.retreat_offset_reserve_wp = wp.from_torch(self.retreat_offset_reserve, wp.transform)
        self.retreat_from_position_wp = wp.from_torch(self.retreat_from_position, wp.transform)

        if self.debug_shapes:
            self._log_shape("dest_poses_warp_order", self.dest_poses)
            self._log_shape("dest_counts", self.dest_counts)
            self._log_shape("dest_arc_color_ids", self.dest_arc_color_ids)
            self._log_shape("reserve_cube_indices", self.reserve_cube_indices)
            self._log_shape("reserve_cubes_stacked_warp_order", self.reserve_cube_poses)
            print(f"[INFO]: sm_state_wp shape={self.sm_state_wp.shape}")
            print(f"[INFO]: sm_wait_time_wp shape={self.sm_wait_time_wp.shape}")
            print(f"[INFO]: dest_poses_wp shape={self.dest_poses_wp.shape}")
            print(f"[INFO]: dest_arc_color_ids_wp shape={self.dest_arc_color_ids_wp.shape}")
            print(f"[INFO]: reserve_cubes_poses_wp shape={self.reserve_cubes_poses_wp.shape}")
            print(f"[INFO]: reserve_cube_indices_wp shape={self.reserve_cube_indices_wp.shape}")

    def update_success_positions(
        self,
        success_positions: SuccessPositionBuffer,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Refresh state-machine goals from the environment-major target buffer."""
        if success_positions.num_envs != self.num_envs:
            raise ValueError(
                f"Expected success positions for {self.num_envs} environments, got {success_positions.num_envs}."
            )
        if success_positions.max_targets != self.dest_poses.shape[1]:
            raise ValueError(f"Expected {self.dest_poses.shape[1]} target slots, got {success_positions.max_targets}.")

        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return

        active_targets = success_positions.active_targets[env_ids_tensor].to(self.device)
        arc_color_ids = success_positions.color_ids[env_ids_tensor].to(self.device)
        color_to_stack = self.color_to_stack[env_ids_tensor].to(device=self.device)
        stack_indices = torch.gather(color_to_stack, 1, arc_color_ids.to(dtype=torch.long))
        missing_reserve_stacks = active_targets & (stack_indices < 0)
        if torch.any(missing_reserve_stacks):
            missing_arc_color_ids = torch.unique(arc_color_ids[missing_reserve_stacks]).tolist()
            raise ValueError(f"Success targets use ARC color IDs without reserve cube stacks: {missing_arc_color_ids}.")

        # The pose orientation is initialized once in ``__init__``. Entries past
        # ``dest_counts`` are intentionally left untouched because Warp never
        # reads them.
        self.dest_poses[env_ids_tensor, :, :3] = success_positions.positions_w[env_ids_tensor].to(self.device)
        self.dest_arc_color_ids[env_ids_tensor] = arc_color_ids.to(dtype=torch.int32)
        self.dest_counts[env_ids_tensor] = success_positions.counts[env_ids_tensor].to(
            device=self.device, dtype=self.dest_counts.dtype
        )

    def update_reserve_cube_stacks(
        self,
        reserve_cube_stacks: ReserveCubeStackBuffer,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Copy reset-time physical stack poses and color mappings into the SM."""
        if reserve_cube_stacks.num_envs != self.num_envs:
            raise ValueError("Reserve stack buffer has a different number of environments.")
        if reserve_cube_stacks.poses_w.shape != (self.num_envs, self.num_stacks, self.initial_stack_idx + 1, 7):
            raise ValueError("Reserve stack buffer shape does not match this state machine.")
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return
        self.reserve_cube_poses[env_ids_tensor] = self._convert_pose_tensor_for_warp(
            reserve_cube_stacks.poses_w[env_ids_tensor]
        ).to(self.device)
        self.color_to_stack[env_ids_tensor] = reserve_cube_stacks.color_to_stack[env_ids_tensor].to(
            device=self.device, dtype=self.color_to_stack.dtype
        )

    def advance(self, executed_steps: int) -> None:
        if executed_steps <= 0:
            return
        self.sm_wait_time += self.dt * float(executed_steps)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = cast(Sequence[int], slice(None))
        self.sm_state[env_ids] = ArcStates.REST.python_value
        self.sm_wait_time[env_ids] = 0.0
        self.current_idx_objects[env_ids] = 0
        self.rest_pose_initialized[env_ids] = 0
        self.reserve_cube_indices[env_ids, :] = self.initial_stack_idx

    def get_current_grasped_object_idx(self) -> torch.Tensor:
        return self.current_idx_objects.clone()

    def get_current_goal_color_idx(self, env_id: int) -> int:
        """Get the ARC color ID of the current goal object for an environment."""
        return int(self.dest_arc_color_ids[env_id, self.current_idx_objects[env_id]].item())

    def get_current_goal_color(self, env_id: int) -> ArcColors:
        """Get the color of the current goal object for a specific environment."""
        arc_color_id = self.get_current_goal_color_idx(env_id)
        return ARC_COLOR_TO_COLOR[arc_color_id]

    def get_current_cube_slot(self, env_id: int) -> tuple[int, int]:
        """Return the static physical stack and cube indices selected for the current goal."""
        arc_color_id = self.get_current_goal_color_idx(env_id)
        stack_index = int(self.color_to_stack[env_id, arc_color_id].item())
        if stack_index < 0:
            raise KeyError(f"No reserve stack is assigned ARC color ID {arc_color_id} for environment {env_id}.")
        return stack_index, int(self.reserve_cube_indices[env_id, stack_index].item())

    def compute_action_metadata(self) -> torch.Tensor:
        """Derive control mode, plan ID, and attachment slot from the current ARC state.

        The returned columns are ``[control_mode, plan_id, attachment_stack,
        attachment_cube]``. A state value is a sufficient plan ID because every
        cuRobo state transition changes the goal, while returning to the same
        state necessarily passes through a DiffIK state and therefore starts a
        fresh request in the hybrid action term.
        """
        states = self.sm_state.to(dtype=torch.long)
        metadata = torch.full((self.num_envs, 4), -1.0, dtype=torch.float32, device=self.device)

        curobo_mask = (
            (states == ArcStates.APPROACHING_ABOVE.python_value)
            | (states == ArcStates.APPROACHING.python_value)
            | (states == ArcStates.APPROACHING_PLACING_ABOVE.python_value)
        )
        metadata[:, 0] = curobo_mask.to(metadata.dtype) * ArcControlMode.CUROBO.value
        metadata[:, 1] = states.to(metadata.dtype)

        attached_mask = (
            (states == ArcStates.APPROACHING_PLACING_ABOVE.python_value)
            | (states == ArcStates.APPROACHING_PLACING.python_value)
            | (states == ArcStates.PLACING.python_value)
        )
        attached_env_ids = torch.nonzero(attached_mask, as_tuple=False).flatten()
        if attached_env_ids.numel() > 0:
            goal_indices = self.current_idx_objects[attached_env_ids].to(dtype=torch.long)
            color_ids = self.dest_arc_color_ids[attached_env_ids, goal_indices].to(dtype=torch.long)
            stack_indices = self.color_to_stack[attached_env_ids, color_ids].to(dtype=torch.long)
            cube_indices = self.reserve_cube_indices[attached_env_ids, stack_indices].to(dtype=torch.long)
            metadata[attached_env_ids, 2] = stack_indices.to(metadata.dtype)
            metadata[attached_env_ids, 3] = cube_indices.to(metadata.dtype)

        return metadata

    def compute(
        self,
        ee_pose: torch.Tensor,
        transition_allowed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the next desired end-effector pose and gripper state based on the current state of the environment.

        Args:
            ee_pose (torch.Tensor): Current end-effector pose (num_envs, 7).
            transition_allowed: Per-environment permission to leave the current
                state. ``None`` preserves the original, always-allow behavior.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Desired end-effector pose (num_envs, 7) and gripper state (num_envs, 1) concatenated along the last dimension, and the current arc state.
        """

        if ee_pose.ndim != 2 or ee_pose.shape[0] != self.num_envs or ee_pose.shape[1] != 7:
            raise ValueError(f"Expected ee_pose shape ({self.num_envs}, 7), got {tuple(ee_pose.shape)}")
        if transition_allowed is None:
            transition_allowed = torch.ones((self.num_envs,), dtype=torch.int32, device=self.device)
        else:
            if transition_allowed.ndim != 1 or transition_allowed.shape[0] != self.num_envs:
                raise ValueError(
                    f"Expected transition_allowed shape ({self.num_envs},), got {tuple(transition_allowed.shape)}"
                )
            transition_allowed = transition_allowed.to(device=self.device, dtype=torch.int32)
        transition_allowed_wp = wp.from_torch(transition_allowed, wp.int32)

        if self.debug_shapes:
            self._log_shape("ee_pose_input", ee_pose)

        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        if self.debug_shapes:
            self._log_shape("ee_pose_warp_order", ee_pose)

        # copy to warp
        ee_pose_wp = wp.from_torch(ee_pose, wp.transform)

        if self.debug_shapes:
            print(f"[INFO]: ee_pose_wp shape={ee_pose_wp.shape}")

        wp.launch(
            kernel=arc_state_machine_kernel,
            dim=self.num_envs,
            device=str(self.device),
            inputs=[
                self.sm_state_wp,
                self.sm_wait_time_wp,
                transition_allowed_wp,
                ee_pose_wp,
                self.rest_pose_wp,
                self.rest_pose_initialized_wp,
                self.dest_poses_wp,
                self.dest_counts_wp,
                self.reserve_cubes_poses_wp,
                self.num_reserve_cubes,
                self.color_to_stack_wp,
                self.dest_arc_color_ids_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.retreat_offset_wp,
                self.retreat_offset_reserve_wp,
                self.retreat_from_position_wp,
                self.current_idx_objects_wp,
                self.reserve_cube_indices_wp,
                self.position_threshold,
            ],
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]  # convert back to (w, x, y, z)

        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1), self.sm_state
