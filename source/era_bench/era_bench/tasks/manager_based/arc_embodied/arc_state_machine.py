from collections.abc import Sequence
from enum import Enum
from typing import TypeGuard, cast

import warp as wp
import torch
from .config import Colors as ArcColors


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


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
    PLACING = (4, wp.constant(4), wp.constant(0.2))
    RETREATING = (5, wp.constant(5), wp.constant(0.1))

    APPROACHING_ABOVE = (6, wp.constant(6), wp.constant(0.2))
    RETREATING_ABOVE = (7, wp.constant(7), wp.constant(0.1))

    APPROACHING_PLACING_ABOVE = (8, wp.constant(8), wp.constant(0.2))
    RETREATING_PLACING_ABOVE = (9, wp.constant(9), wp.constant(0.1))


COLOR_TO_INDEX = {
    ArcColors.RED: 0,
    ArcColors.BLUE: 1,
    ArcColors.GREEN: 2,
    ArcColors.YELLOW: 3,
}

INDEX_TO_COLOR = {v: k for k, v in COLOR_TO_INDEX.items()}


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


def is_vec_t(vector_coord: wp.vec3) -> TypeGuard[wp.Scalar]:
    return True


@wp.func
def get_next_reserve_cube(
    next_idx: int, reserve_cube_stack: wp.array(dtype=wp.transform), num_reserve_cubes: int
) -> wp.transform:
    """Destack the next reserve cube index from the stack."""
    if next_idx < 0:
        next_idx = num_reserve_cubes - 1
    if next_idx >= num_reserve_cubes:
        next_idx = 0
    return reserve_cube_stack[next_idx]


@wp.func
def get_reserve_cube_stack(
    env_id: int, color_idx: int, reserve_cubes_positions: wp.array3d(dtype=wp.transform)
) -> wp.array(dtype=wp.transform):
    """Get the reserve cube stack for a given environment and color."""
    return reserve_cubes_positions[env_id, color_idx]


@wp.func
def distance_below_threshold(current_pos: wp.vec3f, desired_pos: wp.vec3f, threshold: float, state: int) -> bool:
    threshold = (
        threshold if state == ARC_STATE_APPROACHING or state == ARC_STATE_APPROACHING_PLACING else threshold * 2.0
    )  # double threshold for above states
    return wp.length(current_pos - desired_pos) < threshold


@wp.func
def get_state_config(state: int) -> tuple[float, int]:
    """Get wait time and next state for current state."""
    if state == ARC_STATE_REST:  # REST
        return 0.1, ARC_STATE_APPROACHING_ABOVE
    elif state == ARC_STATE_APPROACHING_ABOVE:  # APPROACHING_ABOVE
        return 0.2, ARC_STATE_APPROACHING
    elif state == ARC_STATE_APPROACHING:  # APPROACHING
        return 0.3, ARC_STATE_GRASPING
    elif state == ARC_STATE_GRASPING:  # GRASPING
        return 0.1, ARC_STATE_RETREATING_ABOVE
    elif state == ARC_STATE_RETREATING_ABOVE:  # RETREATING_ABOVE
        return 0.1, ARC_STATE_APPROACHING_PLACING_ABOVE
    elif state == ARC_STATE_APPROACHING_PLACING_ABOVE:  # APPROACHING_PLACING_ABOVE
        return 0.2, ARC_STATE_APPROACHING_PLACING
    elif state == ARC_STATE_APPROACHING_PLACING:  # APPROACHING_PLACING
        return 0.5, ARC_STATE_PLACING
    elif state == ARC_STATE_PLACING:  # PLACING
        return 0.1, ARC_STATE_RETREATING_PLACING_ABOVE
    elif state == ARC_STATE_RETREATING_PLACING_ABOVE:  # RETREATING_PLACING_ABOVE
        return 0.1, ARC_STATE_REST
    else:  # RETREATING (state == ARC_STATE_RETREATING)
        return 0.1, ARC_STATE_REST


@wp.kernel
def arc_state_machine_kernel(
    sm_state: wp.array(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    sm_wait_time: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    ee_pose: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    desired_goal_poses: wp.array2d(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    reserve_cube_poses_stacks: wp.array3d(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    colors_desired_goal_positions: wp.array2d(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    des_ee_pose: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    des_gripper_state: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    offset: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    retreat_offset: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    retreat_offset_reserve: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    current_goal_objects: wp.array(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    reserve_cube_poses_idx_wp: wp.array2d(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    position_threshold: float,
):
    tid = wp.tid()

    state = sm_state[tid]
    current_goal_object_idx = current_goal_objects[tid]
    color_idx = colors_desired_goal_positions[tid, current_goal_object_idx]
    reserve_cube_idx = reserve_cube_poses_idx_wp[tid, color_idx]
    # Get state configuration
    state_wait_time, next_state = get_state_config(state)

    # Check transition condition
    condition = sm_wait_time[tid] > state_wait_time
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
        else condition
    )

    # Handle transition
    if condition:
        sm_state[tid] = next_state
        state = next_state  # update local variable for use in this step
        sm_wait_time[tid] = 0.0
        wp.printf("[INFO]: ArcStateMachine: Transitioning to state %d for env %d\n", next_state, tid)

    if state == ARC_STATE_PLACING and condition:
        current_goal_objects[tid] = (current_goal_object_idx + 1) % wp.len(desired_goal_poses[tid])
        reserve_cube_poses_idx_wp[tid, color_idx] = reserve_cube_idx - 1

    if state == ARC_STATE_REST:
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = ee_pose[tid]

    elif state == ARC_STATE_APPROACHING_ABOVE:
        des_gripper_state[tid] = GripperState.OPEN
        current_stack = get_reserve_cube_stack(tid, color_idx, reserve_cube_poses_stacks)
        object_position = wp.transform_get_translation(
            get_next_reserve_cube(reserve_cube_idx, current_stack, wp.len(current_stack))
        )
        desired_grasp_orientation = wp.transform_get_rotation(desired_goal_poses[tid, current_goal_object_idx])
        approach_pose = wp.transform(object_position, desired_grasp_orientation)
        des_ee_pose[tid] = wp.transform_multiply(retreat_offset_reserve[tid], approach_pose)  # +0.2m above object

    elif state == ARC_STATE_APPROACHING:
        des_gripper_state[tid] = GripperState.OPEN
        current_stack = get_reserve_cube_stack(tid, color_idx, reserve_cube_poses_stacks)
        object_position = wp.transform_get_translation(
            get_next_reserve_cube(reserve_cube_idx, current_stack, wp.len(current_stack))
        )
        desired_grasp_orientation = wp.transform_get_rotation(desired_goal_poses[tid, current_goal_object_idx])
        approach_pose = wp.transform(object_position, desired_grasp_orientation)
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], approach_pose)  # +0.1m above object

    elif state == ARC_STATE_GRASPING:
        des_gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = ee_pose[tid]

    elif state == ARC_STATE_RETREATING_ABOVE:
        des_gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = wp.transform_multiply(retreat_offset_reserve[tid], ee_pose[tid])  # lift +0.2m

    elif state == ARC_STATE_APPROACHING_PLACING_ABOVE:
        des_gripper_state[tid] = GripperState.CLOSE

        des_ee_pose[tid] = wp.transform_multiply(retreat_offset[tid], desired_goal_poses[tid, current_goal_object_idx])

    elif state == ARC_STATE_APPROACHING_PLACING:
        des_gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], desired_goal_poses[tid, current_goal_object_idx])

    elif state == ARC_STATE_PLACING:
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = ee_pose[tid]

    elif state == ARC_STATE_RETREATING_PLACING_ABOVE:
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = wp.transform_multiply(retreat_offset[tid], ee_pose[tid])  # lift +0.2m after placing

    elif state == ARC_STATE_RETREATING:
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = wp.transform_multiply(retreat_offset[tid], ee_pose[tid])


class ArcStateMachine:

    @staticmethod
    def _validate_pose_tensor_shape(name: str, tensor: torch.Tensor, num_envs: int) -> None:
        if tensor.ndim != 3:
            raise ValueError(f"Expected {name} to have shape (num_envs, N, 7), got ndim={tensor.ndim}")
        if tensor.shape[0] != num_envs:
            raise ValueError(f"Expected {name}.shape[0] == num_envs ({num_envs}), got {tensor.shape[0]}")
        if tensor.shape[2] != 7:
            raise ValueError(f"Expected {name}.shape[2] == 7, got {tensor.shape[2]}")

    @staticmethod
    def _log_shape(name: str, tensor: torch.Tensor) -> None:
        print(f"[INFO]: {name} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")

    @staticmethod
    def _convert_pose_tensor_for_warp(pose_tensor: torch.Tensor) -> torch.Tensor:
        return pose_tensor[:, :, [0, 1, 2, 4, 5, 6, 3]]  # convert from (w, x, y, z) to (x, y, z, w) for warp kernel use

    @staticmethod
    def _flatten_color_pose_dict(
        poses_by_color: dict[ArcColors, torch.Tensor], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[ArcColors, ...]]:
        flattened_poses: list[torch.Tensor] = []
        flattened_colors: list[torch.Tensor] = []
        color_order: list[ArcColors] = []

        for color_name, pose_tensor in poses_by_color.items():
            if color_name not in COLOR_TO_INDEX:
                raise ValueError(f"Unsupported color '{color_name}'. Expected one of {tuple(COLOR_TO_INDEX)}")

            color_order.append(color_name)
            warp_pose_tensor = ArcStateMachine._convert_pose_tensor_for_warp(pose_tensor)
            flattened_poses.append(warp_pose_tensor)
            flattened_colors.append(
                torch.full(warp_pose_tensor.shape[:2], COLOR_TO_INDEX[color_name], dtype=torch.int32, device=device)
            )

        if not flattened_poses:
            raise ValueError("Expected at least one destination pose tensor.")

        return torch.cat(flattened_poses, dim=1), torch.cat(flattened_colors, dim=1), tuple(color_order)

    def __init__(
        self,
        dt: float,
        num_envs: int,
        dest_poses: dict[
            ArcColors, torch.Tensor
        ],  # color -> (num_envs, num_objects, 7) tensor of desired poses for each object
        reserve_cubes_poses: dict[
            ArcColors, torch.Tensor
        ],  # color -> (num_envs, num_reserve_cubes, 7) tensor of reserve cube positions
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

        for color_name, pose_tensor in dest_poses.items():
            self._validate_pose_tensor_shape(f"dest_poses[{color_name}]", pose_tensor, self.num_envs)
        for color_name, pose_tensor in reserve_cubes_poses.items():
            self._validate_pose_tensor_shape(f"reserve_cubes_poses[{color_name}]", pose_tensor, self.num_envs)

        if self.debug_shapes:
            for color_name, pose_tensor in dest_poses.items():
                self._log_shape(f"dest_poses[{color_name}]", pose_tensor)
            for color_name, pose_tensor in reserve_cubes_poses.items():
                self._log_shape(f"reserve_cubes_poses[{color_name}]", pose_tensor)

        # goals
        self.dest_poses, self.dest_pose_colors, self.dest_pose_color_order = self._flatten_color_pose_dict(
            dest_poses, device
        )
        # save reserve cubes positions
        self.reserve_cubes_poses = {k: self._convert_pose_tensor_for_warp(v) for k, v in reserve_cubes_poses.items()}
        reserve_cubes_stacked = torch.stack(list(self.reserve_cubes_poses.values()), dim=0).permute(1, 0, 2, 3)
        print(f"[INFO]: Reserve cube poses: {reserve_cubes_poses}")
        initial_stack_idx = next(iter(reserve_cubes_poses.values())).shape[1] - 1
        self.reserve_cube_poses_idx = torch.full(
            (self.num_envs, len(reserve_cubes_poses.keys())),
            initial_stack_idx,
            dtype=torch.int32,
            device=device,
        )

        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=device)
        self.sm_state = torch.full((self.num_envs,), ArcStates.REST.python_value, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=device)
        self.current_idx_objects = torch.full((self.num_envs,), 0, dtype=torch.int32, device=device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=device)
        self.retreat_offset = torch.zeros((self.num_envs, 7), device=device)
        self.retreat_offset_reserve = torch.zeros((self.num_envs, 7), device=device)

        cube_stacks_max_height = cubes_per_stack * cube_sizes[-1]  # max height of stacked cubes
        print(f"[INFO]: Cube stacks max height: {cube_stacks_max_height}")

        self.offset[:, 2] = 0
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)
        self.retreat_offset[:, 2] = 0.2  # retreat 0.2m above the current position
        self.retreat_offset[:, -1] = 1.0  # convert to (x, y, z, w) for warp kernel use
        self.retreat_offset_reserve[:, 2] = (
            cube_sizes[-1] * 0.5 + cube_stacks_max_height
        )  # retreat 0.2m above the reserve cube stack
        self.retreat_offset_reserve[:, -1] = 1.0  # convert

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
        self.retreat_offset_wp = wp.from_torch(self.retreat_offset, wp.transform)
        self.current_idx_objects_wp = wp.from_torch(self.current_idx_objects, wp.int32)
        self.dest_poses_wp = wp.from_torch(self.dest_poses, wp.transform)
        self.dest_pose_colors_wp = wp.from_torch(self.dest_pose_colors, wp.int32)
        self.reserve_cubes_poses_wp = wp.from_torch(reserve_cubes_stacked, wp.transform)
        self.reserve_cube_poses_idx_wp = wp.from_torch(self.reserve_cube_poses_idx, wp.int32)
        self.retreat_offset_reserve_wp = wp.from_torch(self.retreat_offset_reserve, wp.transform)

        if self.debug_shapes:
            self._log_shape("dest_poses_warp_order", self.dest_poses)
            self._log_shape("dest_pose_colors", self.dest_pose_colors)
            self._log_shape("reserve_cube_poses_idx", self.reserve_cube_poses_idx)
            self._log_shape("reserve_cubes_stacked_warp_order", reserve_cubes_stacked)
            print(f"[INFO]: sm_state_wp shape={self.sm_state_wp.shape}")
            print(f"[INFO]: sm_wait_time_wp shape={self.sm_wait_time_wp.shape}")
            print(f"[INFO]: dest_poses_wp shape={self.dest_poses_wp.shape}")
            print(f"[INFO]: dest_pose_colors_wp shape={self.dest_pose_colors_wp.shape}")
            print(f"[INFO]: reserve_cubes_poses_wp shape={self.reserve_cubes_poses_wp.shape}")
            print(f"[INFO]: reserve_cube_poses_idx_wp shape={self.reserve_cube_poses_idx_wp.shape}")

    def advance(self, executed_steps: int) -> None:
        if executed_steps <= 0:
            return
        self.sm_wait_time += self.dt * float(executed_steps)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = cast(Sequence[int], slice(None))
        self.sm_state[env_ids] = ArcStates.REST.python_value
        self.sm_wait_time[env_ids] = 0.0

    def get_current_grasped_object_idx(self) -> torch.Tensor:
        return self.current_idx_objects.clone()

    def get_current_goal_color_idx(self, env_id: int) -> int:
        """Get the color of the current goal object for a specific environment."""
        return int(self.dest_pose_colors[env_id, self.current_idx_objects[env_id]].item())

    def get_current_goal_color(self, env_id: int) -> ArcColors:
        """Get the color of the current goal object for a specific environment."""
        color_idx = self.get_current_goal_color_idx(env_id)
        return INDEX_TO_COLOR[color_idx]

    def get_current_stack_index(self, env_id: int) -> int:
        """Get the current index of the reserve cube stack for a specific environment and color."""
        color_idx = self.get_current_goal_color_idx(env_id)
        return int(self.reserve_cube_poses_idx[env_id, color_idx].item())

    def compute(self, ee_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the next desired end-effector pose and gripper state based on the current state of the environment.

        Args:
            ee_pose (torch.Tensor): Current end-effector pose (num_envs, 7).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Desired end-effector pose (num_envs, 7) and gripper state (num_envs, 1) concatenated along the last dimension, and the current arc state.
        """

        if ee_pose.ndim != 2 or ee_pose.shape[0] != self.num_envs or ee_pose.shape[1] != 7:
            raise ValueError(f"Expected ee_pose shape ({self.num_envs}, 7), got {tuple(ee_pose.shape)}")

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
            inputs=[
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                self.dest_poses_wp,
                self.reserve_cubes_poses_wp,
                self.dest_pose_colors_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.retreat_offset_wp,
                self.retreat_offset_reserve_wp,
                self.current_idx_objects_wp,
                self.reserve_cube_poses_idx_wp,
                self.position_threshold,
            ],
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]  # convert back to (w, x, y, z)

        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1), self.sm_state
