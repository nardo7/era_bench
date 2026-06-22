from collections.abc import Sequence
from enum import Enum
from typing import TypeGuard, cast

import warp as wp
import torch


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class MirrorStates(Enum):
    """States for the mirror task."""

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
    APPROACHING = (1, wp.constant(1), wp.constant(0.3))
    GRASPING = (2, wp.constant(2), wp.constant(0.1))
    MIRRORING = (3, wp.constant(3), wp.constant(0.5))
    PLACING = (4, wp.constant(4), wp.constant(0.1))
    RETREATING = (5, wp.constant(5), wp.constant(0.1))


# Warp kernels cannot consume Python Enum objects directly.
# Bridge enum members to plain module-level Warp constants for kernel use.
MIRROR_STATE_REST = MirrorStates.REST.warp_value
MIRROR_STATE_APPROACHING = MirrorStates.APPROACHING.warp_value
MIRROR_STATE_GRASPING = MirrorStates.GRASPING.warp_value
MIRROR_STATE_MIRRORING = MirrorStates.MIRRORING.warp_value
MIRROR_STATE_PLACING = MirrorStates.PLACING.warp_value
MIRROR_STATE_RETREATING = MirrorStates.RETREATING.warp_value


class MirrorSmWaitTime:
    """Wait times for the mirror task state machine."""

    REST = wp.constant(0.1)
    APPROACHING = wp.constant(0.3)
    GRASPING = wp.constant(0.1)
    MIRRORING = wp.constant(0.5)
    PLACING = wp.constant(0.1)
    RETREATING = wp.constant(0.1)


def is_vec_t(vector_coord: wp.vec3) -> TypeGuard[wp.Scalar]:
    return True


@wp.func
def distance_below_threshold(current_pos: wp.vec3f, desired_pos: wp.vec3f, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.func
def get_state_config(state: int) -> tuple[float, int]:
    """Get wait time and next state for current state."""
    if state == MIRROR_STATE_REST:  # REST
        return 0.1, MIRROR_STATE_APPROACHING
    elif state == MIRROR_STATE_APPROACHING:  # APPROACHING
        return 0.3, MIRROR_STATE_GRASPING
    elif state == MIRROR_STATE_GRASPING:  # GRASPING
        return 0.1, MIRROR_STATE_MIRRORING
    elif state == MIRROR_STATE_MIRRORING:  # MIRRORING
        return 0.5, MIRROR_STATE_PLACING
    elif state == MIRROR_STATE_PLACING:  # PLACING
        return 0.1, MIRROR_STATE_RETREATING
    else:  # RETREATING (state == MIRROR_STATE_RETREATING)
        return 0.1, MIRROR_STATE_REST


@wp.kernel
def mirror_state_machine_kernel(
    dt: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    sm_state: wp.array(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    sm_wait_time: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    ee_pose: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    object_poses: wp.array2d(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    desired_goal_positions: wp.array2d(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    des_ee_pose: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    des_gripper_state: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    offset: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    current_goal_objects: wp.array(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    position_threshold: float,
):
    tid = wp.tid()

    state = sm_state[tid]
    current_goal_object_idx = current_goal_objects[tid]

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
        )
        if state == 1 or state == 3  # APPROACHING or MIRRORING
        else condition
    )

    # Handle transition
    if condition:
        sm_state[tid] = next_state
        state = next_state  # update local variable for use in this step
        sm_wait_time[tid] = 0.0
        wp.printf("[INFO]: MirrorStateMachine: Transitioning to state %d for env %d\n", next_state, tid)

    if state == 5 and condition:  # RETREATING
        current_goal_objects[tid] = (current_goal_object_idx + 1) % wp.len(desired_goal_positions[tid])

    if state == MIRROR_STATE_REST:
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = ee_pose[tid]

    elif state == MIRROR_STATE_APPROACHING:
        des_gripper_state[tid] = GripperState.OPEN

        object_position = wp.transform_get_translation(object_poses[tid, current_goal_object_idx])
        desired_grasp_orientation = wp.transform_get_rotation(desired_goal_positions[tid, current_goal_object_idx])
        approach_pose = wp.transform(object_position, desired_grasp_orientation)
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], approach_pose)

    elif state == MIRROR_STATE_GRASPING:
        des_gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = ee_pose[tid]

    elif state == MIRROR_STATE_MIRRORING:
        des_gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], desired_goal_positions[tid, current_goal_object_idx])

    elif state == MIRROR_STATE_PLACING:
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = ee_pose[tid]

    elif state == MIRROR_STATE_RETREATING:
        des_gripper_state[tid] = GripperState.OPEN
        retreat_offset = offset[tid] * 3.0
        des_ee_pose[tid] = wp.transform_multiply(retreat_offset, ee_pose[tid])


class MirrorStateMachine:

    def __init__(
        self,
        dt: float,
        num_envs: int,
        dest_poses: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        position_threshold: float = 0.01,
    ):

        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # goals
        self.dest_poses = dest_poses[:, :, [0, 1, 2, 6, 3, 4, 5]]  # convert to (x, y, z, w) for warp kernel use

        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=device)
        self.sm_state = torch.full((self.num_envs,), MirrorStates.REST.python_value, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=device)
        self.current_idx_objects = torch.full((self.num_envs,), 0, dtype=torch.int32, device=device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=device)
        # set offset 0.1m above the object
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
        self.current_idx_objects_wp = wp.from_torch(self.current_idx_objects, wp.int32)
        self.dest_poses_wp = wp.from_torch(self.dest_poses, wp.transform)

    def advance(self, executed_steps: int) -> None:
        if executed_steps <= 0:
            return
        self.sm_wait_time += self.dt * float(executed_steps)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = cast(Sequence[int], slice(None))
        self.sm_state[env_ids] = MirrorStates.REST.python_value
        self.sm_wait_time[env_ids] = 0.0

    def get_current_grasped_object_idx(self) -> torch.Tensor:
        return self.current_idx_objects.clone()

    def compute(self, ee_pose: torch.Tensor, object_poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the next desired end-effector pose and gripper state based on the current state of the environment.

        Args:
            ee_pose (torch.Tensor): Current end-effector pose (num_envs, 7).
            object_poses (torch.Tensor): Current object poses (num_envs, num_objects, 7).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Desired end-effector pose (num_envs, 7) and gripper state (num_envs, 1) concatenated along the last dimension, and the current mirror state.
        """

        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_poses = object_poses[:, :, [0, 1, 2, 4, 5, 6, 3]]

        # copy to warp
        ee_pose_wp = wp.from_torch(ee_pose, wp.transform)
        object_poses_wp = wp.from_torch(object_poses, wp.transform)

        wp.launch(
            kernel=mirror_state_machine_kernel,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_poses_wp,
                self.dest_poses_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.current_idx_objects_wp,
                self.position_threshold,
            ],
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]  # convert back to (w, x, y, z)

        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1), self.sm_state
