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

    @property
    def python_value(self) -> int:
        return self._python_value

    @property
    def warp_value(self) -> object:
        return self.value

    def __new__(cls, python_value: int, warp_value: object):
        obj = object.__new__(cls)
        obj._value_ = warp_value
        obj._python_value = python_value
        return obj

    REST = (0, wp.constant(0.0))
    APPROACHING = (1, wp.constant(1.0))
    GRASPING = (2, wp.constant(2.0))
    MIRRORING = (3, wp.constant(3.0))
    PLACING = (4, wp.constant(4.0))
    RETREATING = (5, wp.constant(5.0))


class MirrorSmWaitTime:
    """Wait times for the mirror task state machine."""

    REST = wp.constant(0.5)
    APPROACHING = wp.constant(0.5)
    GRASPING = wp.constant(0.1)
    MIRRORING = wp.constant(0.8)
    PLACING = wp.constant(0.1)
    RETREATING = wp.constant(0.1)


def is_vec_t(vector_coord: wp.vec3) -> TypeGuard[wp.Scalar]:
    return True


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    # compensate for fixed Z offset between EE sensor frame and IK target frame (Franka gripper length ~0.103m)

    adjusted = wp.vec3(current_pos[0], current_pos[1], current_pos[2] + 0.103)  # pyright: ignore[reportOperatorIssue]

    return wp.length(adjusted - desired_pos) < threshold


@wp.func
def compensate_fingertip(ee_pose: wp.transform) -> wp.transform:
    # compensate for fixed Z offset between EE sensor frame and IK target frame (Franka gripper length ~0.103m)
    compensation = wp.transform(wp.vec3(0.0, 0.0, -0.103), wp.quat_identity())
    return wp.transform_multiply(ee_pose, compensation)


@wp.kernel
def mirror_state_machine_kernel(
    dt: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    sm_state: wp.array(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    sm_wait_time: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    ee_pose: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    object_poses: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    desired_goal_positions: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    des_ee_pose: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    des_gripper_state: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    offset: wp.array(dtype=wp.transform),  # pyright: ignore[reportInvalidTypeForm]
    current_goal_objects: wp.array(dtype=int),  # pyright: ignore[reportInvalidTypeForm]
    position_threshold: float,
):
    tid = wp.tid()

    state = sm_state[tid]

    current_goal_object_idx = current_goal_objects[tid]

    if state == MirrorStates.REST:
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = compensate_fingertip(ee_pose[tid])

        if sm_wait_time[tid] > MirrorSmWaitTime.REST:
            sm_state[tid] = MirrorStates.APPROACHING
            sm_wait_time[tid] = 0.0
            wp.printf("[INFO]: MirrorStateMachine: Transitioning to APPROACHING state for env %d\n", tid)

    elif state == MirrorStates.APPROACHING:
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = object_poses[tid][current_goal_object_idx]
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], des_ee_pose[tid])

        if (
            distance_below_threshold(
                wp.transform_get_translation(ee_pose[tid]),
                wp.transform_get_translation(des_ee_pose[tid]),
                position_threshold,
            )
            and sm_wait_time[tid] > MirrorSmWaitTime.APPROACHING
        ):
            sm_state[tid] = MirrorStates.GRASPING
            sm_wait_time[tid] = 0.0
            wp.printf("[INFO]: MirrorStateMachine: Transitioning to GRASPING state for env %d\n", tid)

    elif state == MirrorStates.GRASPING:
        des_gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = compensate_fingertip(ee_pose[tid])

        if sm_wait_time[tid] > MirrorSmWaitTime.GRASPING:
            sm_state[tid] = MirrorStates.MIRRORING
            sm_wait_time[tid] = 0.0
            wp.printf("[INFO]: MirrorStateMachine: Transitioning to MIRRORING state for env %d\n", tid)

    elif state == MirrorStates.MIRRORING:
        des_gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = desired_goal_positions[tid][current_goal_object_idx]
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], des_ee_pose[tid])

        if (
            distance_below_threshold(
                wp.transform_get_translation(ee_pose[tid]),
                wp.transform_get_translation(des_ee_pose[tid]),
                position_threshold,
            )
            and sm_wait_time[tid] > MirrorSmWaitTime.MIRRORING
        ):
            sm_state[tid] = MirrorStates.PLACING
            sm_wait_time[tid] = 0.0
            wp.printf("[INFO]: MirrorStateMachine: Transitioning to PLACING state for env %d\n", tid)

    elif state == MirrorStates.PLACING:
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = compensate_fingertip(ee_pose[tid])

        if sm_wait_time[tid] > MirrorSmWaitTime.PLACING:
            sm_state[tid] = MirrorStates.RETREATING
            sm_wait_time[tid] = 0.0
            wp.printf("[INFO]: MirrorStateMachine: Transitioning to RETREATING state for env %d\n", tid)

    elif state == MirrorStates.RETREATING:
        des_gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = wp.transform(offset[tid] * 3, des_ee_pose[tid])

        if sm_wait_time[tid] > MirrorSmWaitTime.RETREATING:
            sm_state[tid] = MirrorStates.REST
            sm_wait_time[tid] = 0.0
            wp.printf("[INFO]: MirrorStateMachine: Transitioning to REST state for env %d\n", tid)

            # update to next object
            current_goal_objects[tid] = (current_goal_object_idx + 1) % wp.length(desired_goal_positions[tid])

    sm_wait_time[tid] += dt[tid]


class MirrorStateMachine:

    def __init__(
        self, dt: float, num_envs: int, device: torch.device = torch.device("cpu"), position_threshold: float = 0.01
    ):

        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=device)
        self.sm_state = torch.full((self.num_envs,), MirrorStates.REST.python_value, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=device)
        self.current_goal_objects = torch.full((self.num_envs,), 0, dtype=torch.int32, device=device)

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
        self.current_goal_objects_wp = wp.from_torch(self.current_goal_objects, wp.int32)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = cast(Sequence[int], slice(None))
        self.sm_state[env_ids] = MirrorStates.REST.python_value
        self.sm_wait_time[env_ids] = 0.0

    def compute(
        self, ee_pose: torch.Tensor, object_poses: torch.Tensor, desired_goal_positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute the next desired end-effector pose and gripper state based on the current state of the environment.

        Args:
            ee_pose (torch.Tensor): Current end-effector pose (num_envs, 7).
            object_poses (torch.Tensor): Current object poses (num_envs, num_objects, 7).
            desired_goal_positions (torch.Tensor): Desired goal positions for the objects (num_envs, num_objects, 7).

        Returns:
            torch.Tensor: Desired end-effector pose (num_envs, 7) and gripper state (num_envs, 1) concatenated along the last dimension.
        """

        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_poses = object_poses[:, :, [0, 1, 2, 4, 5, 6, 3]]
        desired_goal_positions = desired_goal_positions[:, :, [0, 1, 2, 4, 5, 6, 3]]

        # copy to warp
        ee_pose_wp = wp.from_torch(ee_pose, wp.transform)
        object_poses_wp = wp.from_torch(object_poses, wp.transform)
        desired_goal_positions_wp = wp.from_torch(desired_goal_positions, wp.transform)

        wp.launch(
            kernel=mirror_state_machine_kernel,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_poses_wp,
                desired_goal_positions_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.current_goal_objects_wp,
                self.position_threshold,
            ],
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]  # convert back to (w, x, y, z)

        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)
