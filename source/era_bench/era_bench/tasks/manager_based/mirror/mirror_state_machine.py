from collections.abc import Sequence
from typing import TypeIs, cast

import warp as wp
import torch


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class MirrorStates:
    """States for the mirror task."""

    REST = wp.constant(0.0)
    APPROACHING = wp.constant(1.0)
    GRASPING = wp.constant(2.0)
    MIRRORING = wp.constant(3.0)
    RETREATING = wp.constant(4.0)


class MirrorSmWaitTime:
    """Wait times for the mirror task state machine."""

    REST = wp.constant(0.5)
    APPROACHING = wp.constant(0.5)
    GRASPING = wp.constant(0.1)
    MIRRORING = wp.constant(0.8)
    RETREATING = wp.constant(0.1)


def is_vec_t(vector_coord: wp.vec3) -> TypeIs[wp.Scalar]:
    return True


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    # compensate for fixed Z offset between EE sensor frame and IK target frame (Franka gripper length ~0.103m)

    adjusted = wp.vec3(current_pos[0], current_pos[1], current_pos[2] + 0.103)  # pyright: ignore[reportOperatorIssue]

    return wp.length(adjusted - desired_pos) < threshold


class MirrorStateMachine:

    def __init__(self, dt: float, num_envs: int, device: torch.device = "cpu", position_threshold: float = 0.01):

        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=device)

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
        # self.current_objects_wp = wp.from_torch(self.current_objects, wp.int32)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = cast(Sequence[int], slice(None))
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(
        self, ee_pose: torch.Tensor, object_poses: torch.Tensor, desired_goal_positions: torch.Tensor
    ) -> torch.Tensor:
        pass  # TODO: implement the state machine logic here
