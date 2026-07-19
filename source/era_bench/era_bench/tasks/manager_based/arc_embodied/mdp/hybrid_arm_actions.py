"""Per-environment hybrid DiffIK/cuRobo arm action for ARC embodied tasks.

The arm term consumes ``[target_pose_w(7), control_mode, plan_id,
attachment_slot(2)]``. The separately configured gripper term is appended by
the action manager, yielding the complete environment action
``[target_pose_w(7), control_mode, plan_id, attachment_slot(2), target_gripper]``.

cuRobo trajectories are applied as direct arm joint targets. They are never
converted back to end-effector poses or passed through differential IK.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlannerCfg

from era_bench.curobo_planner_extended import ExtendedCuroboPlanner
from ..arc_state_machine import ArcControlMode
from ..scene_cfg import FRANKA_GRASP_CENTER_OFFSET


TARGET_POSE_SLICE = slice(0, 7)
CONTROL_MODE_INDEX = 7
PLAN_ID_INDEX = 8
ATTACHMENT_STACK_INDEX = 9
ATTACHMENT_CUBE_INDEX = 10
HYBRID_ARM_ACTION_DIM = 11


class _SelectiveDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Add per-environment application to Isaac Lab's batched DiffIK term."""

    def apply_actions(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or env_ids.numel() == self.num_envs:
            super().apply_actions()
            return
        if env_ids.numel() == 0:
            return

        ee_pos, ee_quat = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat[env_ids].norm() == 0:
            joint_pos_des = joint_pos[env_ids].clone()
        else:
            jacobian = self._compute_frame_jacobian()
            controller = self._ik_controller
            full_ee_pos_des = controller.ee_pos_des
            full_ee_quat_des = controller.ee_quat_des
            try:
                controller.ee_pos_des = full_ee_pos_des[env_ids]
                controller.ee_quat_des = full_ee_quat_des[env_ids]
                joint_pos_des = controller.compute(
                    ee_pos[env_ids],
                    ee_quat[env_ids],
                    jacobian[env_ids],
                    joint_pos[env_ids],
                )
            finally:
                controller.ee_pos_des = full_ee_pos_des
                controller.ee_quat_des = full_ee_quat_des

        self._asset.set_joint_position_target(joint_pos_des, joint_ids=self._joint_ids, env_ids=env_ids)


class HybridArmAction(ActionTerm):
    """Select and manage direct cuRobo or DiffIK control per environment."""

    cfg: HybridArmActionCfg

    def __init__(self, cfg: HybridArmActionCfg, env) -> None:
        super().__init__(cfg, env)
        if cfg.diff_ik_cfg.asset_name != cfg.asset_name:
            raise ValueError("The nested DiffIK config must target the HybridArmAction asset.")
        if cfg.diff_ik_cfg.controller.command_type != "pose" or cfg.diff_ik_cfg.controller.use_relative_mode:
            raise ValueError("HybridArmAction requires absolute 7D pose DiffIK commands.")

        self._diff_ik = _SelectiveDifferentialInverseKinematicsAction(cfg.diff_ik_cfg, env)
        self._robot = self._asset
        if not isinstance(self._robot, Articulation):
            raise TypeError("HybridArmAction requires an Articulation asset.")

        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._control_modes = torch.full(
            (self.num_envs,), ArcControlMode.DIFF_IK.value, dtype=torch.long, device=self.device
        )
        self._active_plan_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._plan_cursor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._plans: list[torch.Tensor | None] = [None] * self.num_envs
        self._planners: dict[int, ExtendedCuroboPlanner] = {}

        self.curobo_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.curobo_finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.curobo_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._curobo_joint_targets = self._robot.data.joint_pos[:, self._diff_ik._joint_ids].clone()

        self._hold_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._held_pose_actions = torch.zeros((self.num_envs, 7), device=self.device)
        self._held_joint_targets = self._curobo_joint_targets.clone()

        # Constructing the planner also initializes and warms up its MotionGen.
        # Do that while the environment starts rather than stalling the first
        # control step that escalates from DiffIK to cuRobo.
        with torch.inference_mode(False):
            for env_id in range(self.num_envs):
                self._planner_for_env(env_id)

    @property
    def action_dim(self) -> int:
        return HYBRID_ARM_ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def motion_in_progress(self) -> torch.Tensor:
        """Per-environment true while direct cuRobo replay still owns the arm."""
        return self.curobo_active

    @property
    def motion_finished(self) -> torch.Tensor:
        """Per-environment true after cuRobo successfully emitted its final waypoint."""
        return self.curobo_finished

    @property
    def motion_failed(self) -> torch.Tensor:
        """Per-environment true when the requested cuRobo plan could not be created."""
        return self.curobo_failed

    @property
    def diff_ik_env_mask(self) -> torch.Tensor:
        return self._control_modes == ArcControlMode.DIFF_IK.value

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Expected hybrid arm actions with shape {(self.num_envs, self.action_dim)}, got {tuple(actions.shape)}."
            )
        actions = actions.to(device=self.device, dtype=self._raw_actions.dtype)
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

        modes = actions[:, CONTROL_MODE_INDEX].to(dtype=torch.long)
        valid_modes = (modes == ArcControlMode.DIFF_IK.value) | (modes == ArcControlMode.CUROBO.value)
        if not bool(torch.all(valid_modes)):
            raise ValueError(f"Unsupported hybrid control modes: {torch.unique(modes[~valid_modes]).tolist()}.")
        plan_ids = actions[:, PLAN_ID_INDEX].to(dtype=torch.long)
        attachment_slots = actions[:, ATTACHMENT_STACK_INDEX : ATTACHMENT_CUBE_INDEX + 1].to(dtype=torch.long)
        valid_attachment_slots = ((attachment_slots == -1).all(dim=1)) | ((attachment_slots >= 0).all(dim=1))
        if not bool(torch.all(valid_attachment_slots)):
            raise ValueError("An attachment slot must contain two valid indices or [-1, -1].")

        self._control_modes[:] = ArcControlMode.DIFF_IK.value

        for env_id in range(self.num_envs):
            plan_matches = bool(plan_ids[env_id] == self._active_plan_ids[env_id])
            curobo_requested = modes[env_id] == ArcControlMode.CUROBO.value
            active_plan = plan_matches and self._plans[env_id] is not None and not bool(self.curobo_failed[env_id])

            if bool(curobo_requested) and not plan_matches:
                self._start_curobo_plan(env_id, int(plan_ids[env_id].item()), attachment_slots[env_id], actions[env_id])
                if not bool(self.curobo_failed[env_id]):
                    self._control_modes[env_id] = ArcControlMode.CUROBO.value
            elif active_plan:
                self._advance_curobo_plan(env_id)
                self._control_modes[env_id] = ArcControlMode.CUROBO.value
            elif not bool(curobo_requested) and self._active_plan_ids[env_id] >= 0:
                self._cancel_curobo(torch.tensor([env_id], device=self.device, dtype=torch.long))

        target_poses_w = actions[:, TARGET_POSE_SLICE]
        target_poses_b = self._target_poses_in_robot_frame(target_poses_w)
        if self.diff_ik_env_mask.any():
            self._diff_ik.process_actions(target_poses_b)
        self._update_pose_holds(target_poses_b)

    def apply_actions(self) -> None:
        diff_ik_motion_mask = self.diff_ik_env_mask & ~self._hold_active
        diff_ik_env_ids = torch.nonzero(diff_ik_motion_mask, as_tuple=False).flatten()
        self._diff_ik.apply_actions(diff_ik_env_ids)

        curobo_env_ids = torch.nonzero(
            self._control_modes == ArcControlMode.CUROBO.value, as_tuple=False
        ).flatten()
        if curobo_env_ids.numel() > 0:
            self._robot.set_joint_position_target(
                self._curobo_joint_targets[curobo_env_ids],
                joint_ids=self._diff_ik._joint_ids,
                env_ids=curobo_env_ids,
            )

        held_env_ids = torch.nonzero(self._hold_active, as_tuple=False).flatten()
        if held_env_ids.numel() > 0:
            held_targets = self._held_joint_targets[held_env_ids]
            self._robot.set_joint_position_target(
                held_targets,
                joint_ids=self._diff_ik._joint_ids,
                env_ids=held_env_ids,
            )
            self._robot.write_joint_state_to_sim(
                held_targets,
                torch.zeros_like(held_targets),
                joint_ids=self._diff_ik._joint_ids,
                env_ids=held_env_ids,
            )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._diff_ik.reset(env_ids)
        self._cancel_curobo(env_ids)
        self._control_modes[env_ids] = ArcControlMode.DIFF_IK.value
        self._hold_active[env_ids] = False
        self._held_pose_actions[env_ids] = 0.0
        self._held_joint_targets[env_ids] = self._robot.data.joint_pos[env_ids][:, self._diff_ik._joint_ids]
        self._curobo_joint_targets[env_ids] = self._held_joint_targets[env_ids]

    def _update_pose_holds(self, target_poses: torch.Tensor) -> None:
        curobo_mask = self._control_modes == ArcControlMode.CUROBO.value
        self._hold_active[curobo_mask] = False

        same_held_pose = torch.all(torch.abs(target_poses - self._held_pose_actions) <= 1.0e-6, dim=1)
        changed_hold_mask = self._hold_active & ~same_held_pose
        self._hold_active[changed_hold_mask] = False

        candidate_mask = self.diff_ik_env_mask & ~self._hold_active
        candidate_env_ids = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_env_ids.numel() == 0:
            return
        current_pos, current_quat = self._diff_ik._compute_frame_pose()
        position_error = torch.linalg.vector_norm(
            target_poses[candidate_env_ids, :3] - current_pos[candidate_env_ids], dim=1
        )
        orientation_error = math_utils.quat_error_magnitude(
            target_poses[candidate_env_ids, 3:7], current_quat[candidate_env_ids]
        )
        reached_mask = (position_error <= self.cfg.hold_position_tolerance) & (
            orientation_error <= self.cfg.hold_orientation_tolerance
        )
        reached_env_ids = candidate_env_ids[reached_mask]
        if reached_env_ids.numel() == 0:
            return
        self._hold_active[reached_env_ids] = True
        self._held_pose_actions[reached_env_ids] = target_poses[reached_env_ids]
        self._held_joint_targets[reached_env_ids] = self._robot.data.joint_pos[reached_env_ids][
            :, self._diff_ik._joint_ids
        ]

    def _start_curobo_plan(
        self,
        env_id: int,
        plan_id: int,
        attachment_slot: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        self._cancel_curobo(env_ids)
        self._active_plan_ids[env_id] = plan_id
        # Agents commonly run under torch.inference_mode(), while cuRobo's
        # optimizer needs autograd during planning.
        with torch.inference_mode(False):
            planner = self._planner_for_env(env_id)
            planner.set_expected_attached_object(self._attachment_name(attachment_slot, env_id))
            target_pose_env = action[TARGET_POSE_SLICE].detach().clone()
            target_pose_env[:3] -= self._env.scene.env_origins[env_id]
            success = planner.plan_grasp_center_goal_with_fixed_yaw_candidates(
                target_pose_env, contact_window_fraction=self.cfg.contact_window_fraction
            )
        plan = planner.current_plan_with_repeated_end_pose
        if not success or plan is None or plan.position is None or len(plan.position) == 0 or plan.joint_names is None:
            self.curobo_failed[env_id] = True
            return
        self._plans[env_id] = self._plan_in_action_joint_order(plan.position, list(plan.joint_names))
        self._plan_cursor[env_id] = 0
        self.curobo_active[env_id] = True
        self._curobo_joint_targets[env_id] = self._plans[env_id][0]

    def _advance_curobo_plan(self, env_id: int) -> None:
        plan = self._plans[env_id]
        if plan is None or self.curobo_failed[env_id]:
            return
        cursor = int(self._plan_cursor[env_id].item())
        if cursor + 1 < len(plan):
            cursor += 1
            self._plan_cursor[env_id] = cursor
            self._curobo_joint_targets[env_id] = plan[cursor]
            return
        self.curobo_active[env_id] = False
        self.curobo_finished[env_id] = True
        self._curobo_joint_targets[env_id] = plan[-1]

    def _cancel_curobo(self, env_ids: torch.Tensor) -> None:
        for env_id in env_ids.tolist():
            self._plans[env_id] = None
            self._plan_cursor[env_id] = 0
            self._active_plan_ids[env_id] = -1
            self.curobo_active[env_id] = False
            self.curobo_finished[env_id] = False
            self.curobo_failed[env_id] = False
            planner = self._planners.get(env_id)
            if planner is not None:
                planner.reset_plan()

    def _attachment_name(self, attachment_slot: torch.Tensor, env_id: int) -> str | None:
        stack_idx, cube_idx = attachment_slot.tolist()
        if stack_idx < 0:
            return None
        return self._env.scene.cfg.get_reserve_cube_name(stack_idx, cube_idx)

    def _target_poses_in_robot_frame(self, target_poses_w: torch.Tensor) -> torch.Tensor:
        """Convert state-machine world-frame targets to the frame DiffIK consumes."""
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            target_poses_w[:, :3],
            target_poses_w[:, 3:7],
        )
        return torch.cat((target_pos_b, target_quat_b), dim=-1)

    def _plan_in_action_joint_order(self, positions: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
        joint_indices = []
        for joint_name in self._diff_ik._joint_names:
            try:
                joint_indices.append(joint_names.index(joint_name))
            except ValueError as exc:
                raise ValueError(f"cuRobo plan did not contain action joint '{joint_name}'.") from exc
        return positions[:, joint_indices].to(device=self.device, dtype=self._raw_actions.dtype)

    def _planner_for_env(self, env_id: int) -> ExtendedCuroboPlanner:
        planner = self._planners.get(env_id)
        if planner is not None:
            return planner

        planner_cfg = CuroboPlannerCfg.from_task_name(self.cfg.curobo_task_name)
        planner_cfg.visualize_plan = False
        planner_cfg.visualize_spheres = False
        planner_cfg.debug_planner = self.cfg.planner_debug
        planner_cfg.retreat_distance = 0.0
        planner_cfg.approach_distance = 0.0
        planner_cfg.time_dilation_factor = 1.0
        planner_cfg.interpolation_dt = 0.06
        planner_cfg.trajopt_tsteps = 60
        planner_cfg.n_repeat = 10
        planner_cfg.enable_graph = True
        env_prim_path = f"/World/envs/env_{env_id}"
        planner_cfg.world_ignore_substrings = list(
            dict.fromkeys(
                [
                    f"{env_prim_path}/Robot",
                    f"{env_prim_path}/target",
                    *(planner_cfg.world_ignore_substrings or []),
                    f"{env_prim_path}/input_grid_cube_",
                ]
            )
        )
        planner = ExtendedCuroboPlanner(
            env=self._env,
            robot=self._robot,
            config=planner_cfg,
            env_id=env_id,
            grasp_center_offset=FRANKA_GRASP_CENTER_OFFSET,
            use_native_goalset=False,
        )
        self._planners[env_id] = planner
        return planner


@configclass
class HybridArmActionCfg(ActionTermCfg):
    """Configuration for :class:`HybridArmAction`."""

    class_type: type[ActionTerm] = HybridArmAction
    diff_ik_cfg: DifferentialInverseKinematicsActionCfg = MISSING
    curobo_task_name: str = MISSING
    planner_debug: bool = False
    contact_window_fraction: float | None = 0.1
    hold_position_tolerance: float = 1.0e-2
    hold_orientation_tolerance: float = 2.0e-2
