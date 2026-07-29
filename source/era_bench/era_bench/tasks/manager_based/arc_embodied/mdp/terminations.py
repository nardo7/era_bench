from __future__ import annotations

import torch
from typing import Any, TYPE_CHECKING, cast

from arc_utils import ARC_COLOR_TO_COLOR
from era_bench.tasks.manager_based.common.geometry_utils import mirror_points_across_line
from era_bench.tasks.manager_based.arc_embodied.scene_cfg import EraBenchSceneCfg
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import FrameTransformer
from ...common.utils import line_direction_xy_from_quat
from .commands import LoadTaskCommand
from .observations import objects_grasped
from isaaclab.envs.mdp import time_out as isaaclab_time_out

if TYPE_CHECKING:
    from era_bench.tasks.manager_based.arc_embodied.era_bench_env import EraBenchEnv
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: EraBenchEnv) -> torch.Tensor:
    """Terminate timed-out episodes and mark their current tasks as failed."""
    timed_out = isaaclab_time_out(env)

    if torch.any(timed_out):
        load_task_command = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
        for env_id in timed_out.nonzero(as_tuple=False).flatten().tolist():
            print(
                f"[INFO]: Environment {env_id} timed out. Marking current task '{load_task_command.current_tasks[env_id]}' as failed."
            )
            load_task_command.set_task_failed(env_id)

    return timed_out


def reserve_cubes_displaced(
    env: EraBenchEnv,
    displacement_threshold: float = 0.03,
) -> torch.Tensor:
    """Fail when an ungrasped reserve cube is pushed away from its reset pose.

    The reset event stores world-frame poses for every physical reserve-cube
    slot in ``reserve_cube_stacks``. Cubes that have been grasped are excluded:
    moving one of those is the intended pick-and-place operation. The grasp
    latch is updated here as well because termination is evaluated before the
    regular observation pass.
    """
    live_positions_w = env.get_live_reserve_cube_stack_poses()[..., :3]
    reset_positions_w = env.reserve_cube_stacks.poses_w[..., :3]

    currently_grasped = objects_grasped(env)
    env.reserve_cube_was_grasped |= currently_grasped
    was_grasped = env.reserve_cube_was_grasped.reshape_as(live_positions_w[..., 0])

    displacement = torch.linalg.vector_norm(live_positions_w - reset_positions_w, dim=-1)
    displaced = torch.any((displacement > displacement_threshold) & ~was_grasped, dim=(1, 2))

    if torch.any(displaced):
        load_task_command = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
        for env_id in displaced.nonzero(as_tuple=False).flatten().tolist():
            print(
                f"[INFO]: Environment {env_id} displaced an ungrasped reserve cube. "
                f"Marking current task '{load_task_command.current_tasks[env_id]}' as failed."
            )
            load_task_command.set_task_failed(env_id)

    return displaced


class no_progress(ManagerTermBase):
    """Terminate episodes whose end-effector pose does not move enough across steps."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        params = cast(dict[str, Any], cfg.params)
        self._patience = int(params.get("patience", 25))
        self._warmup_steps = int(params.get("warmup_steps", 0))
        self._position_delta_threshold = float(
            params.get("min_step_position_delta", params.get("min_improvement", 5.0e-4))
        )
        self._orientation_delta_threshold = float(params.get("orientation_delta_threshold", 0.0))
        self._step_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._stagnant_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._prev_ee_pos = torch.full((self.num_envs, 3), torch.nan, dtype=torch.float32, device=self.device)
        self._prev_ee_quat = torch.full((self.num_envs, 4), torch.nan, dtype=torch.float32, device=self.device)

    def reset(self, env_ids=None) -> None:
        self._step_count[env_ids] = 0
        self._stagnant_steps[env_ids] = 0
        self._prev_ee_pos[env_ids] = torch.nan
        self._prev_ee_quat[env_ids] = torch.nan

    def __call__(
        self,
        env: EraBenchEnv,
        patience: int = 25,
        min_improvement: float = 5.0e-4,
        min_step_position_delta: float = 5.0e-4,
        orientation_delta_threshold: float = 0.0,
        divergence_ratio: float | None = None,
        divergence_abs_delta: float | None = None,
        warmup_steps: int = 0,
        trigger_on_divergence: bool = False,
        goal_change_reset_threshold: float = 0.02,
    ) -> torch.Tensor:
        del (
            patience,
            min_improvement,
            min_step_position_delta,
            orientation_delta_threshold,
            divergence_ratio,
            divergence_abs_delta,
            warmup_steps,
            trigger_on_divergence,
            goal_change_reset_threshold,
        )
        ee_frame_sensor: FrameTransformer = env.scene["ee_frame"]
        ee_pos = ee_frame_sensor.data.target_pos_w[:, 0, :].clone() - env.scene.env_origins[:, 0:3]
        ee_quat = ee_frame_sensor.data.target_quat_w[:, 0, :].clone()

        valid_pose = torch.isfinite(ee_pos).all(dim=1) & torch.isfinite(ee_quat).all(dim=1)
        if not torch.any(valid_pose):
            self.reset()
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        prev_pose_valid = torch.isfinite(self._prev_ee_pos).all(dim=1) & torch.isfinite(self._prev_ee_quat).all(dim=1)
        comparable = valid_pose & prev_pose_valid

        pos_delta = torch.linalg.norm(ee_pos - self._prev_ee_pos, dim=1)
        moved = pos_delta > self._position_delta_threshold
        if self._orientation_delta_threshold > 0.0:
            quat_dot = torch.sum(ee_quat * self._prev_ee_quat, dim=1).abs().clamp(max=1.0)
            rot_delta = 2.0 * torch.acos(quat_dot)
            moved |= rot_delta > self._orientation_delta_threshold

        moved &= comparable
        self._stagnant_steps = torch.where(
            moved,
            torch.zeros_like(self._stagnant_steps),
            torch.where(comparable, self._stagnant_steps + 1, torch.zeros_like(self._stagnant_steps)),
        )

        self._step_count += valid_pose.to(dtype=torch.int64)
        self._prev_ee_pos.copy_(ee_pos)
        self._prev_ee_quat.copy_(ee_quat)

        active = comparable & (self._step_count > self._warmup_steps)
        stalled = active & (self._stagnant_steps >= self._patience)

        if torch.any(stalled):
            load_task_command = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
            for env_id in stalled.nonzero(as_tuple=False).flatten().tolist():
                print(
                    f"[INFO]: Environment {env_id} has not made progress for {self._stagnant_steps[env_id].item()} steps. Marking current task '{load_task_command.current_tasks[env_id]}' as failed."
                )
                load_task_command.set_task_failed(env_id)

        return stalled


def _success_positions_reached(
    env: EraBenchEnv,
    distance_threshold: float,
) -> torch.Tensor:
    """Return whether each active success-position slot has a matching cube."""
    success_positions = env.success_positions
    live_reserve_positions = env.get_live_reserve_cube_stack_poses()[..., :3]
    reserve_positions = live_reserve_positions.flatten(start_dim=1, end_dim=2)
    stack_color_ids = env.reserve_cube_stacks.color_ids
    reached = torch.zeros_like(success_positions.active_targets)

    for arc_color_id, color in ARC_COLOR_TO_COLOR.items():
        color_target_slots = success_positions.color_mask(color)
        if not torch.any(color_target_slots):
            continue

        matching_stacks = stack_color_ids == arc_color_id
        matching_cubes = matching_stacks.unsqueeze(-1).expand_as(live_reserve_positions[..., 0]).flatten(start_dim=1)
        distances = torch.cdist(success_positions.positions_w, reserve_positions, p=2)
        distances = torch.where(matching_cubes.unsqueeze(1), distances, torch.full_like(distances, torch.inf))
        reached |= color_target_slots & (torch.min(distances, dim=2).values <= distance_threshold)

    return reached


def success_position_gap(
    env: EraBenchEnv,
) -> torch.Tensor:
    """Fail when the ordered success mask contains ``True, False, True``."""
    reached = env.success_positions_reached
    active_reached = reached & env.success_positions.active_targets
    reached_before = torch.cumsum(active_reached, dim=1, dtype=torch.int32) > 0
    reached_after = torch.flip(
        torch.cumsum(torch.flip(active_reached, dims=(1,)), dim=1, dtype=torch.int32) > 0,
        dims=(1,),
    )
    gaps = env.success_positions.active_targets & ~reached & reached_before & reached_after
    failed = torch.any(gaps, dim=1)

    if torch.any(failed):
        load_task_command = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
        for env_id in failed.nonzero(as_tuple=False).flatten().tolist():
            gap_position = int(torch.nonzero(gaps[env_id], as_tuple=False).flatten()[0].item())
            print(
                f"[INFO]: Environment {env_id} has a gap at success position {gap_position}. "
                f"Marking current task '{load_task_command.current_tasks[env_id]}' as failed."
            )
            load_task_command.set_task_failed(env_id)

    return failed


def are_objects_in_success_positions(
    env: EraBenchEnv,
    distance_threshold: float = 0.01,
) -> torch.Tensor:
    """Check whether every output target has a nearby reserve cube of its color.

    ``env.success_positions`` stores a fixed number of target slots for every
    environment. Only the first ``counts[env_id]`` slots belong to that
    environment's current ARC output; the remaining slots are ignored. For each
    ARC color, this function selects the active slots of that color, compares
    them with reserve cubes of the same color, and combines the result across
    all colors. An output with no active targets is already complete. Otherwise,
    the TCP must also be more than 5cm away from every target and beyond the
    front edge of the populated output grid before the task is considered
    complete.
    """
    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)

    success_positions = env.success_positions
    reached = _success_positions_reached(env, distance_threshold)
    env.success_positions_reached.copy_(reached)
    live_reserve_positions = env.get_live_reserve_cube_stack_poses()[..., :3]
    stack_color_ids = env.reserve_cube_stacks.color_ids
    success_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # All values are world-frame positions: destination targets, reserve cubes,
    # and the TCP frame are directly comparable for every environment.
    ee_frame_sensor: FrameTransformer = env.scene["ee_frame"]
    tcp_position_w = ee_frame_sensor.data.target_pos_w[..., 0, :].clone()

    for arc_color_id, color in ARC_COLOR_TO_COLOR.items():
        color_target_slots = success_positions.color_mask(color)
        num_targets = color_target_slots.sum(dim=1)
        if not torch.any(num_targets):
            continue

        matching_stacks = stack_color_ids == arc_color_id
        matching_cubes = matching_stacks.unsqueeze(-1).expand_as(live_reserve_positions[..., 0]).flatten(start_dim=1)
        num_reserve = matching_cubes.sum(dim=1)

        non_color_slots_are_ignored = torch.ones_like(reached)
        color_targets_are_reached = torch.where(
            color_target_slots,
            reached,
            non_color_slots_are_ignored,
        )
        color_success = torch.all(color_targets_are_reached, dim=1)

        # Every output target needs a distinct available cube in principle. This
        # count check preserves the existing success semantics while ruling out
        # colors with fewer reserve cubes than requested output targets.
        enough_reserve_cubes = num_reserve >= num_targets
        color_success &= enough_reserve_cubes
        success_mask &= color_success

        tcp_to_targets = torch.cdist(tcp_position_w.unsqueeze(1), success_positions.positions_w, p=2).squeeze(1)
        tcp_to_color_targets = torch.where(
            color_target_slots,
            tcp_to_targets,
            torch.full_like(tcp_to_targets, torch.inf),
        )
        tcp_is_away_from_color_targets = torch.min(tcp_to_color_targets, dim=1).values > 0.05
        success_mask &= tcp_is_away_from_color_targets

    # Keep the TCP on the side of the table-camera/top-left-grid line opposite
    # the output.  The output grid grows toward +x/+y from its top-left target,
    # while the table camera is at positive x.  A 15 cm margin covers the
    # desired 10 cm clearance plus half of the 10.34 cm Panda gripper offset.
    active_target_x = torch.where(
        success_positions.active_targets,
        success_positions.positions_w[..., 0],
        torch.full_like(success_positions.positions_w[..., 0], torch.inf),
    )
    active_target_y = torch.where(
        success_positions.active_targets,
        success_positions.positions_w[..., 1],
        torch.full_like(success_positions.positions_w[..., 1], torch.inf),
    )
    output_grid_top_left_w = torch.stack((active_target_x.min(dim=1).values, active_target_y.min(dim=1).values), dim=1)
    table_camera_w = env.scene.env_origins[:, :2] + torch.tensor(
        scene_cfg.table_cam.offset.pos[:2], device=env.device, dtype=tcp_position_w.dtype
    )
    output_grid_top_left_w = torch.where(
        success_positions.has_targets.unsqueeze(1), output_grid_top_left_w, table_camera_w
    )
    camera_to_grid = output_grid_top_left_w - table_camera_w
    grid_side_normal = torch.stack((camera_to_grid[:, 1], -camera_to_grid[:, 0]), dim=1)
    signed_tcp_distance = torch.sum(grid_side_normal * (tcp_position_w[:, :2] - output_grid_top_left_w), dim=1)
    signed_tcp_distance /= torch.linalg.vector_norm(grid_side_normal, dim=1).clamp_min(torch.finfo(tcp_position_w.dtype).eps)
    tcp_clear_of_table_camera = (~success_positions.has_targets) | (signed_tcp_distance <= -0.15)
    success_mask &= tcp_clear_of_table_camera

    if torch.any(success_mask):
        load_task_command = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
        for env_id in success_mask.nonzero(as_tuple=False).flatten().tolist():
            load_task_command.set_task_completed(env_id)

    return success_mask
