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
from isaaclab.envs.mdp import time_out as isaaclab_time_out

if TYPE_CHECKING:
    from era_bench.tasks.manager_based.arc_embodied.era_bench_env import EraBenchEnv
    from isaaclab.envs import ManagerBasedRLEnv


def _signed_distance_to_line(
    points_xy: torch.Tensor, line_pos_xy: torch.Tensor, line_dir_xy: torch.Tensor
) -> torch.Tensor:
    """Signed distance surrogate to 2D line; sign indicates side of line."""
    points_to_line = points_xy - line_pos_xy.unsqueeze(1)
    return (
        line_dir_xy[:, 0].unsqueeze(1) * points_to_line[:, :, 1]
        - line_dir_xy[:, 1].unsqueeze(1) * points_to_line[:, :, 0]
    )


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
    all colors. The TCP must also be more than 5cm away from every target before
    the task is considered complete.
    """
    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)
    del scene_cfg  # Kept for parity with other terminations and typing context.

    success_positions = env.success_positions
    if not torch.any(success_positions.has_targets):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    reserve_cubes_by_color = env.get_reserve_cubes_positions_by_color()
    success_mask = success_positions.has_targets.clone()

    # All values are world-frame positions: destination targets, reserve cubes,
    # and the TCP frame are directly comparable for every environment.
    ee_frame_sensor: FrameTransformer = env.scene["ee_frame"]
    tcp_position_w = ee_frame_sensor.data.target_pos_w[..., 0, :].clone()

    for color in ARC_COLOR_TO_COLOR.values():
        color_target_slots = success_positions.color_mask(color)
        num_targets = color_target_slots.sum(dim=1)
        if not torch.any(num_targets):
            continue

        reserve_positions_raw = reserve_cubes_by_color.get(color)
        if reserve_positions_raw is None:
            # A task requiring a color not present in the reserve stacks cannot
            # succeed in the environments that contain that color.
            success_mask &= num_targets == 0
            continue

        reserve_positions = reserve_positions_raw[..., :3]
        num_reserve = reserve_positions.shape[1]
        if num_reserve == 0:
            success_mask &= num_targets == 0
            continue

        # Pairwise distances: [environment, target slot, reserve cube]. Slots
        # belonging to other colors are filtered out below.
        distances = torch.cdist(success_positions.positions_w, reserve_positions, p=2)
        closest_distances = torch.min(distances, dim=2).values
        target_is_reached = closest_distances <= distance_threshold
        non_color_slots_are_ignored = torch.ones_like(target_is_reached)
        color_targets_are_reached = torch.where(
            color_target_slots,
            target_is_reached,
            non_color_slots_are_ignored,
        )
        color_success = torch.all(color_targets_are_reached, dim=1)

        # Every output target needs a distinct available cube in principle. This
        # count check preserves the existing success semantics while ruling out
        # colors with fewer reserve cubes than requested output targets.
        enough_reserve_cubes = num_reserve >= num_targets
        color_success &= enough_reserve_cubes
        success_mask &= color_success

        # Do not mark success while the TCP is still at a target location. Set
        # non-color slots to infinity so they cannot affect the minimum.
        tcp_to_targets = torch.cdist(tcp_position_w.unsqueeze(1), success_positions.positions_w, p=2).squeeze(1)
        tcp_to_color_targets = torch.where(
            color_target_slots,
            tcp_to_targets,
            torch.full_like(tcp_to_targets, torch.inf),
        )
        tcp_is_away_from_color_targets = torch.min(tcp_to_color_targets, dim=1).values > 0.05
        success_mask &= tcp_is_away_from_color_targets

    if torch.any(success_mask):
        load_task_command = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
        for env_id in success_mask.nonzero(as_tuple=False).flatten().tolist():
            load_task_command.set_task_completed(env_id)

    return success_mask


def are_objects_mirrored(
    env: ManagerBasedRLEnv,
    object_names: list[str] | None = None,
    middle_line_name: str = "middle_line",
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    # require_side_swap: bool = True,
    # line_side_epsilon: float = 1.0e-4,
) -> torch.Tensor:
    """Success if each object reaches the mirror of its reset pose across the middle line.

    The function caches per-env reference positions at episode reset (when
    ``env.episode_length_buf == 0``) and evaluates success by comparing the current
    object pose against the mirrored reference pose.
    """
    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)
    if object_names is None:
        object_names = scene_cfg.object_names
    cubes: list[RigidObject] = [env.scene[name] for name in object_names]
    current_pos = torch.stack([cube.data.root_pos_w for cube in cubes], dim=1)

    line: RigidObject = env.scene[middle_line_name]
    line_pos_xy = line.data.root_pos_w[:, :2]
    line_dir_xy = line_direction_xy_from_quat(line.data.root_quat_w)

    cache_name = "_mirror_reference_positions"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, {})
    reference_cache: dict[tuple[tuple[str, ...], str], torch.Tensor] = getattr(env, cache_name)

    cache_key = (tuple(object_names), middle_line_name)
    if cache_key not in reference_cache:
        reference_cache[cache_key] = current_pos.clone()

    reference_pos = reference_cache[cache_key]
    if reference_pos.shape != current_pos.shape:
        reference_pos = current_pos.clone()
        reference_cache[cache_key] = reference_pos

    # Refresh references for envs that were just reset.
    reset_envs = env.episode_length_buf == 0
    if torch.any(reset_envs):
        reference_pos[reset_envs] = current_pos[reset_envs]

    target_xy = mirror_points_across_line(reference_pos[:, :, :2], line_pos_xy, line_dir_xy)
    xy_error = torch.linalg.norm(current_pos[:, :, :2] - target_xy, dim=-1)
    z_error = torch.abs(current_pos[:, :, 2] - reference_pos[:, :, 2])

    success_per_object = (xy_error <= xy_threshold) & (z_error <= height_threshold)

    # if require_side_swap:
    #     reference_side = _signed_distance_to_line(reference_pos[:, :, :2], line_pos_xy, line_dir_xy)
    #     current_side = _signed_distance_to_line(current_pos[:, :, :2], line_pos_xy, line_dir_xy)
    #     ref_not_on_line = torch.abs(reference_side) > line_side_epsilon
    #     side_swapped = reference_side * current_side <= 0.0
    #     success_per_object = success_per_object & (~ref_not_on_line | side_swapped)

    return torch.all(success_per_object, dim=1)
