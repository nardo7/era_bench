from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from era_bench.tasks.manager_based.common.geometry_utils import mirror_points_across_line
from era_bench.tasks.manager_based.arc_embodied.scene_cfg import EraBenchSceneCfg
from isaaclab.assets import RigidObject
from ...common.utils import line_direction_xy_from_quat

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


def are_objects_in_success_positions(
    env: EraBenchEnv,
    distance_threshold: float = 0.01,
) -> torch.Tensor:
    """Check if reserve cubes are close to color-matched success positions.

    Supports ragged targets per environment (different number of target positions
    in each env) by padding and masking targets in a batched computation.

    For each color, it computes all pairwise distances between reserve cubes and
    target positions, sorts distances ascending, and verifies that the first
    ``num_targets[env]`` distances are below ``distance_threshold`` for each env.
    """
    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)
    del scene_cfg  # Kept for parity with other terminations and typing context.

    success_positions = env.success_positions
    if not success_positions:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def _as_env_major_positions(positions: torch.Tensor) -> torch.Tensor:
        """Normalize positions to shape [num_envs, num_points, dim]."""
        if positions.ndim == 2:
            positions = positions.unsqueeze(1)
        if positions.shape[0] == env.num_envs:
            return positions
        if positions.ndim >= 2 and positions.shape[1] == env.num_envs:
            return positions.transpose(0, 1)
        raise ValueError(f"Unexpected position tensor shape {tuple(positions.shape)} for {env.num_envs} envs.")

    def _targets_to_padded_env_major(
        targets: list[torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded planar targets [E, T_max, 2] and valid-counts [E]."""
        if isinstance(targets, torch.Tensor):
            env_major = _as_env_major_positions(targets)[..., :2]
            counts = torch.full((env.num_envs,), env_major.shape[1], device=env.device, dtype=torch.long)
            return env_major, counts

        if len(targets) != env.num_envs:
            raise ValueError(f"Ragged target list must have length {env.num_envs}, got {len(targets)}.")

        counts_list: list[int] = []

        for t in targets:
            t_dev = t.to(device=env.device)
            if t_dev.ndim == 1:
                t_dev = t_dev.unsqueeze(0)
            elif t_dev.ndim > 2:
                raise ValueError(f"Unexpected ragged target tensor shape {tuple(t_dev.shape)}.")
            counts_list.append(t_dev.shape[0])

        counts = torch.tensor(counts_list, device=env.device, dtype=torch.long)
        max_targets = int(counts.max().item()) if counts.numel() > 0 else 0
        padded = torch.zeros((env.num_envs, max_targets, 2), device=env.device)
        for env_id, t in enumerate(targets):
            t_dev = t.to(device=env.device)
            if t_dev.ndim == 1:
                t_dev = t_dev.unsqueeze(0)
            if t_dev.shape[0] > 0:
                padded[env_id, : t_dev.shape[0]] = t_dev[..., :2]
        return padded, counts

    reserve_cubes_by_color = env.get_reserve_cubes_positions_by_color()
    success_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for color, target_entries in success_positions.items():
        if len(target_entries) == 0:
            continue

        target_positions, num_targets = _targets_to_padded_env_major(target_entries)
        if target_positions.shape[1] == 0:
            continue

        reserve_positions_raw = reserve_cubes_by_color.get(color)
        if reserve_positions_raw is None:
            success_mask &= num_targets == 0
            continue

        reserve_positions = _as_env_major_positions(reserve_positions_raw)[..., :2]
        num_reserve = reserve_positions.shape[1]
        if num_reserve == 0:
            success_mask &= num_targets == 0
            continue

        # Pairwise planar distances per env: [E, num_targets_max, num_reserve]
        distances = torch.cdist(target_positions, reserve_positions, p=2)
        target_valid_mask = torch.arange(target_positions.shape[1], device=env.device).unsqueeze(
            0
        ) < num_targets.unsqueeze(1)
        # For each target, keep only its closest reserve cube: [E, num_targets_max]
        closest_distances = torch.min(distances, dim=2).values

        within_thresh = closest_distances <= distance_threshold
        selected_within_thresh = torch.where(target_valid_mask, within_thresh, torch.ones_like(within_thresh))

        color_success = torch.all(selected_within_thresh, dim=1)

        has_enough_pairs = num_reserve >= num_targets
        color_success &= has_enough_pairs
        success_mask &= color_success

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
