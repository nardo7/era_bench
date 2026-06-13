from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from .utils import line_direction_xy_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _mirror_points_across_line(
    points_xy: torch.Tensor, line_pos_xy: torch.Tensor, line_dir_xy: torch.Tensor
) -> torch.Tensor:
    """Mirrors points across a 2D line defined by a point and unit direction."""
    points_to_line = points_xy - line_pos_xy.unsqueeze(1)
    parallel_mag = torch.sum(
        points_to_line * line_dir_xy.unsqueeze(1), dim=-1, keepdim=True
    )
    proj_parallel = parallel_mag * line_dir_xy.unsqueeze(1)
    perpendicular = points_to_line - proj_parallel
    return line_pos_xy.unsqueeze(1) + proj_parallel - perpendicular


def _signed_distance_to_line(
    points_xy: torch.Tensor, line_pos_xy: torch.Tensor, line_dir_xy: torch.Tensor
) -> torch.Tensor:
    """Signed distance surrogate to 2D line; sign indicates side of line."""
    points_to_line = points_xy - line_pos_xy.unsqueeze(1)
    return (
        line_dir_xy[:, 0].unsqueeze(1) * points_to_line[:, :, 1]
        - line_dir_xy[:, 1].unsqueeze(1) * points_to_line[:, :, 0]
    )


def are_objects_mirrored(
    env: ManagerBasedRLEnv,
    object_names: tuple[str, ...] = (
        "cube_1",
        "cube_2",
        "cube_3",
        "cube_4",
        "cube_5",
        "cube_6",
    ),
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
    cubes: list[RigidObject] = [env.scene[name] for name in object_names]
    current_pos = torch.stack([cube.data.root_pos_w for cube in cubes], dim=1)

    line: RigidObject = env.scene[middle_line_name]
    line_pos_xy = line.data.root_pos_w[:, :2]
    line_dir_xy = line_direction_xy_from_quat(line.data.root_quat_w)

    cache_name = "_mirror_reference_positions"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, {})
    reference_cache: dict[tuple[tuple[str, ...], str], torch.Tensor] = getattr(
        env, cache_name
    )

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

    target_xy = _mirror_points_across_line(
        reference_pos[:, :, :2], line_pos_xy, line_dir_xy
    )
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
