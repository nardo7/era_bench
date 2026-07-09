# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from era_bench.tasks.manager_based.arc_embodied.config import Colors, DEFAULT_COLORS

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ..era_bench_env import EraBenchEnv
    from isaaclab.envs import ManagerBasedEnv


def _get_reserve_cube_color_order(env: EraBenchEnv) -> list[Colors]:
    """Return reserve cube colors in the scene's existing name order without duplicates."""
    return list(set(env.cfg.scene.get_reserve_cube_colors()))


def init_reserve_cubes_positions(env: EraBenchEnv, env_ids: list[int]):
    """Initialize the positions of reserve cubes in the environment.

    This function sets the positions of reserve cubes based on their color and index.
    The positions are determined by the scene configuration and are set in a grid-like
    pattern with specified offsets and separation distances.

    Args:
        env (EraBenchEnv): The environment instance containing the scene and assets.
        env_ids (list[int]): List of environment IDs for which to initialize the reserve cubes.
    """
    scene_cfg = env.cfg.scene
    x_offsets = [0.60]  # , 0.55]
    y_offsets = [-0.1, 0.1]
    base_cube_center_height = scene_cfg.cubes_size[2] / 2
    colors = _get_reserve_cube_color_order(env)
    for color_idx, color in enumerate(DEFAULT_COLORS[:2]):
        for i in range(scene_cfg.reserve_cubes_amount_per_color):
            cube_name = scene_cfg.get_reserve_cube_name(color, i)

            offset_x = x_offsets[int(color_idx / 2)]
            offset_y = y_offsets[color_idx % 2]
            x = offset_x + (i % scene_cfg.reserve_cubes_amount_per_line) * (
                scene_cfg.cubes_size[0] + scene_cfg.separation_distance
            )
            y = offset_y
            z = base_cube_center_height + (i // scene_cfg.reserve_cubes_amount_per_line) * scene_cfg.cubes_size[2]
            cube_asset: RigidObject = env.scene[cube_name]
            cube_asset.write_root_pose_to_sim(
                torch.cat(
                    [
                        torch.tensor([x, y, z], device=env.device),
                        torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device),
                    ],
                    dim=-1,
                ).repeat(len(env_ids), 1),
                env_ids=env_ids,
            )
            cube_asset.write_root_velocity_to_sim(
                torch.zeros((1, 6), device=env.device),
                env_ids=env_ids,
            )


def sample_success_positions(
    env: EraBenchEnv,
    env_ids: list[int],
):
    scene_cfg = env.cfg.scene
    # if len(env.success_positions.keys()) == 0:
    #     env.success_positions =
    dest_poses = (
        torch.tensor(
            [
                [0.0, 0.25, scene_cfg.cubes_size[2] * 0.5, 0.0, 0.0, 0.0, 0.0],
                [-0.335, 0.73, scene_cfg.cubes_size[2] * 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.335, 0.73, scene_cfg.cubes_size[2] * 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.335, 0.25, scene_cfg.cubes_size[2] * 0.5, 0.0, 0.0, 0.0, 0.0],
                [-0.335, 0.25, scene_cfg.cubes_size[2] * 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.55, scene_cfg.cubes_size[2] * 0.5, 0.0, 0.0, 0.0, 0.0],
                [-0.335, 0.55, scene_cfg.cubes_size[2] * 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.335, 0.55, scene_cfg.cubes_size[2] * 0.5, 0.0, 0.0, 0.0, 0.0],
            ],
            device=env.device,
        )
        .unsqueeze(0)
        .expand(env.num_envs, -1, -1)
    )

    possible_colors = set(scene_cfg.get_reserve_cube_colors())
    colors = [scene_cfg.get_reserve_cube_colors()[-1]] * dest_poses.shape[1]

    pairs = list(zip(colors, dest_poses.permute(1, 0, 2)))
    dest_poses_by_color: dict[Colors, torch.Tensor] = {}
    for color in possible_colors:
        color_poses = [pose for c, pose in pairs if c == color]
        if color_poses:
            dest_poses_by_color[color] = torch.stack(color_poses, dim=0).permute(1, 0, 2)
        else:
            dest_poses_by_color[color] = torch.empty((env.num_envs, 0, dest_poses.shape[-1]), device=env.device)
    env.success_positions = {
        color: [cube_pose for cube_pose in dest_poses_by_color[color]] for color in dest_poses_by_color
    }
    print(f"[INFO]: Destination poses by color: {env.success_positions}")


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: list[int],
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, str(joint_pos.device))

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
