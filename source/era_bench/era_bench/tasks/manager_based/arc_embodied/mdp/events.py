# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from arc_utils import ARC_COLOR_TO_COLOR, Colors

from era_bench.tasks.manager_based.arc_embodied.mdp.commands import LoadTaskCommand
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ..era_bench_env import EraBenchEnv
    from isaaclab.envs import ManagerBasedEnv


def _get_reserve_cube_color_order(env: EraBenchEnv) -> list[Colors]:
    """Return reserve cube colors in the scene's existing name order without duplicates."""
    return sorted(
        list(set(env.cfg.scene.get_reserve_cube_colors())),
        key=lambda c: env.cfg.scene.get_reserve_cube_colors().index(c),
    )


def prepare_task_command(env: EraBenchEnv, env_ids: list[int] | torch.Tensor) -> None:
    """Advance the task command before later reset events read its grid features."""
    load_task_command: LoadTaskCommand = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
    load_task_command.prepare_reset(env_ids)


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
    x_offsets = [0.30]  # , 0.55]
    y_offsets = [-0.1, 0.1]
    base_cube_center_height = scene_cfg.cubes_size[2] / 2
    colors = _get_reserve_cube_color_order(env)
    for color_idx, color in enumerate(colors):
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

            position = torch.tensor([x, y, z], device=env.device).unsqueeze(0).repeat(len(env_ids), 1)
            position += env.scene.env_origins[env_ids]

            cube_asset.write_root_pose_to_sim(
                torch.cat(
                    [
                        position,
                        torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(len(env_ids), 1),
                    ],
                    dim=-1,
                ),
                env_ids=env_ids,
            )
            cube_asset.write_root_velocity_to_sim(
                torch.zeros((1, 6), device=env.device),
                env_ids=env_ids,
            )


def _reset_input_grid_cubes(env: EraBenchEnv, env_ids_tensor: torch.Tensor):

    scene_cfg = env.cfg.scene
    # Kinematic reference cubes are hidden below the ground plane before a task
    # activates the slots it needs. This replaces deleting and respawning prims.
    hidden_pose = torch.tensor([0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0], device=env.device).repeat(
        env_ids_tensor.numel(), 1
    )
    hidden_pose[:, :3] += env.scene.env_origins[env_ids_tensor]
    zero_velocity = torch.zeros((env_ids_tensor.numel(), 6), device=env.device)

    for cube_name in scene_cfg.input_grid_cubes:
        cube_asset: RigidObject = env.scene[cube_name]
        cube_asset.write_root_pose_to_sim(hidden_pose, env_ids=env_ids_tensor)  # pyright: ignore[reportArgumentType]
        cube_asset.write_root_velocity_to_sim(
            zero_velocity, env_ids=env_ids_tensor  # pyright: ignore[reportArgumentType]
        )


def _setup_input_grid(env: EraBenchEnv, env_id: int, input_grid_cubes: torch.Tensor) -> None:
    """Clear and repopulate the static input-grid cube pool for reset environments."""
    scene_cfg = env.cfg.scene

    next_slot_by_color: dict[int, int] = {}

    for x, y, color in input_grid_cubes.tolist():
        arc_color = int(color)
        slot = next_slot_by_color.get(arc_color, 0)
        next_slot_by_color[arc_color] = slot + 1
        cube_name = scene_cfg.get_input_grid_cube_name(arc_color, slot)
        cube_asset: RigidObject = env.scene[cube_name]

        root_pose = torch.tensor(
            [x, y, scene_cfg.cubes_size[2] * 0.5, 1.0, 0.0, 0.0, 0.0],
            device=env.device,
        ).unsqueeze(0)
        root_pose[:, :3] += env.scene.env_origins[env_id] + torch.tensor(
            list(env.cfg.scene.input_grid_center_local), device=env.device
        )
        env_id_tensor = torch.tensor([env_id], device=env.device, dtype=torch.long)
        cube_asset.write_root_pose_to_sim(root_pose, env_ids=env_id_tensor)  # pyright: ignore[reportArgumentType]
        cube_asset.write_root_velocity_to_sim(
            torch.zeros((1, 6), device=env.device), env_ids=env_id_tensor  # pyright: ignore[reportArgumentType]
        )


def _setup_success_positions(env: EraBenchEnv, env_id: int, output_grid_cubes: torch.Tensor) -> None:
    """Store the current task's output objects as world-frame success targets."""
    scene_cfg = env.cfg.scene
    target_count = output_grid_cubes.shape[0]
    positions_w = torch.empty((target_count, 3), device=env.device, dtype=output_grid_cubes.dtype)
    positions_w[:, :2] = (
        output_grid_cubes[:, :2]
        + env.scene.env_origins[env_id, :2]
        + torch.tensor(scene_cfg.output_grid_center_local[:2], device=env.device, dtype=output_grid_cubes.dtype)
    )
    positions_w[:, 2] = scene_cfg.cubes_size[2] * 0.5 + env.scene.env_origins[env_id, 2]

    env.success_positions.replace(
        env_id,
        positions_w=positions_w,
        color_ids=output_grid_cubes[:, 2].to(dtype=torch.long),
    )


def setup_task_grid(env: EraBenchEnv, env_ids: list[int]):
    load_task_command: LoadTaskCommand = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
    env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids_tensor.numel() == 0:
        return

    _reset_input_grid_cubes(env, env_ids_tensor)
    for env_id in env_ids:
        input_grid_cubes, output_grid_cubes = load_task_command.get_features(env_id)
        _setup_input_grid(env, env_id, input_grid_cubes)
        # _setup_success_positions(env, env_id, output_grid_cubes)


def sample_success_positions(env: EraBenchEnv, env_ids: list[int] | torch.Tensor) -> None:
    """Load fixed destination positions for state-machine testing.

    This is intentionally separate from task-derived output targets and remains
    available as a reset event when testing the manipulation stack in isolation.
    """
    scene_cfg = env.cfg.scene
    local_positions = torch.tensor(
        [
            [-0.25, 0.25, scene_cfg.cubes_size[2] * 0.5],
            [-0.19, 0.25, scene_cfg.cubes_size[2] * 0.5],
            [-0.25, 0.31, scene_cfg.cubes_size[2] * 0.5],
            [-0.25, 0.37, scene_cfg.cubes_size[2] * 0.5],
            [-0.25, 0.43, scene_cfg.cubes_size[2] * 0.5],
            [-0.19, 0.43, scene_cfg.cubes_size[2] * 0.5],
            [-0.19, 0.37, scene_cfg.cubes_size[2] * 0.5],
            [-0.19, 0.31, scene_cfg.cubes_size[2] * 0.5],
        ],
        device=env.device,
    )
    test_color = scene_cfg.get_reserve_cube_colors()[-1]
    test_color_id = next(arc_color for arc_color, color in ARC_COLOR_TO_COLOR.items() if color is test_color)
    env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    for env_id in env_ids_tensor.tolist():
        positions_w = local_positions + env.scene.env_origins[env_id]
        color_ids = torch.full((local_positions.shape[0],), test_color_id, device=env.device, dtype=torch.long)
        env.success_positions.replace(env_id, positions_w=positions_w, color_ids=color_ids)


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
