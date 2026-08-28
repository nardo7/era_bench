# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from arc_utils import ARC_COLOR_TO_COLOR, COLOR_RGB, GridPair

from era_bench.tasks.manager_based.arc_embodied.mdp.commands import LoadTaskCommand
from era_bench.tasks.manager_based.arc_embodied.scene_cfg import GridArea
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils import get_current_stage
from pxr import Gf, UsdGeom

if TYPE_CHECKING:
    from ..era_bench_env import EraBenchEnv
    from isaaclab.envs import ManagerBasedEnv


def prepare_task_command(env: EraBenchEnv, env_ids: list[int] | torch.Tensor) -> None:
    """Advance the task command before later reset events read its grid features."""
    load_task_command: LoadTaskCommand = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
    load_task_command.prepare_reset(env_ids)


def _reserve_palette(
    current_output_grid_cubes: torch.Tensor,
    task_input_colors: torch.Tensor,
    task_output_colors: torch.Tensor,
    stack_count: int,
) -> torch.Tensor:
    """Select stack colors from the active pair, then the rest of its task."""
    current_output_colors = torch.unique(current_output_grid_cubes[:, 2].to(dtype=torch.long), sorted=True)
    if current_output_colors.numel() > stack_count:
        raise ValueError(
            f"Task requires {current_output_colors.numel()} output colors, but only {stack_count} reserve stacks exist."
        )

    palette: list[int] = []
    for candidates in (
        current_output_colors,
        task_output_colors,
        task_input_colors,
    ):
        for arc_color_id in candidates.tolist():
            if arc_color_id not in palette:
                palette.append(arc_color_id)
            if len(palette) == stack_count:
                return _shuffle_reserve_palette(palette, current_output_grid_cubes.device)

    for arc_color_id in sorted(ARC_COLOR_TO_COLOR):
        if arc_color_id not in palette:
            palette.append(arc_color_id)
        if len(palette) == stack_count:
            return _shuffle_reserve_palette(palette, current_output_grid_cubes.device)
    raise RuntimeError("ARC palette is smaller than the configured reserve stack count.")


def _shuffle_reserve_palette(palette: list[int], device: torch.device) -> torch.Tensor:
    """Randomize physical stack assignment after selecting the reserve colors."""
    color_ids = torch.tensor(palette, device=device, dtype=torch.long)
    return color_ids[torch.randperm(color_ids.numel(), device=device)]


def _task_grid_colors(task_grid_pairs: list[GridPair], grid: str, device: torch.device) -> torch.Tensor:
    """Return the non-background ARC colors used in one grid across a task's pairs."""
    return torch.tensor(
        sorted({color for pair in task_grid_pairs for row in getattr(pair, grid) for color in row if color != 0}),
        device=device,
        dtype=torch.long,
    )


def _set_reserve_cube_color(
    env: EraBenchEnv, env_id: int, stack_index: int, cube_index: int, arc_color_id: int
) -> None:
    """Set a reserve cube's rendered color and color-aware semantic class."""
    shader = env.cfg.scene.get_reserve_cube_shader(env_id, stack_index, cube_index)
    shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*COLOR_RGB[ARC_COLOR_TO_COLOR[arc_color_id]]))
    semantics = env.cfg.scene.get_reserve_cube_semantics(env_id, stack_index, cube_index)
    semantics.GetSemanticDataAttr().Set(env.cfg.scene.get_cube_semantic_label(arc_color_id))


def reset_reserve_cube_stacks(env: EraBenchEnv, env_ids: list[int] | torch.Tensor) -> None:
    """Assign task-local colors and fixed reset poses to all physical reserve stacks."""
    scene_cfg = env.cfg.scene
    env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids_tensor.numel() == 0:
        return

    base_cube_center_height = scene_cfg.cubes_size[2] / 2
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    load_task_command: LoadTaskCommand = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
    for env_id in env_ids_tensor.tolist():
        input_grid_cubes, output_grid_cubes = load_task_command.get_features(env_id)
        task_pixel_pairs = load_task_command.get_current_task_grid_pairs(env_id)
        palette = _reserve_palette(
            output_grid_cubes,
            _task_grid_colors(task_pixel_pairs, "input", env.device),
            _task_grid_colors(task_pixel_pairs, "output", env.device),
            scene_cfg.reserve_stacks_amount,
        )
        stack_poses_w = torch.empty(
            (scene_cfg.reserve_stacks_amount, scene_cfg.reserve_cubes_amount_per_stack, 7),
            device=env.device,
        )
        env_id_tensor = torch.tensor([env_id], device=env.device, dtype=torch.long)

        for stack_index, arc_color_id in enumerate(palette.tolist()):
            base_x, base_y = scene_cfg.get_reserve_stack_start_local(stack_index)
            for cube_index in range(scene_cfg.reserve_cubes_amount_per_stack):
                cube_name = scene_cfg.get_reserve_cube_name(stack_index, cube_index)
                x = base_x + (cube_index % scene_cfg.reserve_cubes_amount_per_line) * (
                    scene_cfg.cubes_size[0] + scene_cfg.separation_distance
                )
                z = (
                    base_cube_center_height
                    + (cube_index // scene_cfg.reserve_cubes_amount_per_line) * scene_cfg.cubes_size[2]
                )
                pose_w = torch.cat(
                    (
                        torch.tensor([x, base_y, z], device=env.device) + env.scene.env_origins[env_id],
                        identity_quat,
                    )
                )
                stack_poses_w[stack_index, cube_index] = pose_w
                cube_asset: RigidObject = env.scene[cube_name]
                cube_asset.write_root_pose_to_sim(pose_w.unsqueeze(0), env_ids=env_id_tensor)
                cube_asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=env_id_tensor)
                _set_reserve_cube_color(env, env_id, stack_index, cube_index, arc_color_id)

        env.reserve_cube_stacks.replace(env_id, stack_poses_w, palette)


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
    positions_w[:, 2] = scene_cfg.cubes_size[2] * 0.5
    env.success_positions.replace(
        env_id,
        positions_w=positions_w,
        color_ids=output_grid_cubes[:, 2].to(dtype=torch.long),
    )


def _set_visual_transform(
    env: EraBenchEnv,
    env_id: int,
    asset_name: str,
    position: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> None:
    """Set one visual-only grid prim's local USD transform for a reset environment."""
    prim_path = f"/World/envs/env_{env_id}/{asset_name}"
    prim = get_current_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Grid visual prim does not exist: {prim_path}")
    xformable = UsdGeom.Xformable(prim)

    translate_attr = prim.GetAttribute("xformOp:translate")
    if not translate_attr.IsValid():
        translate_attr = xformable.AddTranslateOp().GetAttr()
    translate_attr.Set(Gf.Vec3d(*position))

    scale_attr = prim.GetAttribute("xformOp:scale")
    if not scale_attr.IsValid():
        scale_attr = xformable.AddScaleOp().GetAttr()
    scale_attr.Set(Gf.Vec3f(*scale))


def _set_visual_visibility(env: EraBenchEnv, env_id: int, asset_name: str, visible: bool) -> None:
    """Set one visual-only grid prim's USD visibility for a reset environment."""
    prim_path = f"/World/envs/env_{env_id}/{asset_name}"
    prim = get_current_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Grid visual prim does not exist: {prim_path}")
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def _set_grid_visual_visibility(env: EraBenchEnv, env_id: int, grid_area: GridArea, visible: bool) -> None:
    """Show or hide every background and line visual for one grid area."""
    scene_cfg = env.cfg.scene
    _set_visual_visibility(env, env_id, scene_cfg.get_grid_background_name(grid_area), visible)
    for line_index in range(scene_cfg.max_grid_size + 1):
        _set_visual_visibility(env, env_id, scene_cfg.get_grid_row_line_name(grid_area, line_index), visible)
        _set_visual_visibility(env, env_id, scene_cfg.get_grid_column_line_name(grid_area, line_index), visible)


def _setup_grid_visuals(env: EraBenchEnv, env_id: int, grid_area: GridArea, grid: list[list[int]]) -> None:
    """Scale and position the visual background and boundary lines for one raw ARC grid."""
    scene_cfg = env.cfg.scene
    rows, columns = len(grid), len(grid[0])
    if rows > scene_cfg.max_grid_size or columns > scene_cfg.max_grid_size:
        raise ValueError(
            f"{grid_area.value} grid shape {rows}x{columns} exceeds the supported {scene_cfg.max_grid_size}x"
            f"{scene_cfg.max_grid_size} visual line pool."
        )

    anchor_x, anchor_y, _ = scene_cfg.get_grid_center_local(grid_area)
    visual_center_x = anchor_x - scene_cfg.cubes_size[0] / 2
    visual_center_y = anchor_y - scene_cfg.cubes_size[1] / 2
    cell_size = scene_cfg.grid_cell_size
    background_name = scene_cfg.get_grid_background_name(grid_area)

    _set_visual_transform(
        env,
        env_id,
        background_name,
        (visual_center_x, visual_center_y, scene_cfg.grid_visual_height / 2),
        (float(rows), float(columns), 1.0),
    )
    _set_visual_visibility(env, env_id, background_name, True)

    for line_index in range(scene_cfg.max_grid_size + 1):
        row_line_visible = line_index <= rows
        row_line_name = scene_cfg.get_grid_row_line_name(grid_area, line_index)
        if row_line_visible:
            _set_visual_transform(
                env,
                env_id,
                row_line_name,
                (
                    visual_center_x + (line_index - rows / 2) * cell_size,
                    visual_center_y,
                    scene_cfg.grid_visual_height * 1.5,
                ),
                (1.0, float(columns), 1.0),
            )
        _set_visual_visibility(env, env_id, row_line_name, row_line_visible)

        column_line_visible = line_index <= columns
        column_line_name = scene_cfg.get_grid_column_line_name(grid_area, line_index)
        if column_line_visible:
            _set_visual_transform(
                env,
                env_id,
                column_line_name,
                (
                    visual_center_x,
                    visual_center_y + (line_index - columns / 2) * cell_size,
                    scene_cfg.grid_visual_height * 1.5,
                ),
                (float(rows), 1.0, 1.0),
            )
        _set_visual_visibility(env, env_id, column_line_name, column_line_visible)


def setup_task_grid(env: EraBenchEnv, env_ids: list[int]):
    load_task_command: LoadTaskCommand = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
    env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids_tensor.numel() == 0:
        return

    _reset_input_grid_cubes(env, env_ids_tensor)
    for env_id in env_ids:
        input_grid_cubes, output_grid_cubes = load_task_command.get_features(env_id)
        grid_pair = load_task_command.current_grid_pairs[env_id]
        _setup_input_grid(env, env_id, input_grid_cubes)
        _setup_success_positions(env, env_id, output_grid_cubes)
        _setup_grid_visuals(env, env_id, GridArea.INPUT, grid_pair.input)
        if load_task_command.is_current_pair_test(env_id):
            _set_grid_visual_visibility(env, env_id, GridArea.OUTPUT, visible=False)
        else:
            _setup_grid_visuals(env, env_id, GridArea.OUTPUT, grid_pair.output)


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
    test_color_id = min(ARC_COLOR_TO_COLOR)
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
