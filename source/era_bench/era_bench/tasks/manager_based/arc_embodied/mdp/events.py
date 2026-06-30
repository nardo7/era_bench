# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import random
import numpy as np
import torch
from typing import TYPE_CHECKING, TypeAlias, TypeGuard, cast

from isaacsim.core.utils.extensions import enable_extension

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.managers import SceneEntityCfg
from ...common.utils import line_direction_xy_from_quat_unnormalized
from ..scene_cfg import EraBenchSceneCfg
from isaacsim.core.prims import XFormPrim
from pxr import Usd, UsdGeom
from ...common.geometry_utils import sample_points_on_half_plane

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


PoseAxisRanges: TypeAlias = dict[str, tuple[float, float]]
PerEnvPoseAxisRanges: TypeAlias = dict[int, PoseAxisRanges]
PoseRangeInput: TypeAlias = PoseAxisRanges | PerEnvPoseAxisRanges


def _xform_prim_world_size_meters(xform_prim: XFormPrim, device: str) -> torch.Tensor:
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
    size_list = []
    for prim in xform_prim.prims:
        aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        size_list.append(list(aligned_range.GetSize()))
    return torch.tensor(size_list, dtype=torch.float32, device=device)


def _is_shared_pose_range(pose_range: PoseRangeInput) -> TypeGuard[PoseAxisRanges]:
    return all(isinstance(key, str) for key in pose_range)


def _normalize_env_ids(env_ids: list[int] | torch.Tensor | None) -> list[int]:
    if env_ids is None:
        return []
    if isinstance(env_ids, torch.Tensor):
        return [int(env_id) for env_id in env_ids.tolist()]
    return env_ids


def _normalize_pose_range(
    pose_range: PoseRangeInput | None,
    env_id_list: list[int],
) -> PerEnvPoseAxisRanges:
    pose_range = pose_range or {}

    if _is_shared_pose_range(pose_range):
        return {env_id: pose_range for env_id in env_id_list}

    # TypeGuard narrows the true branch; cast once here for the complementary case.
    return cast(PerEnvPoseAxisRanges, pose_range)


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


def sample_random_color(base=(0.75, 0.75, 0.75), variation=0.1):
    """
    Generates a randomized color that stays close to the base color while preserving overall brightness.
    The relative balance between the R, G, and B components is maintained by ensuring that
    the sum of random offsets is zero.

    Parameters:
        base (tuple): The base RGB color with each component between 0 and 1.
        variation (float): Maximum deviation to sample for each channel before balancing.

    Returns:
        tuple: A new RGB color with balanced random variation.
    """
    # Generate random offsets for each channel in the range [-variation, variation]
    offsets = [random.uniform(-variation, variation) for _ in range(3)]
    # Compute the average offset
    avg_offset = sum(offsets) / 3
    # Adjust offsets so their sum is zero (maintaining brightness)
    balanced_offsets = [offset - avg_offset for offset in offsets]

    # Apply the balanced offsets to the base color and clamp each channel between 0 and 1
    new_color = tuple(max(0, min(1, base_component + offset)) for base_component, offset in zip(base, balanced_offsets))

    return new_color


def randomize_scene_lighting_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    color_variation: float,
    textures: list[str],
    default_intensity: float = 3000.0,
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    default_texture: str = "",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(default_intensity)

    color_attr = light_prim.GetAttribute("inputs:color")
    color_attr.Set(default_color)

    texture_file_attr = light_prim.GetAttribute("inputs:texture:file")
    texture_file_attr.Set(default_texture)

    if not hasattr(env.cfg, "eval_mode") or not env.cfg.eval_mode:
        return

    if env.cfg.eval_type in ["light_intensity", "all"]:
        # Sample new light intensity
        new_intensity = random.uniform(intensity_range[0], intensity_range[1])
        # Set light intensity to light prim
        intensity_attr.Set(new_intensity)

    if env.cfg.eval_type in ["light_color", "all"]:
        # Sample new light color
        new_color = sample_random_color(base=default_color, variation=color_variation)
        # Set light color to light prim
        color_attr.Set(new_color)

    if env.cfg.eval_type in ["light_texture", "all"]:
        # Sample new light texture (background)
        new_texture = random.sample(textures, 1)[0]
        # Set light texture to light prim
        texture_file_attr.Set(new_texture)


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: PoseAxisRanges | None = None,
    max_sample_tries: int = 5000,
):
    pose_range = pose_range or {}
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: list[int] | torch.Tensor | None,
    asset_cfgs: list[SceneEntityCfg],
    pose_range: PoseRangeInput | None = None,
    min_separation: float = 0.0,
    max_sample_tries: int = 5000,
):
    env_id_list = _normalize_env_ids(env_ids)
    if not env_id_list:
        return

    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)

    pose_range_by_env = _normalize_pose_range(pose_range, env_id_list)

    # Randomize poses in each environment independently
    for cur_env in env_id_list:
        if cur_env not in pose_range_by_env:
            raise KeyError(f"Missing pose_range entry for env_id {cur_env}.")

        yaw_offset = scene_cfg.table.init_state.rot[2]
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range_by_env[cur_env],
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset: AssetBase = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3],
                pose_tensor[:, 4],
                pose_tensor[:, 5] + yaw_offset,
            )
            env_id_tensor = torch.tensor([cur_env], device=env.device)
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1),
                env_ids=env_id_tensor,
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=env_id_tensor,
            )


def randomize_objects_pose_one_side_line_half_plane(
    env: ManagerBasedEnv,
    env_ids: list[int] | torch.Tensor | None,
    line_cfg: SceneEntityCfg,
    asset_cfgs: list[SceneEntityCfg] | None = None,
    height: float = 0.0203,
    min_separation: float = 0.1,
    max_sample_tries: int = 5000,
):
    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)
    if env_ids is None:
        return
    env_ids = _normalize_env_ids(env_ids)

    if asset_cfgs is None:
        asset_cfgs = [SceneEntityCfg(name) for name in scene_cfg.object_names]

    number_of_objects = len(asset_cfgs)
    line: RigidObject = env.scene[line_cfg.name]
    env_ids_tensor = torch.tensor(env_ids, device=env.device)
    env_origins_xy: torch.Tensor = env.scene.env_origins[env_ids_tensor, :2]  # (len(env_ids), 2)

    # Convert world poses to env-local frame because randomize_object_pose adds env origins back.
    line_pos_xy: torch.Tensor = line.data.root_pos_w[env_ids_tensor, :2] - env_origins_xy  # (len(env_ids), 2)
    line_dir_xy: torch.Tensor = line_direction_xy_from_quat_unnormalized(line.data.root_quat_w)[
        env_ids_tensor
    ]  # (len(env_ids), 2)

    # get table limits from env cfg — index to the reset envs only
    table_cfg = SceneEntityCfg("table")
    table: XFormPrim = env.scene[table_cfg.name]
    table_size_xy = _xform_prim_world_size_meters(table, env.device).expand(len(env_ids), -1)[:, :2]
    table_pos = torch.tensor(scene_cfg.table.init_state.pos, device=env.device, dtype=torch.float32).expand(
        len(env_ids), -1
    )  # (len(env_ids), 3)

    sampled_points = sample_points_on_half_plane(
        env_ids=env_ids,
        line_pos_xy=line_pos_xy,
        line_dir_xy=line_dir_xy,
        table_pos=table_pos,
        table_size_xy=table_size_xy,
        number_of_objects=number_of_objects,
        min_separation=min_separation,
    )  # (len(env_ids), number_of_objects, 2)

    for env_idx, cur_env in enumerate(env_ids):
        for obj_idx, asset_cfg in enumerate(asset_cfgs):
            asset: AssetBase = env.scene[asset_cfg.name]
            position = sampled_points[env_idx, obj_idx] + env.scene.env_origins[cur_env, 0:2]
            position = torch.cat([position, torch.tensor([height], device=env.device, dtype=torch.float32)])
            orientation = math_utils.quat_from_euler_xyz(
                torch.tensor(0.0, device=env.device, dtype=torch.float32),
                torch.tensor(0.0, device=env.device, dtype=torch.float32),
                torch.tensor(random.uniform(-math.pi, math.pi), device=env.device, dtype=torch.float32),
            )
            env_id_tensor = torch.tensor([cur_env], device=env.device)
            pose = torch.cat([position.unsqueeze(0), orientation.unsqueeze(0)], dim=-1)
            asset.write_root_pose_to_sim(pose, env_ids=env_id_tensor)

            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=env_id_tensor,
            )


def randomize_rigid_objects_in_focus(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    out_focus_state: torch.Tensor,
    min_separation: float = 0.0,
    pose_range: PoseAxisRanges | None = None,
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    pose_range = pose_range or {}

    # List of rigid objects in focus for each env (dim = [num_envs, num_rigid_objects])
    env.rigid_objects_in_focus = []

    for cur_env in env_ids.tolist():
        # Sample in focus object poses
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        selected_ids = []
        for asset_idx in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[asset_idx]
            asset = env.scene[asset_cfg.name]

            # Randomly select an object to bring into focus
            object_id = random.randint(0, asset.num_objects - 1)
            selected_ids.append(object_id)

            # Create object state tensor
            object_states = torch.stack([out_focus_state] * asset.num_objects).to(device=env.device)
            pose_tensor = torch.tensor([pose_list[asset_idx]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            object_states[object_id, 0:3] = positions
            object_states[object_id, 3:7] = orientations

            asset.write_object_state_to_sim(
                object_state=object_states,
                env_ids=torch.tensor([cur_env], device=env.device),
            )

        env.rigid_objects_in_focus.append(selected_ids)


def randomize_visual_texture_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    textures: list[str],
    default_texture: str = "",
    texture_rotation: tuple[float, float] = (0.0, 0.0),
):
    """Randomize the visual texture of bodies on an asset using Replicator API.

    This function randomizes the visual texture of the bodies of the asset using the Replicator API.
    The function samples random textures from the given texture paths and applies them to the bodies
    of the asset. The textures are projected onto the bodies and rotated by the given angles.

    .. note::
        The function assumes that the asset follows the prim naming convention as:
        "{asset_prim_path}/{body_name}/visuals" where the body name is the name of the body to
        which the texture is applied. This is the default prim ordering when importing assets
        from the asset converters in Isaac Lab.

    .. note::
        When randomizing the texture of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """
    if hasattr(env.cfg, "eval_mode") and (
        not env.cfg.eval_mode or env.cfg.eval_type not in [f"{asset_cfg.name}_texture", "all"]
    ):
        return
        # textures = [default_texture]

    # enable replicator extension if not already enabled
    enable_extension("omni.replicator.core")
    # we import the module here since we may not always need the replicator
    import omni.replicator.core as rep

    # check to make sure replicate_physics is set to False, else raise error
    # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
    #   and the event manager doesn't check in that case.
    if env.cfg.scene.replicate_physics:
        raise RuntimeError(
            "Unable to randomize visual texture material with scene replication enabled."
            " For stable USD-level randomization, please disable scene replication"
            " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
        )

    # convert from radians to degrees
    texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

    # obtain the asset entity
    asset = env.scene[asset_cfg.name]

    # join all bodies in the asset
    body_names = asset_cfg.body_names
    if isinstance(body_names, str):
        body_names_regex = body_names
    elif isinstance(body_names, list):
        body_names_regex = "|".join(body_names)
    else:
        body_names_regex = ".*"

    if not hasattr(asset, "cfg"):
        prims_group = rep.get.prims(path_pattern=f"{asset.prim_paths[0]}/visuals")
    else:
        prims_group = rep.get.prims(path_pattern=f"{asset.cfg.prim_path}/{body_names_regex}/visuals")

    with prims_group:
        rep.randomizer.texture(
            textures=textures,
            project_uvw=True,
            texture_rotate=rep.distribution.uniform(*texture_rotation),
        )
