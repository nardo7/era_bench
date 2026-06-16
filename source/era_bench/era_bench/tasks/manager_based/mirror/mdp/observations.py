from __future__ import annotations

from era_bench.tasks.manager_based.mirror.scene_cfg import EraBenchSceneCfg
import torch
from typing import TYPE_CHECKING, cast


from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection, AssetBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
import isaaclab.sim as sim_utils
from isaacsim.core.prims import XFormPrim

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(scene_cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(
                scene_cfg.gripper_joint_names
            )  # pyright: ignore[reportAttributeAccessIssue]
            assert len(gripper_joint_ids) == 2, "Observation gripper_pos only support parallel gripper for now"
            finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
            return torch.cat((finger_joint_1, finger_joint_2), dim=1)
        else:
            raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")


def object_obs(
    env: ManagerBasedRLEnv,
    object_names: list[str] | None = None,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to cube_1,
        gripper to cube_2,
        gripper to cube_3,
        cube_1 to cube_2,
        cube_2 to cube_3,
        cube_1 to cube_3,
    """
    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)
    if object_names is None:
        object_names = scene_cfg.object_names
    cubes: list[RigidObject] = [env.scene[name] for name in object_names]

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cubes_pos_w = torch.stack([cube.data.root_pos_w for cube in cubes], dim=1)
    cubes_quat_w = torch.stack([cube.data.root_quat_w for cube in cubes], dim=1)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_cubes = cubes_pos_w - ee_pos_w.unsqueeze(1)

    cubes_pos_w_local = (cubes_pos_w - env.scene.env_origins.unsqueeze(1)).reshape(env.num_envs, -1)
    cubes_quat_w_flat = cubes_quat_w.reshape(env.num_envs, -1)
    gripper_to_cubes_flat = gripper_to_cubes.reshape(env.num_envs, -1)

    return torch.cat(
        (
            cubes_pos_w_local,
            cubes_quat_w_flat,
            gripper_to_cubes_flat,
        ),
        dim=1,
    )


def cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cubes_names: list[str] | None = None,
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)
    if cubes_names is None:
        cubes_names = scene_cfg.object_names

    cubes: list[RigidObject] = [env.scene[name] for name in cubes_names]
    return torch.cat([cube.data.root_pos_w for cube in cubes], dim=1)


def cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cubes_names: list[str] | None = None,
):
    """The orientation of the cubes in the world frame."""
    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)
    if cubes_names is None:
        cubes_names = scene_cfg.object_names

    cubes: list[RigidObject] = [env.scene[name] for name in cubes_names]

    return torch.cat([cube.data.root_quat_w for cube in cubes], dim=1)


def objects_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfgs: list[SceneEntityCfg] | None = None,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)
    if object_cfgs is None:
        object_cfgs = [SceneEntityCfg(name) for name in scene_cfg.object_names]

    objects: list[RigidObject] = [env.scene[obj_cfg.name] for obj_cfg in object_cfgs]

    object_poses = torch.stack([obj.data.root_pos_w for obj in objects], dim=1)
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff: torch.Tensor = torch.linalg.vector_norm(object_poses - end_effector_pos.unsqueeze(1), dim=2)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if scene_cfg.gripper_joint_names is not None:
            gripper_joint_ids, _ = robot.find_joints(scene_cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"

            grasped = torch.logical_and(
                pose_diff < diff_threshold,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(scene_cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                ).unsqueeze(1)
                > scene_cfg.gripper_threshold,
            )
            grasped = torch.logical_and(
                grasped,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]]
                    - torch.tensor(scene_cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                ).unsqueeze(1)
                > scene_cfg.gripper_threshold,
            )

    return grasped


def objects_placed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    objects_cfg: list[SceneEntityCfg] | None = None,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    grasp_diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if an object is placed, released, and was previously grasped in this episode."""

    robot: Articulation = env.scene[robot_cfg.name]
    scene_cfg: EraBenchSceneCfg = cast(EraBenchSceneCfg, env.scene.cfg)
    if objects_cfg is None:
        objects_cfg = [SceneEntityCfg(name) for name in scene_cfg.object_names]

    upper_objects: list[RigidObject] = [env.scene[obj_cfg.name] for obj_cfg in objects_cfg]
    lower_object: AssetBase = env.scene[table_cfg.name]

    upper_pos_w = torch.stack([obj.data.root_pos_w for obj in upper_objects], dim=1)

    # Handle both RigidObject (has .data.root_pos_w) and XFormPrim (use .get_world_poses())
    if hasattr(lower_object, "data"):
        lower_pos_w = lower_object.data.root_pos_w.unsqueeze(1)
    elif isinstance(lower_object, XFormPrim):
        pos, _ = lower_object.get_world_poses()
        lower_pos_w = torch.tensor(pos, device=upper_pos_w.device, dtype=upper_pos_w.dtype).unsqueeze(1)
    else:
        raise RuntimeError(f"Unable to get position from lower_object of type {type(lower_object).__name__}")

    pos_diff = upper_pos_w - lower_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, :, 2:], dim=2)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :, :2], dim=2)

    placed_geom = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        placed = torch.logical_and(suction_cup_is_open, placed_geom)

    else:
        if hasattr(scene_cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(scene_cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"
            placed = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(scene_cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=1e-4,
                    rtol=1e-4,
                ),
                placed_geom,
            )
            placed = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(scene_cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=1e-4,
                    rtol=1e-4,
                ),
                placed,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    # Gate placed on "was grasped in this episode"
    history_attr = "_objects_placed_was_grasped"
    if (not hasattr(env, history_attr)) or getattr(env, history_attr).shape != placed.shape:
        setattr(
            env,
            history_attr,
            torch.zeros_like(placed, dtype=torch.bool, device=env.device),
        )
    was_grasped = getattr(env, history_attr)

    # Reset latch when environments reset
    reset_env_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if reset_env_ids.numel() > 0:
        was_grasped[reset_env_ids] = False

    # Update latch: object is placed only if it was previously grasped
    currently_grasped = objects_grasped(env, robot_cfg, ee_frame_cfg, objects_cfg, diff_threshold=grasp_diff_threshold)
    was_grasped |= currently_grasped

    return torch.logical_and(placed, was_grasped)
