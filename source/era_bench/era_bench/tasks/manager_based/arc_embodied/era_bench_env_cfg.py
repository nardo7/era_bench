# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from era_bench.tasks.manager_based.common.recorders import EpisodeVideoRecorderCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FRAME_MARKER_CFG,
    FrameTransformerCfg,
    OffsetCfg,
    VisualizationMarkersCfg,
)
from typing import Callable, cast
import torch
from dataclasses import MISSING


from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from . import mdp
from .scene_cfg import FRANKA_GRASP_CENTER_OFFSET, EraBenchSceneCfg

##
# Scene definition
##


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_actions: mdp.HybridArmActionCfg = mdp.HybridArmActionCfg(
        asset_name="robot",
        diff_ik_cfg=DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=FRANKA_GRASP_CENTER_OFFSET),
        ),
        curobo_task_name="ARC-Embodied-Era-Bench-v0",
    )
    gripper_action: mdp.TernaryJointPositionActionCfg = mdp.TernaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        release_command_expr={"panda_finger_.*": 0.021},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CameraCfg(ObsGroup):
        """Camera observations recorded to the dataset outside the policy group."""

        table_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        table_cam_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "distance_to_image_plane",
                "normalize": False,
            },
        )
        table_cam_semantic_segmentation = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "semantic_segmentation",
                "normalize": False,
            },
        )
        origin_top_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("origin_top_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        )
        wrist_cam_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "distance_to_image_plane",
                "normalize": False,
            },
        )
        wrist_cam_semantic_segmentation = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "semantic_segmentation",
                "normalize": False,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubTasksCfg(ObsGroup):
        """Observations for sub-task groups."""

        object_grasped = ObsTerm(
            func=mdp.objects_grasped,
        )
        object_placed = ObsTerm(
            func=mdp.objects_placed,
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    camera: CameraCfg = CameraCfg()
    subtasks: SubTasksCfg = SubTasksCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # line_name: str = "line"

    prepare_task_command = EventTerm(func=mdp.prepare_task_command, mode="reset")

    init_franka_arm_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [
                0.0444,
                -0.1894,
                -0.1107,
                -2.5148,
                0.0044,
                2.3775,
                0.6952,
                0.0400,
                0.0400,
            ],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    setup_task_grid = EventTerm(func=mdp.setup_task_grid, mode="reset")
    reset_reserve_cube_stacks = EventTerm(func=mdp.reset_reserve_cube_stacks, mode="reset")
    # sample_objects_destination_positions = EventTerm(func=mdp.sample_success_positions, mode="reset")


@configclass
class CommandsCfg:
    """Commands for the MDP."""

    load_task = mdp.LoadTaskCommandCfg(
        class_type=mdp.LoadTaskCommand,
        retries=2,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    reserve_cubes_displaced = DoneTerm(
        func=mdp.reserve_cubes_displaced,
        params={"displacement_threshold": 0.04},
    )

    no_progress = DoneTerm(
        func=cast(Callable[..., torch.Tensor], mdp.no_progress),
        params={
            "patience": 25,
            "min_step_position_delta": 5.0e-4,
            "warmup_steps": 10,
        },
    )

    success = DoneTerm(func=mdp.are_objects_in_success_positions, params={"distance_threshold": 0.015})

    success_position_gap = DoneTerm(func=mdp.success_position_gap)


##
# Environment configuration
##


@configclass
class EraBenchEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: EraBenchSceneCfg = EraBenchSceneCfg(num_envs=1, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    recorders: mdp.ARCTaskRecorderManagerCfg = mdp.ARCTaskRecorderManagerCfg(
        record_episode_video=EpisodeVideoRecorderCfg(sensor_name="table_cam")
    )

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        # general settings
        self.decimation = 5  # 50Hz
        self.episode_length_s = 320.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render.rendering_mode = "balanced"
        self.sim.render_interval = 10  # 10Hz
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.enable_dl_denoiser = True

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        if self.scene.robot.spawn is None or self.scene.table.spawn is None:
            return

        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        self.scene.plane.semantic_tags = [("class", "ground")]

        marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()  # type: ignore
        marker_cfg.markers["frame"].scale = (
            0.1,
            0.1,
            0.1,
        )  # pyright: ignore[reportAttributeAccessIssue]
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=FRANKA_GRASP_CENTER_OFFSET,
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

        # Set settings for camera rendering
        self.rerender_on_reset = True

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "origin_top_cam", "wrist_cam"]
