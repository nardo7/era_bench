# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the ARC state machine against task-derived success positions."""

"""Launch Isaac Sim Simulator first."""


import sys
import os
import time
from typing import cast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
from parser_utils import parse_arguments

# add argparse arguments
parser = parse_arguments()

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv, VecEnvObs
from env_cfg import configure_env, configure_curobo_planner
from isaaclab.sensors import FrameTransformer
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from era_bench.curobo_planner_extended import ExtendedCuroboPlanner
from era_bench.tasks.manager_based.arc_embodied.arc_state_machine import ArcStates
from era_bench.tasks.manager_based.arc_embodied import mdp
from era_bench.tasks.manager_based.arc_embodied.arc_state_machine import ArcStateMachine
from era_bench.tasks.manager_based.arc_embodied.ik_recovery_policy import ArcIKRecoveryPolicy
from era_bench.tasks.manager_based.arc_embodied.ik_watchdog import ArcIKWatchdog
from era_bench.tasks.manager_based.arc_embodied.scene_cfg import EraBenchSceneCfg
from era_bench.tasks.manager_based.arc_embodied.era_bench_env import EraBenchEnv
from era_bench.tasks.manager_based.common.geometry_utils import rigid_object_world_size_meters
import era_bench.tasks  # noqa: F401
import random


def target_eef_pose_to_action(
    target_eef_pose_dict: dict, gripper_action_dict: dict, noise: float | None = None
) -> torch.Tensor:
    """Convert target pose to action.

    This method transforms a dictionary of target end-effector poses and gripper actions
    into a single action tensor that can be used by the environment.

    The function:
    1. Extracts target position and rotation from the pose dictionary
    2. Extracts gripper action for the end effector
    3. Concatenates position and quaternion rotation into a pose action
    4. Optionally adds noise to the pose action for exploration
    5. Combines pose action with gripper action into a final action tensor

    Args:
        target_eef_pose_dict: Dictionary containing target end-effector pose(s),
            with keys as eef names and values as pose tensors.
        gripper_action_dict: Dictionary containing gripper action(s),
            with keys as eef names and values as action tensors.
        noise: Optional noise magnitude to apply to the pose action for exploration.
            If provided, random noise is generated and added to the pose action.
        env_id: Environment ID for multi-environment setups, defaults to 0.

    Returns:
        torch.Tensor: A single action tensor combining pose and gripper commands.
    """
    # target position and rotation
    (target_eef_pose,) = target_eef_pose_dict.values()
    target_pos, target_rot = math_utils.unmake_pose(target_eef_pose)

    # get gripper action for single eef
    (gripper_action,) = gripper_action_dict.values()

    # add noise to action
    pose_action = torch.cat([target_pos, math_utils.quat_from_matrix(target_rot)], dim=0)
    if noise is not None:
        pose_noise = noise * torch.randn_like(pose_action)
        pose_action += pose_noise

    return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)


def execute_plan(
    env: EraBenchEnv, planner: ExtendedCuroboPlanner, target_gripper_pose: torch.Tensor
) -> tuple[VecEnvObs | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict | None, int]:
    gripper_raw_action = target_gripper_pose
    planned_trayectory_poses = planner.get_planned_poses()
    current_plan = planner.current_plan_with_repeated_end_pose
    obs: VecEnvObs | None = None
    reward: torch.Tensor | None = None
    terminated: torch.Tensor | None = None
    truncated: torch.Tensor | None = None
    info: dict | None = None

    if current_plan is None or current_plan.position is None or len(current_plan.position) == 0:
        print("[INFO]: No planned trajectory poses available. Skipping execution.")
        return obs, reward, terminated, truncated, info, 0

    grasp_center_poses = [planner.panda_hand_pose_to_grasp_center_pose(pose) for pose in planned_trayectory_poses]

    dest_pose = grasp_center_poses[-1]
    last_action = target_eef_pose_to_action({"eef": dest_pose}, {"eef": target_gripper_pose.unsqueeze(-1)})
    print(f"[INFO]: Trayectory goal planned: {last_action[:3].tolist()} with quaternion {last_action[3:7].tolist()}")
    executed_steps = 0
    plan_joint_positions = cast(torch.Tensor, current_plan.position)
    print(
        "[INFO]: Executing plan with "
        f"{plan_joint_positions.shape[0]} waypoints "
        f"(including planner-configured settling repeats)."
    )
    if current_plan.joint_names is None:
        raise ValueError("Expected current CuRobo plan to include joint names.")
    plan_joint_names = list(current_plan.joint_names)
    for joint_waypoint in plan_joint_positions:
        obs, reward, terminated, truncated, info = env.step_joint_targets(
            joint_waypoint.unsqueeze(0),
            joint_names=plan_joint_names,
            gripper_raw_action=gripper_raw_action,
        )
        executed_steps += 1
        if terminated is not None and truncated is not None and (terminated | truncated).any():
            return obs, reward, terminated, truncated, info, executed_steps

    print("[INFO]: Completed execution of planned trajectory in {} steps.".format(executed_steps))
    return obs, reward, terminated, truncated, info, executed_steps


def _pose_from_xy_quat(xy: torch.Tensor, z: float, quat: torch.Tensor) -> torch.Tensor:
    """Build a 4×4 pose given xy (Tensor[2]), z, and quaternion."""
    device = xy.device
    dtype = xy.dtype
    pos = torch.cat([xy, torch.tensor([z], dtype=dtype, device=device)])
    rot = math_utils.matrix_from_quat(quat.to(device).unsqueeze(0))[0]
    return math_utils.make_pose(pos, rot)


def matrix_pose_from_pose(pose: torch.Tensor) -> torch.Tensor:
    """Convert a 7D pose (x, y, z, qw, qx, qy, qz) to a 4×4 matrix pose."""
    xy = pose[..., :2]
    z = pose[..., 2:3]
    quat = pose[..., 3:]
    return _pose_from_xy_quat(xy, z.item(), quat)


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = (
        parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        if not args_cli.create_dataset
        else configure_env(args_cli)
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env = cast(EraBenchEnv, env)
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # configure curobo planner
    planner = configure_curobo_planner(env, args_cli.task, env_id=0)
    if env.action_space.shape is None:
        raise ValueError("Expected action space to be continuous, but got: {env.action_space}")
    scene_cfg = cast(EraBenchSceneCfg, env.scene.cfg)

    if not bool(torch.any(env.success_positions.has_targets)):
        raise RuntimeError("The current ARC task has no output objects to use as success targets.")

    reserve_cube_stacks = env.get_reserve_cubes_poses_by_color()
    arc_sm = ArcStateMachine(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.num_envs,
        success_positions=env.success_positions,
        device=env.device,  # pyright: ignore[reportArgumentType]
        position_threshold=0.01,
        reserve_cubes_poses=reserve_cube_stacks,
        cube_sizes=scene_cfg.cubes_size,
        cubes_per_stack=int(scene_cfg.reserve_cubes_amount_per_color / scene_cfg.reserve_cubes_amount_per_line),
    )
    place_above_watchdog = ArcIKWatchdog(
        num_envs=env.num_envs,
        device=env.device,
        patience=10,
        min_improvement=1e-3,
        divergence_ratio=1.25,
        warmup_steps=3,
        watched_states=[ArcStates.RETREATING_PLACING_ABOVE, ArcStates.APPROACHING_PLACING],
    )
    place_above_recovery = ArcIKRecoveryPolicy(
        num_envs=env.num_envs,
        device=env.device,
        curobo_planner=planner,
        session_label=args_cli.task,
        watched_states=[ArcStates.RETREATING_PLACING_ABOVE, ArcStates.APPROACHING_PLACING],
    )

    # simulate environment
    step = 0
    state_steps = 0
    start = time.time()
    print(f"[INFO]: Arc IK recovery log file: {place_above_recovery.log_path}")
    recovery_active_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # warmup

            # compute zero actions
            # apply actions

            # -- end-effector frame
            ee_frame_sensor: FrameTransformer = env.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone()
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            print(f"[DEBUG]: ArcStateMachine: Current TCP position: {tcp_rest_position[0, :3].tolist()}")

            next_pose, arc_state = arc_sm.compute(torch.cat((tcp_rest_position, tcp_rest_orientation), dim=-1))
            if arc_state[0] in [
                ArcStates.APPROACHING_PLACING_ABOVE.python_value,
                ArcStates.APPROACHING_PLACING.python_value,
                ArcStates.PLACING.python_value,
            ]:
                color = arc_sm.get_current_goal_color(0)
                reserve_cube_idx = arc_sm.get_current_stack_index(0)
                cube_name = scene_cfg.get_reserve_cube_name(color, reserve_cube_idx)
                planner.set_expected_attached_object(cube_name)
            else:
                planner.set_expected_attached_object(None)
            print(f"[DEBUG]: ArcStateMachine: Next desired pose: {next_pose[0, :3].tolist()}")
            plan_success = False

            print(f"[DEBUG]: ArcStateMachine: Current state: {ArcStates(arc_state[0].item())}")
            if recovery_active_mask.any():
                place_above_watchdog.reset(torch.nonzero(recovery_active_mask, as_tuple=False).flatten())
            watchdog_update = place_above_watchdog.update(
                arc_state,
                tcp_rest_position[:, :3],
                next_pose,
            )
            for warning_message in watchdog_update.warning_messages:
                print(warning_message)
            recovery_update = place_above_recovery.update(
                arc_state,
                tcp_rest_position[:, :3],
                tcp_rest_orientation,
                next_pose,
                watchdog_update,
                step=step,
                elapsed_s=time.time() - start,
            )
            recovery_active_mask = recovery_update.active_mask.clone()
            # this supports only one environment for now
            if (
                recovery_update.execute_curobo_env_ids.numel() > 0
                and 0 in recovery_update.execute_curobo_env_ids.tolist()
            ):
                plan_success = True
            if arc_state[0] in [
                ArcStates.APPROACHING_ABOVE.python_value,
                ArcStates.APPROACHING_PLACING_ABOVE.python_value,
                ArcStates.APPROACHING.python_value,
            ]:  # , ArcStates.APPROACHING.python_value]:
                plan_success = planner.plan_grasp_center_goal_with_fixed_yaw_candidates(
                    next_pose[0, :7], contact_window_fraction=0.1
                )

            if not plan_success:
                print("[WARNING]: Skipping execution for this state-machine step with CUROBO. trying with plain IK.")
                obs, reward, terminated, truncated, info = env.step(recovery_update.command)
                state_steps = 1
            else:
                obs, reward, terminated, truncated, info, state_steps = execute_plan(
                    env, planner, recovery_update.command[0, -1]
                )

            if terminated is None or truncated is None:
                dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            else:
                dones = terminated | truncated

            arc_sm.advance(state_steps)

        recovery_active_mask = recovery_active_mask.clone()

        if dones.any():
            print(f"[INFO]: Resetting environments {torch.where(dones)[0].tolist()} after episode completion.")
            done_env_ids = torch.where(dones)[0]
            place_above_watchdog.reset(done_env_ids)
            place_above_recovery.reset(done_env_ids)
            recovery_active_mask[done_env_ids] = False
            arc_sm.reset_idx(torch.where(dones)[0].tolist())
            arc_sm.update_success_positions(env.success_positions, done_env_ids)

        print(
            f"[INFO]: Step: {step}, State steps executed: {state_steps}, Total time elapsed: {time.time() - start:.2f} seconds"
        )
        step += state_steps
        if step >= args_cli.max_steps:
            print(f"[INFO]: Reached max steps: {args_cli.max_steps}. Exiting...")
            break

    # close the simulator
    env.close()
    end = time.time()
    print(f"[INFO]: Total simulation time: {end - start:.2f} seconds")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
