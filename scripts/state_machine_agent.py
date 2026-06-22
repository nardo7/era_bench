# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

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
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
import era_bench.tasks  # noqa: F401
from env_cfg import configure_env, configure_curobo_planner
from era_bench.tasks.manager_based.mirror.mirror_state_machine import MirrorStateMachine
from era_bench.tasks.manager_based.mirror.scene_cfg import EraBenchSceneCfg
from era_bench.tasks.manager_based.common.geometry_utils import mirror_points_across_line
from isaaclab.sensors import FrameTransformer
import isaaclab.utils.math as math_utils
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from era_bench.tasks.manager_based.common.utils import line_direction_xy_from_quat
from era_bench.tasks.manager_based.mirror.mirror_state_machine import MirrorStates
from era_bench.curobo_planner_extended import ExtendedCuroboPlanner


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
    env: ManagerBasedRLEnv, planner: CuroboPlanner, target_gripper_pose: torch.Tensor
) -> tuple[VecEnvObs | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict | None, int]:
    planned_trayectory_poses = planner.get_planned_poses()
    obs: VecEnvObs | None = None
    reward: torch.Tensor | None = None
    terminated: torch.Tensor | None = None
    truncated: torch.Tensor | None = None
    info: dict | None = None

    if not planned_trayectory_poses:
        print("[INFO]: No planned trajectory poses available. Skipping execution.")
        return obs, reward, terminated, truncated, info, 0

    executed_steps = 0
    for pose in planned_trayectory_poses:
        # from matrix pose to action
        action = target_eef_pose_to_action({"eef": pose}, {"eef": target_gripper_pose.unsqueeze(-1)})
        obs, reward, terminated, truncated, info = env.step(action)
        executed_steps += 1
        if terminated | truncated:
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


def generate_yaw_candidates(
    position: torch.Tensor, base_quat: torch.Tensor, num_candidates: int = 8
) -> list[torch.Tensor]:
    """Generate fixed wrist-yaw variants for a target position using a chosen base orientation."""
    yaw_angles = torch.linspace(
        0.0,
        2.0 * torch.pi,
        steps=num_candidates + 1,
        device=position.device,
        dtype=position.dtype,
    )[:-1]
    zeros = torch.zeros_like(yaw_angles)
    yaw_quats = math_utils.quat_from_euler_xyz(zeros, zeros, yaw_angles)
    candidate_quats = math_utils.quat_mul(yaw_quats, base_quat.unsqueeze(0).expand(num_candidates, -1))
    return [torch.cat((position, quat), dim=0) for quat in candidate_quats]


def plan_with_fixed_yaw_candidates(
    planner: ExtendedCuroboPlanner, target_position: torch.Tensor, base_quat: torch.Tensor
) -> bool:
    """Try a small fixed set of wrist yaws and keep the first feasible plan."""
    candidate_poses = generate_yaw_candidates(target_position, base_quat, num_candidates=8)
    candidate_poses = torch.stack(candidate_poses, dim=0)
    return planner.plan_goalset(candidate_poses[:, :3], candidate_poses[:, 3:7])
    # for candidate_index, candidate_pose in enumerate(candidate_poses, start=1):
    #     candidate_matrix_pose = matrix_pose_from_pose(candidate_pose)
    #     if planner.update_world_and_plan_motion(candidate_matrix_pose):
    #         print(
    #             "[INFO]: Planner: Selected yaw candidate "
    #             f"{candidate_index}/{len(candidate_poses)} with quaternion {candidate_pose[3:7].tolist()}"
    #         )
    #         return True

    # print(f"[INFO]: Planner: No feasible plan found across {len(candidate_poses)} fixed yaw candidates.")
    # return False


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
    env = cast(ManagerBasedRLEnv, env)
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
    num_objects = len(scene_cfg.object_names)

    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.num_envs, num_objects, 4), device=env.device)
    desired_orientation[:, :, 1] = 1.0
    object_names = scene_cfg.object_names
    object_cfgs = [SceneEntityCfg(name=name) for name in object_names]
    objects: list[RigidObject] = [env.scene[cfg.name] for cfg in object_cfgs]
    objects_pos = torch.stack([obj.data.root_pos_w.clone() for obj in objects], dim=1) - env.scene.env_origins
    objects_quat = torch.stack([obj.data.root_quat_w.clone() for obj in objects], dim=1)
    line_cfg = SceneEntityCfg(name=scene_cfg.line_name)
    line: RigidObject = env.scene[line_cfg.name]
    line_pos = line.data.root_pos_w[:, :2].clone() - env.scene.env_origins[:, :2]
    line_dir_w = line_direction_xy_from_quat(line.data.root_quat_w.clone())

    dest_pos_xy = mirror_points_across_line(objects_pos[:, :, :2], line_pos, line_dir_w)
    dest_poses = torch.cat((dest_pos_xy, objects_pos[:, :, 2:3], desired_orientation), dim=-1)
    mirror_sm = MirrorStateMachine(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.num_envs,
        dest_poses=dest_poses,
        device=env.device,  # pyright: ignore[reportArgumentType]
    )

    # simulate environment
    step = 0
    state_steps = 0
    start = time.time()
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # warmup

            # compute zero actions
            # apply actions

            # -- end-effector frame
            ee_frame_sensor: FrameTransformer = env.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            print(f"[INFO]: MirrorStateMachine: Current TCP position: {tcp_rest_position[0, :3].tolist()}")

            object_names = scene_cfg.object_names
            object_cfgs = [SceneEntityCfg(name=name) for name in object_names]
            objects: list[RigidObject] = [env.scene[cfg.name] for cfg in object_cfgs]
            objects_pos = torch.stack([obj.data.root_pos_w.clone() for obj in objects], dim=1) - env.scene.env_origins
            objects_quat = torch.stack([obj.data.root_quat_w.clone() for obj in objects], dim=1)
            line_cfg = SceneEntityCfg(name=scene_cfg.line_name)
            line: RigidObject = env.scene[line_cfg.name]
            line_pos = line.data.root_pos_w[:, :2].clone() - env.scene.env_origins[:, :2]
            line_dir_w = line_direction_xy_from_quat(line.data.root_quat_w.clone())

            objects_pos = torch.cat((objects_pos, objects_quat), dim=-1)
            print(f"[INFO]: MirrorStateMachine: Current object position: {objects_pos[0, :, :3].tolist()}")

            next_pose, mirror_state = mirror_sm.compute(
                torch.cat((tcp_rest_position, tcp_rest_orientation), dim=-1), objects_pos
            )

            print(f"[INFO]: MirrorStateMachine: Next desired pose: {next_pose[0, :3].tolist()}")
            plan_success = False

            print(f"[INFO]: MirrorStateMachine: Current state: {MirrorStates(mirror_state[0].item())}")
            # this supports only one environment for now
            if mirror_state[0] in [
                MirrorStates.APPROACHING.python_value,
                MirrorStates.MIRRORING.python_value,
                MirrorStates.RETREATING.python_value,
            ]:
                next_pose = next_pose[0]
                if mirror_state[0] == MirrorStates.APPROACHING.python_value:
                    current_goal_idx = int(mirror_sm.current_idx_objects[0].item())
                    grasp_base_quat = desired_orientation[0, current_goal_idx]
                    plan_success = plan_with_fixed_yaw_candidates(planner, next_pose[:3], grasp_base_quat)
                else:
                    if mirror_state[0] == MirrorStates.MIRRORING.python_value:
                        print(
                            f"[INFO]: Planner: Planning for mirroring object {object_names[int(mirror_sm.current_idx_objects[0].item())]}"
                        )
                        next_pose = torch.cat(
                            (next_pose[:3], tcp_rest_orientation[0], next_pose[-1].unsqueeze(-1)), dim=0
                        )
                    matrix_pose = matrix_pose_from_pose(next_pose[:-1])
                    plan_success = planner.update_world_and_plan_motion(
                        matrix_pose,
                        expected_attached_object=(
                            object_names[int(mirror_sm.current_idx_objects[0].item())]
                            if mirror_state[0] == MirrorStates.MIRRORING.python_value
                            else None
                        ),
                    )
                next_pose = next_pose.unsqueeze(0)

            if not plan_success:
                print("[INFO]: Skipping execution for this state-machine step.")
                current_pose = torch.cat(
                    (tcp_rest_position, tcp_rest_orientation, next_pose[:, -1].unsqueeze(-1)), dim=-1
                )

                obs, reward, terminated, truncated, info = env.step(current_pose)
                state_steps = 1
            else:
                obs, reward, terminated, truncated, info, state_steps = execute_plan(env, planner, next_pose[0, -1])

            if terminated is None or truncated is None:
                dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            else:
                dones = terminated | truncated

            mirror_sm.advance(state_steps)

        if dones.any():
            print(f"[INFO]: Resetting environments {torch.where(dones)[0].tolist()} after episode completion.")
            mirror_sm.reset_idx(torch.where(dones)[0].tolist())
            env.scene.reset(torch.where(dones)[0].tolist())

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
