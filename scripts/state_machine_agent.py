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
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
import era_bench.tasks  # noqa: F401
from env_cfg import configure_env, configure_curobo_planner
from era_bench.tasks.manager_based.mirror.mirror_state_machine import MirrorStateMachine
from era_bench.tasks.manager_based.common.geometry_utils import mirror_points_across_line
from isaaclab.sensors import FrameTransformer
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from era_bench.tasks.manager_based.common.utils import line_direction_xy_from_quat


def execute_plan(env: ManagerBasedRLEnv, planner: CuroboPlanner, target_gripper_pose: torch.Tensor):
    planned_trayectory_poses = planner.get_planned_poses()
    if planned_trayectory_poses is None:
        print("[INFO]: No planned trajectory poses available. Skipping execution.")
        return
    for pose in planned_trayectory_poses:
        action = torch.cat([pose, target_gripper_pose.unsqueeze(-1)], dim=-1)
        env.step(action)


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

    # configure curobo planner
    planner = configure_curobo_planner(env, args_cli.task, env_id=0)
    if env.action_space.shape is None:
        raise ValueError("Expected action space to be continuous, but got: {env.action_space}")
    actions = torch.zeros(*env.action_space.shape, device=env.device)
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.num_envs, 4), device=env.device)
    desired_orientation[:, 1] = 1.0

    mirror_sm = MirrorStateMachine(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.num_envs,
        device=env.device,  # pyright: ignore[reportArgumentType]
    )

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    step = 0
    start = time.time()
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            # apply actions
            obs, reward, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            # -- end-effector frame
            ee_frame_sensor: FrameTransformer = env.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

            object_names = env.scene.cfg.object_names
            object_cfgs = [SceneEntityCfg(name=name) for name in object_names]
            objects: list[RigidObject] = [env.scene[cfg.name] for cfg in object_cfgs]
            objects_pos = torch.stack([obj.data.root_pos_w.clone() for obj in objects], dim=1) - env.scene.env_origins
            line_cfg = SceneEntityCfg(name=env.scene.cfg.line_name)
            line: RigidObject = env.scene[line_cfg.name]
            line_pos = line.data.root_pos_w[:, :2].clone() - env.scene.env_origins[:, :2]
            line_dir_w = line_direction_xy_from_quat(line.data.root_quat_w.clone())

            dest_pos_xy = mirror_points_across_line(objects_pos[:, :, :2], line_pos, line_dir_w)
            dest_pose = torch.cat((dest_pos_xy, objects_pos[:, :, 2:3], desired_orientation), dim=-1)

            next_pose = mirror_sm.compute(
                torch.cat((tcp_rest_position, tcp_rest_orientation), dim=-1), objects_pos, dest_pose
            )
            planner.update_world_and_plan_motion(next_pose[..., :-1])

            execute_plan(env, planner, next_pose[..., -1])

        if dones.any():
            print(f"[INFO]: Resetting environments {torch.where(dones)[0].tolist()} after episode completion.")
            mirror_sm.reset_idx(torch.where(dones)[0].tolist())
            env.scene.reset(torch.where(dones)[0].tolist())

        step += 1
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
