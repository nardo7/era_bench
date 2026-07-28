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
# ERA_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, os.path.join(ERA_BENCH_ROOT, "source", "era_bench"))

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

from arc_wandb_logger import ArcWandbLogger
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from env_cfg import configure_env
from isaaclab.sensors import FrameTransformer
from era_bench.tasks.manager_based.arc_embodied.arc_state_machine import ArcControlMode, ArcStates
from era_bench.tasks.manager_based.arc_embodied.arc_state_machine import ArcStateMachine
from era_bench.tasks.manager_based.arc_embodied.ik_recovery_policy import ArcIKRecoveryPolicy
from era_bench.tasks.manager_based.arc_embodied.ik_watchdog import ArcIKWatchdog
from era_bench.tasks.manager_based.arc_embodied.scene_cfg import EraBenchSceneCfg
from era_bench.tasks.manager_based.arc_embodied.era_bench_env import EraBenchEnv
import era_bench.tasks  # noqa: F401


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
    hybrid_arm_action = env.action_manager.get_term("arm_actions")
    if env.action_space.shape is None:
        raise ValueError("Expected action space to be continuous, but got: {env.action_space}")
    scene_cfg = cast(EraBenchSceneCfg, env.scene.cfg)
    arc_sm = ArcStateMachine(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.num_envs,
        success_positions=env.success_positions,
        device=env.device,  # pyright: ignore[reportArgumentType]
        position_threshold=0.01,
        reserve_cube_stacks=env.reserve_cube_stacks,
        cube_sizes=scene_cfg.cubes_size,
        cubes_per_stack=int(scene_cfg.reserve_cubes_amount_per_stack / scene_cfg.reserve_cubes_amount_per_line),
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
        session_label=args_cli.task,
        watched_states=[ArcStates.RETREATING_PLACING_ABOVE, ArcStates.APPROACHING_PLACING],
    )

    # simulate environment
    step = 0
    wandb_logger = ArcWandbLogger(args_cli, env)
    start = time.time()
    timestamps = torch.full((env.num_envs,), start, dtype=torch.float64, device=env.device)
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

            # The hybrid arm term owns direct cuRobo replay.  While a replay is
            # active (including its repeated settling waypoints), defer only
            # the state transition for that environment.  The state machine
            # itself remains controller-neutral.
            transition_allowed = ~hybrid_arm_action.motion_in_progress
            next_pose, arc_state = arc_sm.compute(
                torch.cat((tcp_rest_position, tcp_rest_orientation), dim=-1),
                transition_allowed=transition_allowed,
            )
            if args_cli.debug:
                states = [ArcStates(int(state.item())).name for state in arc_state]
                print(f"[DEBUG]: ArcStateMachine states={states}")
                print(f"[DEBUG]: TCP rest position={tcp_rest_position}")
                print(f"[DEBUG]: next_pose={next_pose}")

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
            command = next_pose.clone()
            command[recovery_update.active_mask] = recovery_update.command[recovery_update.active_mask]
            action_metadata = arc_sm.compute_action_metadata()
            action_metadata[recovery_update.request_curobo_env_ids, 0] = ArcControlMode.CUROBO.value
            actions = torch.cat((command[:, :7], action_metadata, command[:, 7:8]), dim=1)
            obs, reward, terminated, truncated, info = env.step(actions)

            if terminated is None or truncated is None:
                dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            else:
                dones = terminated | truncated

            arc_sm.advance(1)

        recovery_active_mask = recovery_active_mask.clone()

        done_env_ids = torch.where(dones)[0]
        if dones.any():
            print(f"[INFO]: Resetting environments {done_env_ids.tolist()} after episode completion.")
            place_above_watchdog.reset(done_env_ids)
            place_above_recovery.reset(done_env_ids)
            recovery_active_mask[done_env_ids] = False
            arc_sm.reset_idx(done_env_ids.tolist())
            arc_sm.update_reserve_cube_stacks(env.reserve_cube_stacks, done_env_ids)
            arc_sm.update_success_positions(env.success_positions, done_env_ids)
            end_time = time.time()
            elapsed_times = end_time - timestamps[done_env_ids]
            print(
                f"[INFO]: Episode completed for envs {done_env_ids.tolist()} in {elapsed_times.tolist()} seconds. Resetting..."
            )
            timestamps[done_env_ids] = end_time

        elapsed_s, env_steps_per_s, physics_steps_per_s = wandb_logger.update(step, done_env_ids.tolist())
        print(
            f"[INFO]: Step: {step}, Total time elapsed: {elapsed_s:.2f} seconds, "
            f"Environment steps/s: {env_steps_per_s:.2f}, Physics steps/s: {physics_steps_per_s:.2f}"
        )
        step += 1
        if step >= args_cli.max_steps:
            print(f"[INFO]: Reached max steps: {args_cli.max_steps}. Exiting...")
            break

    # close the simulator
    env.close()
    end = time.time()
    print(f"[INFO]: Total simulation time: {end - start:.2f} seconds")
    wandb_logger.finish()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
