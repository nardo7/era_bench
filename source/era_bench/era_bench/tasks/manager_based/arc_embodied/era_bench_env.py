import os
from typing import Any, cast

from torch import Tensor
import torch

from arc_utils import Colors
from isaaclab.envs import ManagerBasedRLMimicEnv
from .era_bench_env_cfg import EraBenchEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils


class EraBenchEnv(ManagerBasedRLMimicEnv):
    """Environment for the ERA Bench task."""

    cfg: EraBenchEnvCfg
    success_positions: dict[Colors, list[Tensor]] = {}

    def __init__(self, cfg: EraBenchEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._arm_pose_hold_active = False
        self._held_arm_pose_action: Tensor | None = None
        self._held_arm_joint_targets: Tensor | None = None
        # Match the state-machine arrival threshold so the hold is latched
        # before the first gripper-contact step, not one step afterwards.
        self._arm_pose_hold_position_tolerance = 1.0e-2
        self._arm_pose_hold_orientation_tolerance = 2.0e-2
        self._joint_replay_debug_log_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "logs", "joint_replay_debug.log"
            )
        )

    def _action_slice(self, term_name: str) -> slice:
        """Return the slice of the concatenated action belonging to a named term."""
        start = 0
        for name, action_dim in zip(self.action_manager.active_terms, self.action_manager.action_term_dim):
            end = start + action_dim
            if name == term_name:
                return slice(start, end)
            start = end
        raise KeyError(f"Action term '{term_name}' is not active.")

    def _clear_arm_pose_hold(self) -> None:
        self._arm_pose_hold_active = False
        self._held_arm_pose_action = None
        self._held_arm_joint_targets = None

    def _should_hold_arm_pose(self, action: Tensor) -> bool:
        """Return whether the arm target is a stationary grasp-center pose.

        The hold starts only when the commanded absolute pose already matches the
        measured grasp-center pose. Once started, it remains active while that arm
        pose command is unchanged, even if contact forces perturb the TCP.
        """
        arm_action_slice = self._action_slice("arm_actions")
        arm_action = action[:, arm_action_slice].to(self.device)

        if self._arm_pose_hold_active:
            if self._held_arm_pose_action is not None and torch.allclose(
                arm_action, self._held_arm_pose_action, rtol=0.0, atol=1.0e-6
            ):
                print("[INFO]: Holding arm pose.")
                return True
            self._clear_arm_pose_hold()

        arm_action_term = self.action_manager.get_term("arm_actions")
        current_pos, current_quat = arm_action_term._compute_frame_pose()
        position_error = torch.linalg.vector_norm(arm_action[:, :3] - current_pos, dim=1)
        orientation_error = math_utils.quat_error_magnitude(arm_action[:, 3:7], current_quat)
        target_is_current_pose = (position_error <= self._arm_pose_hold_position_tolerance) & (
            orientation_error <= self._arm_pose_hold_orientation_tolerance
        )

        if not bool(torch.all(target_is_current_pose)):
            return False

        robot: Articulation = self.scene["robot"]
        arm_joint_ids, _ = robot.find_joints(["panda_joint.*"], preserve_order=True)
        self._held_arm_pose_action = arm_action.clone()
        self._held_arm_joint_targets = robot.data.joint_pos[:, arm_joint_ids].clone()
        self._arm_pose_hold_active = True
        print("[INFO]: Latched hard arm hold at stationary pose.")
        return True

    def step(self, action: Tensor):
        """Step with DiffIK motion, or a joint-space hold at a stationary target pose.

        This keeps the public action interface unchanged. When an absolute arm pose
        is already reached and remains unchanged, the arm holds its current joint
        targets while the gripper action is still applied normally. A changed arm
        pose immediately resumes the configured DiffIK action term.
        """
        if not self._should_hold_arm_pose(action):
            return super().step(action)

        if self._held_arm_joint_targets is None:
            raise RuntimeError("Missing arm joint targets for stationary-pose hold.")
        gripper_action = action[:, self._action_slice("gripper_action")]
        return self.step_joint_targets(
            self._held_arm_joint_targets,
            gripper_raw_action=gripper_action,
            hard_hold_arm=True,
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_arm_pose_hold_active"):
            self._clear_arm_pose_hold()

    def _log_joint_replay_debug(self, phase: str) -> None:
        robot: Articulation = self.scene["robot"]
        gripper_joint_ids, gripper_joint_names = robot.find_joints(
            self.cfg.scene.gripper_joint_names, preserve_order=True
        )
        arm_joint_ids, arm_joint_names = robot.find_joints(["panda_joint.*"], preserve_order=True)

        with open(self._joint_replay_debug_log_path, "a", encoding="utf-8", buffering=1) as log_file:
            log_file.write(
                f"sim_step={self._sim_step_counter} phase={phase} "
                f"gripper_names={gripper_joint_names} "
                f"gripper_pos={robot.data.joint_pos[0, gripper_joint_ids].tolist()} "
                f"gripper_vel={robot.data.joint_vel[0, gripper_joint_ids].tolist()} "
                f"gripper_pos_target={robot.data.joint_pos_target[0, gripper_joint_ids].tolist()} "
                f"gripper_vel_target={robot.data.joint_vel_target[0, gripper_joint_ids].tolist()} "
                f"arm_names={arm_joint_names} "
                f"arm_pos={robot.data.joint_pos[0, arm_joint_ids].tolist()} "
                f"arm_pos_target={robot.data.joint_pos_target[0, arm_joint_ids].tolist()}\n"
            )

    def _filter_arm_joint_replay_targets(
        self,
        joint_targets: Tensor,
        *,
        joint_names: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[Tensor, list[str] | tuple[str, ...] | None]:
        if joint_names is None:
            return joint_targets, joint_names

        robot: Articulation = self.scene["robot"]
        gripper_joint_relative_ids, _ = robot.find_joints(
            self.cfg.scene.gripper_joint_names,
            joint_subset=list(joint_names),
            preserve_order=False,
        )
        if len(gripper_joint_relative_ids) == 0:
            return joint_targets, joint_names

        arm_joint_indices = [index for index in range(len(joint_names)) if index not in set(gripper_joint_relative_ids)]
        filtered_joint_names = [joint_names[index] for index in arm_joint_indices]
        filtered_joint_targets = joint_targets[:, arm_joint_indices]
        return filtered_joint_targets, filtered_joint_names

    def _apply_joint_targets(
        self,
        arm_joint_targets: Tensor,
        *,
        joint_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        robot: Articulation = self.scene["robot"]
        # self._log_joint_replay_debug("before_apply_joint_targets")
        if joint_names is None:
            arm_joint_ids, _ = robot.find_joints(["panda_joint.*"], preserve_order=True)
        else:
            arm_joint_ids, _ = robot.find_joints(list(joint_names), preserve_order=True)

        arm_joint_targets = arm_joint_targets.to(self.device)
        robot.set_joint_position_target(arm_joint_targets, joint_ids=arm_joint_ids)

        # self._log_joint_replay_debug("after_apply_joint_targets")

    def step_joint_targets(
        self,
        arm_joint_targets: Tensor,
        *,
        joint_names: list[str] | tuple[str, ...] | None = None,
        gripper_raw_action: Tensor | None = None,
        hard_hold_arm: bool = False,
    ):
        """Step the environment while directly applying joint targets.

        This mirrors ManagerBasedRLEnv.step, but replaces the action-manager
        application path with direct arm/gripper joint targets. When
        ``hard_hold_arm`` is true, the arm joint state is pinned before every
        physics substep; this is intentionally stronger than a PD position
        target and prevents contact deflection from accumulating.
        """

        arm_joint_targets, joint_names = self._filter_arm_joint_replay_targets(
            arm_joint_targets,
            joint_names=joint_names,
        )
        arm_joint_targets = arm_joint_targets.to(self.device)

        robot: Articulation = self.scene["robot"]
        if joint_names is None:
            arm_joint_ids, _ = robot.find_joints(["panda_joint.*"], preserve_order=True)
        else:
            arm_joint_ids, _ = robot.find_joints(list(joint_names), preserve_order=True)
        held_arm_joint_velocities = torch.zeros_like(arm_joint_targets)

        gripper_action_term = None
        if gripper_raw_action is not None:
            gripper_action_term = self.action_manager.get_term("gripper_action")
            gripper_raw_action = gripper_raw_action.to(self.device)
            if gripper_raw_action.ndim == 0:
                gripper_raw_action = gripper_raw_action.view(1, 1)
            elif gripper_raw_action.ndim == 1:
                gripper_raw_action = gripper_raw_action.unsqueeze(-1)
            gripper_action_term.process_actions(gripper_raw_action)

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_joint_targets(
                arm_joint_targets,
                joint_names=joint_names,
            )
            if hard_hold_arm:
                # robot.set_joint_velocity_target(held_arm_joint_velocities, joint_ids=arm_joint_ids)
                robot.write_joint_state_to_sim(
                    arm_joint_targets,
                    held_arm_joint_velocities,
                    joint_ids=arm_joint_ids,
                )
            if gripper_action_term is not None:
                gripper_action_term.apply_actions()
            # self._log_joint_replay_debug("before_scene_write")
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            # self._log_joint_replay_debug("after_scene_update")

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute(update_history=True)
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            recorder_reset_env_ids = cast(Any, reset_env_ids)
            self.recorder_manager.record_pre_reset(recorder_reset_env_ids)

            self._reset_idx(cast(Any, reset_env_ids))

            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            self.recorder_manager.record_post_reset(recorder_reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        if len(self.recorder_manager.active_terms) == 0:
            self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def get_reserve_cubes(self) -> list[RigidObject]:
        """Get the reserve cubes."""
        reserve_cube_names = self.cfg.scene.reserve_cubes
        entities = [SceneEntityCfg(name) for name in reserve_cube_names]
        return [self.scene[entity.name] for entity in entities]

    def get_reserve_cubes_by_color(self) -> dict[Colors, list[RigidObject]]:
        """Get the reserve cubes grouped by color."""
        reserve_cubes = self.get_reserve_cubes()
        cubes_by_color: dict[Colors, list[RigidObject]] = {}
        for cube in reserve_cubes:
            color = self.cfg.scene.get_reserve_cube_color(cube.body_names[0])
            if color not in cubes_by_color:
                cubes_by_color[color] = []
            cubes_by_color[color].append(cube)
        return cubes_by_color

    def get_reserve_cubes_positions(self) -> list[Tensor]:
        """Get the positions of the reserve cubes."""
        reserve_cubes = self.get_reserve_cubes()
        return [cube.data.root_link_pos_w for cube in reserve_cubes]

    def _stack_env_major(self, tensors: list[Tensor], value_dim: int) -> Tensor:
        """Stack per-cube tensors into shape (num_envs, num_cubes, value_dim)."""
        if not tensors:
            return torch.empty((self.num_envs, 0, value_dim), device=self.device)

        sample = tensors[0]
        if sample.ndim == 1:
            stacked = torch.stack(tensors, dim=0)
            return stacked.unsqueeze(0)

        if sample.ndim == 2:
            if sample.shape[0] == self.num_envs:
                return torch.stack(tensors, dim=1)
            if sample.shape[1] == value_dim:
                stacked = torch.stack(tensors, dim=0)
                return stacked.unsqueeze(0)

        raise ValueError(
            f"Unexpected reserve cube tensor shape {tuple(sample.shape)}; expected ({self.num_envs}, {value_dim}) or ({value_dim},)."
        )

    def get_reserve_cubes_positions_by_color(self) -> dict[Colors, Tensor]:
        """Get the positions of the reserve cubes grouped by color."""
        reserve_cubes_by_color = self.get_reserve_cubes_by_color()
        positions_by_color: dict[Colors, Tensor] = {}
        for color, cubes in reserve_cubes_by_color.items():
            positions_by_color[color] = self._stack_env_major([cube.data.root_link_pos_w for cube in cubes], 3)
        return positions_by_color

    def get_reserve_cubes_poses_by_color(self) -> dict[Colors, Tensor]:
        """Get the live reserve cube poses grouped by color as (x, y, z, qw, qx, qy, qz)."""
        reserve_cubes_by_color = self.get_reserve_cubes_by_color()
        poses_by_color: dict[Colors, Tensor] = {}
        for color, cubes in reserve_cubes_by_color.items():
            poses_by_color[color] = self._stack_env_major(
                [torch.cat([cube.data.root_link_pos_w, cube.data.root_quat_w], dim=-1) for cube in cubes],
                7,
            )
        return poses_by_color
