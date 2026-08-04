from __future__ import annotations

from collections.abc import Sequence

import torch


from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from .commands import LoadTaskCommand
from ...common.recorders import RecorderManagerCfg
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..era_bench_env import EraBenchEnv


class ARCTaskRecorderTerm(RecorderTerm):
    def __init__(self, cfg: "ARCTaskRecorderTermCfg", env: EraBenchEnv):
        super().__init__(cfg, env)
        self._task_command: LoadTaskCommand = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
        self._has_completed_initial_reset = False

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | None]:
        env_ids = self._resolve_env_ids(env_ids)
        if not self._has_completed_initial_reset:
            # Isaac Lab invokes pre-reset recording before the initial reset.  There is no
            # completed ARC episode to annotate yet, so do not create task metadata for it.
            return None, None
        current_task_ids = [self._task_command.current_tasks[env_id] for env_id in env_ids]
        return "task_info/task_id", torch.tensor(
            [list(task_id.encode("ascii")) for task_id in current_task_ids],
            dtype=torch.uint8,
            device=self._env.device,
        )

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[None, None]:
        self._has_completed_initial_reset = True
        return None, None

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> list[int]:
        if env_ids is None:
            return list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            return env_ids.tolist()
        return list(env_ids)


class ARCTaskPairRecorderTerm(RecorderTerm):
    def __init__(self, cfg: "ARCTaskPairRecorderTermCfg", env: EraBenchEnv):
        super().__init__(cfg, env)
        self._task_command: LoadTaskCommand = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
        self._has_completed_initial_reset = False

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | None]:
        env_ids = self._resolve_env_ids(env_ids)
        if not self._has_completed_initial_reset:
            return None, None
        current_task_pairs = [self._task_command.current_task_pair[env_id] for env_id in env_ids]
        return "task_info/task_pair", torch.tensor(
            [[task_pair] for task_pair in current_task_pairs],
            dtype=torch.int64,
            device=self._env.device,
        )

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[None, None]:
        self._has_completed_initial_reset = True
        return None, None

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> list[int]:
        if env_ids is None:
            return list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            return env_ids.tolist()
        return list(env_ids)


class CameraObservationsRecorderTerm(RecorderTerm):
    """Record camera observations under ``obs`` without retaining CUDA tensors."""

    def __init__(self, cfg: "CameraObservationsRecorderTermCfg", env: EraBenchEnv):
        super().__init__(cfg, env)
        self._has_completed_initial_reset = False

    def record_pre_step(self) -> tuple[str, dict[str, torch.Tensor]]:
        camera_obs = cast(dict[str, torch.Tensor], self._env.obs_buf["camera"])
        return "obs", {name: value.detach().cpu() for name, value in camera_obs.items()}

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, dict[str, torch.Tensor] | None]:
        if not self._has_completed_initial_reset:
            # The initial reset has no completed episode to which a terminal
            # frame could belong.
            return None, None

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # Termination has already been evaluated, but reset has not started.
        # Force a render now so the final camera sample reflects the terminal
        # arm and cube state rather than the previous render interval.
        self._env.sim.render()
        camera_obs = cast(dict[str, torch.Tensor], self._env.observation_manager.compute_group("camera"))
        return "obs", {name: value[env_ids].detach().cpu() for name, value in camera_obs.items()}

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[None, None]:
        self._has_completed_initial_reset = True
        return None, None


class ARCHighLevelActionsRecorderTerm(RecorderTerm):

    _object_grasped: dict[int, list[torch.Tensor]]
    _object_placed: dict[int, list[torch.Tensor]]

    def __init__(self, cfg: "ARCHighLevelActionsRecorderTermCfg", env: EraBenchEnv):
        super().__init__(cfg, env)
        self._env = env
        self._task_command = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
        self._object_grasped = {i: [] for i in range(env.num_envs)}
        self._object_placed = {i: [] for i in range(env.num_envs)}

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear buffered observations for environments beginning a new episode."""
        for env_id in self._resolve_env_ids(env_ids):
            self._object_grasped[env_id].clear()
            self._object_placed[env_id].clear()

    def record_post_step(self) -> tuple[None, None]:
        obs: dict[str, torch.Tensor] = cast(dict[str, torch.Tensor], self._env.obs_buf["subtasks"])
        object_grasped = obs["object_grasped"]
        object_placed = obs["object_placed"]
        for env_id in range(self._env.num_envs):
            self._object_grasped[env_id].append(object_grasped[env_id].clone())
            self._object_placed[env_id].append(object_placed[env_id].clone())
        return None, None

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | None]:
        env_ids = self._resolve_env_ids(env_ids)
        counts_T = [len(self._object_grasped[env_id]) for env_id in env_ids]
        max_T = max(counts_T)
        if max_T == 0:
            # The initial env.reset() has no completed steps to derive an action timeline from.
            return None, None

        reserve_cubes_amount = len(self._env.cfg.scene.reserve_cubes)
        g = torch.zeros((len(env_ids), max_T, reserve_cubes_amount), dtype=torch.bool, device=self._env.device)
        p = torch.zeros((len(env_ids), max_T, reserve_cubes_amount), dtype=torch.bool, device=self._env.device)
        for i, env_id in enumerate(env_ids):
            if counts_T[i] > 0:
                g[i, : counts_T[i], :] = torch.stack(self._object_grasped[env_id], dim=0)
                p[i, : counts_T[i], :] = torch.stack(self._object_placed[env_id], dim=0)

        grasped_ended = self._get_action_ended(g)
        placed_ended = self._get_action_ended(p)

        any_action_ended = grasped_ended.any(dim=2) | placed_ended.any(dim=2)  # E, T
        any_action_ended_encoded = torch.where(  # E, T
            grasped_ended.any(dim=2),
            torch.tensor(1, device=self._env.device),
            torch.where(
                placed_ended.any(dim=2),
                torch.tensor(2, device=self._env.device),
                torch.tensor(0, device=self._env.device),
            ),
        )
        which_cube_was_grasped = torch.argmax(grasped_ended.to(torch.int32), dim=2)  # E, T
        which_cube_was_placed = torch.argmax(placed_ended.to(torch.int32), dim=2)  # E, T

        action_encoded_which_cube = torch.cat(
            [
                any_action_ended_encoded.unsqueeze(2),
                torch.where(
                    any_action_ended_encoded == 1,
                    which_cube_was_grasped,
                    torch.where(
                        any_action_ended_encoded == 2, which_cube_was_placed, torch.zeros_like(which_cube_was_grasped)
                    ),
                ).unsqueeze(2),
            ],
            dim=2,
        )  # E, T, 2

        steps = torch.arange(max_T, device=self._env.device).view(1, max_T).expand(len(env_ids), max_T)
        sentinel = torch.full_like(steps, max_T)

        end_indices = torch.where(any_action_ended, steps, sentinel)

        next_end = torch.flip(
            torch.cummin(torch.flip(end_indices, dims=[1]), dim=1).values,
            dims=[1],
        )

        # Append the no-action sentinel so positions after the last completed action are valid.
        action_encoded_which_cube = torch.cat(
            [
                action_encoded_which_cube,
                torch.zeros((len(env_ids), 1, 2), dtype=action_encoded_which_cube.dtype, device=self._env.device),
            ],
            dim=1,
        )
        action_timeline = torch.gather(
            action_encoded_which_cube,
            dim=1,
            index=next_end.unsqueeze(-1).expand(-1, -1, 2),
        )
        phase = action_timeline[:, :, 0]
        has_action = phase > 0

        scene_cfg = self._env.cfg.scene
        encoded_timeline = torch.full(
            (len(env_ids), max_T, 5), -1, dtype=action_timeline.dtype, device=self._env.device
        )
        encoded_timeline[:, :, 0] = phase
        encoded_timeline[:, :, 2] = torch.where(has_action, phase - 1, -1)
        for i, env_id in enumerate(env_ids):
            for flat_cube_index in torch.unique(action_timeline[i, has_action[i], 1]).tolist():
                stack_index, cube_index = divmod(flat_cube_index, scene_cfg.reserve_cubes_amount_per_stack)

                color_id = scene_cfg.get_reserve_cube_arc_color(env_id, stack_index, cube_index)
                mask = has_action[i] & (action_timeline[i, :, 1] == flat_cube_index)
                encoded_timeline[i, mask, 1] = color_id
                place_mask = mask & (phase[i] == 2)
                if not place_mask.any():
                    continue

                cube_name = scene_cfg.reserve_cubes[flat_cube_index]
                cube_obj = self._env.scene[cube_name]
                destination = (
                    cube_obj.data.root_pos_w[env_id, :2]
                    - self._env.scene.env_origins[env_id, :2]
                    - torch.tensor(scene_cfg.output_grid_center_local[:2], device=self._env.device)
                )
                goal = self._task_command.current_grid_pairs[env_id].to_grid_coordinates(
                    *destination.tolist(), scene_cfg.cubes_size[0], scene_cfg.separation_distance, grid="output"
                )
                encoded_timeline[i, place_mask, 3:] = torch.tensor(goal, device=self._env.device)
        return "high_level_actions/action_timeline", encoded_timeline

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> list[int]:
        if env_ids is None:
            return list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            return env_ids.tolist()
        return list(env_ids)

    def _get_action_ended(self, values: torch.Tensor) -> torch.Tensor:
        """Get the rising edge of a boolean tensor along the time dimension."""
        previous = torch.cat(
            [torch.zeros_like(values[:, :1]), values[:, :-1]],
            dim=1,
        )
        rising_edge = values & ~previous
        return rising_edge


@configclass
class ARCTaskRecorderTermCfg(RecorderTermCfg):
    """Configuration for the ARC task recorder term wrapper."""

    class_type: type[RecorderTerm] = ARCTaskRecorderTerm


@configclass
class ARCTaskPairRecorderTermCfg(RecorderTermCfg):
    """Configuration for the ARC task pair recorder term wrapper."""

    class_type: type[RecorderTerm] = ARCTaskPairRecorderTerm


@configclass
class CameraObservationsRecorderTermCfg(RecorderTermCfg):
    """Configuration for recording camera observations in CPU episode buffers."""

    class_type: type[RecorderTerm] = CameraObservationsRecorderTerm


@configclass
class ARCHighLevelActionsRecorderTermCfg(RecorderTermCfg):
    """Configuration for the ARC high-level action timeline recorder."""

    class_type: type[RecorderTerm] = ARCHighLevelActionsRecorderTerm


@configclass
class ARCTaskRecorderManagerCfg(RecorderManagerCfg):
    """Configuration for the ARC task recorder manager."""

    task_recorder_term_cfg = ARCTaskRecorderTermCfg()
    task_pair_recorder_term_cfg = ARCTaskPairRecorderTermCfg()
    record_pre_step_camera_observations = CameraObservationsRecorderTermCfg()
    high_level_actions_recorder_term_cfg = ARCHighLevelActionsRecorderTermCfg()
