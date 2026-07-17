from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.recorders import InitialStateRecorderCfg, PostStepStatesRecorderCfg
from .commands import LoadTaskCommand
from ...common.recorders import RecorderManagerCfg
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..era_bench_env import EraBenchEnv


class ARCStatesRecorderWrapper(RecorderTerm):
    """
    Post-step states recorder for the ARC task. It records the state
    of the cubes that are really on the scene (on the table) and filters out
    the ones below the table.

    """

    def __init__(self, cfg: "ARCStatesRecorderWrapperCfg", env):
        super().__init__(cfg, env)
        self._state_recorder_term: RecorderTerm = cfg.state_recorder_term_cfg.class_type(
            cfg.state_recorder_term_cfg, env
        )

    def _filter_rigid_objects_below_table(
        self, value: dict[str, dict[str, dict[str, torch.Tensor]]]
    ) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        rigid_objects = value["rigid_object"]
        poses = rigid_objects["root_pose"]

        filtered_rigid_objects = {name: obj for name, obj in rigid_objects.items() if obj["root_pose"][2] >= 0.0}
        value["rigid_object"] = filtered_rigid_objects
        return value

    def record_post_step(self):
        key, value = self._state_recorder_term.record_post_step()
        # Filter out cubes that are below the table (z < 0.0)
        if value is not None:
            value = self._filter_rigid_objects_below_table(value)
        return key, value

    def record_post_reset(self, env_ids):
        key, value = self._state_recorder_term.record_post_reset(env_ids)
        if value is not None:
            value = self._filter_rigid_objects_below_table(value)
        return key, value


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
        current_task_ids = (
            [self._task_command.current_tasks[env_id] for env_id in env_ids]
        )
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
        current_task_pairs = (
            [self._task_command.current_task_pair[env_id] for env_id in env_ids]
        )
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


class ARCHighLevelActionsRecorderTerm(RecorderTerm):

    _object_grasped: dict[int, list[torch.Tensor]]
    _object_placed: dict[int, list[torch.Tensor]]

    def __init__(self, cfg: "ARCHighLevelActionsRecorderTermCfg", env: EraBenchEnv):
        super().__init__(cfg, env)
        self._env = env
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
        return "high_level_actions/action_timeline", action_timeline

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
class ARCStatesRecorderWrapperCfg(RecorderTermCfg):
    """Configuration for the ARC post-step states recorder term wrapper."""

    class_type: type[RecorderTerm] = ARCStatesRecorderWrapper

    state_recorder_term_cfg: RecorderTermCfg = MISSING  # pyright: ignore[reportAssignmentType]


@configclass
class ARCTaskRecorderTermCfg(RecorderTermCfg):
    """Configuration for the ARC task recorder term wrapper."""

    class_type: type[RecorderTerm] = ARCTaskRecorderTerm


@configclass
class ARCTaskPairRecorderTermCfg(RecorderTermCfg):
    """Configuration for the ARC task pair recorder term wrapper."""

    class_type: type[RecorderTerm] = ARCTaskPairRecorderTerm


@configclass
class ARCHighLevelActionsRecorderTermCfg(RecorderTermCfg):
    """Configuration for the ARC high-level action timeline recorder."""

    class_type: type[RecorderTerm] = ARCHighLevelActionsRecorderTerm


@configclass
class ARCTaskRecorderManagerCfg(RecorderManagerCfg):
    """Configuration for the ARC task recorder manager."""

    task_recorder_term_cfg = ARCTaskRecorderTermCfg()
    task_pair_recorder_term_cfg = ARCTaskPairRecorderTermCfg()
    high_level_actions_recorder_term_cfg = ARCHighLevelActionsRecorderTermCfg()
    # record_post_step_states = ARCStatesRecorderWrapperCfg(state_recorder_term_cfg=PostStepStatesRecorderCfg())
    # record_initial_state = ARCStatesRecorderWrapperCfg(state_recorder_term_cfg=InitialStateRecorderCfg())
