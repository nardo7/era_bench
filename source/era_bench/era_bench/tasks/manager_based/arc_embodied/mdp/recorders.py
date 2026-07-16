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

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[str, torch.Tensor]:
        current_task_ids = (
            [self._task_command.current_tasks[int(env_id)] for env_id in env_ids]
            if env_ids is not None
            else self._task_command.current_tasks
        )
        return "task_info/task_id", torch.tensor(
            [list(task_id.encode("ascii")) for task_id in current_task_ids],
            dtype=torch.uint8,
            device=self._env.device,
        )


class ARCTaskPairRecorderTerm(RecorderTerm):
    def __init__(self, cfg: "ARCTaskPairRecorderTermCfg", env: EraBenchEnv):
        super().__init__(cfg, env)
        self._task_command: LoadTaskCommand = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[str, torch.Tensor]:
        current_task_pairs = (
            [self._task_command.current_task_pair[int(env_id)] for env_id in env_ids]
            if env_ids is not None
            else self._task_command.current_task_pair
        )
        return "task_info/task_pair", torch.tensor(
            [[task_pair] for task_pair in current_task_pairs],
            dtype=torch.int64,
            device=self._env.device,
        )


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
class ARCTaskRecorderManagerCfg(RecorderManagerCfg):
    """Configuration for the ARC task recorder manager."""

    task_recorder_term_cfg = ARCTaskRecorderTermCfg()
    task_pair_recorder_term_cfg = ARCTaskPairRecorderTermCfg()
    # record_post_step_states = ARCStatesRecorderWrapperCfg(state_recorder_term_cfg=PostStepStatesRecorderCfg())
    # record_initial_state = ARCStatesRecorderWrapperCfg(state_recorder_term_cfg=InitialStateRecorderCfg())
