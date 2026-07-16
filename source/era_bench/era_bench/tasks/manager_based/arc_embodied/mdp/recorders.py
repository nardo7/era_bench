from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from .commands import LoadTaskCommand
from ...common.recorders import RecorderManagerCfg
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..era_bench_env import EraBenchEnv


class ARCTaskRecorderTerm(RecorderTerm):
    def __init__(self, cfg: "ARCTaskRecorderTermWrapperCfg", env: EraBenchEnv):
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
    def __init__(self, cfg: "ARCTaskPairRecorderTermWrapperCfg", env: EraBenchEnv):
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
class ARCTaskRecorderTermWrapperCfg(RecorderTermCfg):
    """Configuration for the ARC task recorder term wrapper."""

    class_type: type[RecorderTerm] = ARCTaskRecorderTerm


@configclass
class ARCTaskPairRecorderTermWrapperCfg(RecorderTermCfg):
    """Configuration for the ARC task pair recorder term wrapper."""

    class_type: type[RecorderTerm] = ARCTaskPairRecorderTerm


class ARCTaskRecorderManagerCfg(RecorderManagerCfg):
    """Configuration for the ARC task recorder manager."""

    task_recorder_term_cfg = ARCTaskRecorderTermWrapperCfg()
    task_pair_recorder_term_cfg = ARCTaskPairRecorderTermWrapperCfg()
