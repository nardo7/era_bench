"""Weights & Biases progress logging for the ARC state-machine runner."""

from __future__ import annotations

import time
from argparse import Namespace
from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

from era_bench.tasks.manager_based.arc_embodied.era_bench_env import EraBenchEnv
from era_bench.tasks.manager_based.arc_embodied.mdp.commands import LoadTaskCommand


class ArcWandbLogger:
    """Render command progress and simulation throughput in W&B."""

    def __init__(self, args: Namespace, env: EraBenchEnv) -> None:
        self._env = env
        self._start_time = time.time()
        self._start_env_step = env.common_step_counter
        self._run: Any | None = None
        self._wandb: Any | None = None
        self._load_task_command: LoadTaskCommand | None = None
        self._pair_start: dict[int, tuple[str, int, int, int, float]] = {}
        self._pair_attempt_rows: list[list[Any]] = []
        self._finished_task_rows: list[list[Any]] = []
        self._finished_task_ids: set[str] = set()
        self._task_env_ids: dict[str, int] = {}
        self._task_pair_stats: dict[str, list[tuple[int, float]]] = {}

        if not args.track:
            return

        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "W&B tracking was requested, but the 'wandb' package is unavailable. "
                "Install the ERA Bench package dependencies in the Apptainer image."
            ) from error

        run_name = args.wandb_name or f"{args.task}-{datetime.now():%Y%m%d-%H%M%S}"
        init_kwargs = {
            "project": args.wandb_project_name,
            "entity": args.wandb_entity,
            "name": run_name,
            "config": {
                "task": args.task,
                "num_envs": env.num_envs,
                "device": str(env.device),
                "max_steps": args.max_steps,
                "sim_dt": env.cfg.sim.dt,
                "decimation": env.cfg.decimation,
            },
        }
        if args.wandb_mode is not None:
            init_kwargs["mode"] = args.wandb_mode
        self._wandb = wandb
        self._run = wandb.init(**init_kwargs)

        self._load_task_command = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
        current_tasks = self._load_task_command.current_tasks
        current_pairs = self._load_task_command.current_task_pair
        current_retries = self._load_task_command.retries_per_task
        self._start_time = time.time()
        self._start_env_step = env.common_step_counter
        self._pair_start = {
            env_id: (
                task_id,
                current_pairs[env_id],
                current_retries[task_id],
                env.common_step_counter,
                self._start_time,
            )
            for env_id, task_id in enumerate(current_tasks)
        }
        self._task_env_ids = {task_id: env_id for env_id, task_id in enumerate(current_tasks)}

    def update(self, step: int, done_env_ids: Sequence[int]) -> tuple[float, float, float]:
        """Log one simulation step and return elapsed, environment-rate, and physics-rate values."""
        elapsed_s = time.time() - self._start_time
        env_steps = self._env.common_step_counter - self._start_env_step
        env_steps_per_s = env_steps / elapsed_s if elapsed_s else 0.0
        physics_steps_per_s = env_steps_per_s * self._env.cfg.decimation

        if self._run is None:
            return elapsed_s, env_steps_per_s, physics_steps_per_s

        metrics: dict[str, Any] = {
            "simulation/elapsed_s": elapsed_s,
            "throughput/environment_steps_per_s": env_steps_per_s,
            "throughput/physics_steps_per_s": physics_steps_per_s,
            "throughput/parallel_env_steps_per_s": env_steps_per_s * self._env.num_envs,
            "throughput/realtime_factor": physics_steps_per_s * self._env.physics_dt,
        }
        if step == 0 or done_env_ids:
            metrics.update(self._task_progress(step, done_env_ids))
        self._run.log(metrics, step=step)
        return elapsed_s, env_steps_per_s, physics_steps_per_s

    def _task_progress(self, step: int, done_env_ids: Sequence[int]) -> dict[str, Any]:
        command = self._load_task_command
        if command is None or self._wandb is None:
            return {}

        current_tasks = list(command.current_tasks)
        current_pairs = list(command.current_task_pair)
        current_retries = dict(command.retries_per_task)
        current_statuses = dict(command.tasks_status)
        now = time.time()
        current_rows = [
            [
                env_id,
                task_id,
                current_pairs[env_id],
                current_retries[task_id],
                current_statuses[task_id].value,
            ]
            for env_id, task_id in enumerate(current_tasks)
        ]

        for env_id in done_env_ids:
            task_id, pair_index, retry_index, start_step, start_time = self._pair_start[env_id]
            new_task = current_tasks[env_id]
            new_pair = current_pairs[env_id]
            new_retry = current_retries[new_task]
            task_status = (
                "completed"
                if self._env.termination_manager.get_term("success")[env_id].item()
                else "failed"
            )
            retry_scheduled = (
                task_status == "failed"
                and task_id == new_task
                and pair_index == new_pair
                and new_retry > retry_index
            )
            pair_steps = self._env.common_step_counter - start_step
            pair_time_s = now - start_time
            self._pair_attempt_rows.append(
                [
                    step,
                    env_id,
                    task_id,
                    pair_index,
                    retry_index,
                    task_status,
                    retry_scheduled,
                    pair_steps,
                    pair_time_s,
                ]
            )

            task_stats = self._task_pair_stats.setdefault(task_id, [])
            task_stats.append((pair_steps, pair_time_s))
            self._task_env_ids[new_task] = env_id
            if current_statuses[new_task].value not in {"completed", "failed"}:
                self._pair_start[env_id] = (
                    new_task,
                    new_pair,
                    new_retry,
                    self._env.common_step_counter,
                    now,
                )

        for task_id, status in current_statuses.items():
            if status.value in {"completed", "failed"} and task_id not in self._finished_task_ids:
                self._finished_task_rows.append(
                    [step, self._task_env_ids.get(task_id), task_id, status.value]
                )
                self._finished_task_ids.add(task_id)

        average_rows = [
            [
                task_id,
                len(values),
                sum(value[0] for value in values) / len(values),
                sum(value[1] for value in values) / len(values),
            ]
            for task_id, values in self._task_pair_stats.items()
        ]
        return {
            "task_progress/current_assignments": self._wandb.Table(
                columns=["environment", "task", "pair", "retry", "status"], data=current_rows
            ),
            "task_progress/pair_attempts": self._wandb.Table(
                columns=[
                    "simulation_step",
                    "environment",
                    "task",
                    "pair",
                    "retry_index",
                    "status",
                    "retry_scheduled",
                    "environment_steps",
                    "wall_time_s",
                ],
                data=self._pair_attempt_rows,
            ),
            "task_progress/task_pair_averages": self._wandb.Table(
                columns=["task", "pair_attempts", "avg_environment_steps", "avg_wall_time_s"],
                data=average_rows,
            ),
            "task_progress/finished_tasks": self._wandb.Table(
                columns=["simulation_step", "environment", "task", "status"],
                data=self._finished_task_rows,
            ),
            "task_progress/completed_count": sum(
                row[3] == "completed" for row in self._finished_task_rows
            ),
            "task_progress/failed_count": sum(row[3] == "failed" for row in self._finished_task_rows),
        }

    def finish(self) -> None:
        """Finish the active W&B run, if tracking was enabled."""
        if self._run is not None:
            self._run.finish()
