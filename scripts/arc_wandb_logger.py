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
    """Log simulation throughput and task progress without modifying task commands."""

    def __init__(self, args: Namespace, env: EraBenchEnv) -> None:
        self._env = env
        self._start_time = time.time()
        self._start_env_step = env.common_step_counter
        self._run: Any | None = None
        self._load_task_command: LoadTaskCommand | None = None

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
        self._run = wandb.init(**init_kwargs)

        self._load_task_command = cast(LoadTaskCommand, env.command_manager.get_term("load_task"))
        self._previous_tasks = list(self._load_task_command.current_tasks)
        self._previous_pairs = list(self._load_task_command.current_task_pair)
        self._previous_retries = dict(self._load_task_command.retries_per_task)
        self._start_time = time.time()
        self._start_env_step = env.common_step_counter

    def update(self, step: int, done_env_ids: Sequence[int]) -> tuple[float, float, float]:
        """Log one simulation step and return elapsed, environment-rate, and physics-rate values."""
        elapsed_s = time.time() - self._start_time
        env_steps = self._env.common_step_counter - self._start_env_step
        env_steps_per_s = env_steps / elapsed_s if elapsed_s else 0.0
        physics_steps_per_s = env_steps_per_s * self._env.cfg.decimation

        if self._run is None:
            return elapsed_s, env_steps_per_s, physics_steps_per_s

        metrics: dict[str, float | str] = {
            "simulation/elapsed_s": elapsed_s,
            "throughput/environment_steps_per_s": env_steps_per_s,
            "throughput/physics_steps_per_s": physics_steps_per_s,
            "throughput/parallel_env_steps_per_s": env_steps_per_s * self._env.num_envs,
            "throughput/realtime_factor": physics_steps_per_s * self._env.physics_dt,
        }
        if done_env_ids:
            metrics.update(self._task_progress(done_env_ids))
        self._run.log(metrics, step=step)
        return elapsed_s, env_steps_per_s, physics_steps_per_s

    def _task_progress(self, done_env_ids: Sequence[int]) -> dict[str, str]:
        command = self._load_task_command
        if command is None:
            return {}

        current_tasks = list(command.current_tasks)
        current_pairs = list(command.current_task_pair)
        current_retries = dict(command.retries_per_task)
        current_statuses = dict(command.tasks_status)
        events = []
        for env_id in done_env_ids:
            old_task = self._previous_tasks[env_id]
            new_task = current_tasks[env_id]
            old_pair = self._previous_pairs[env_id]
            new_pair = current_pairs[env_id]
            old_retry = self._previous_retries[old_task]
            new_retry = current_retries[new_task]
            if old_task != new_task:
                event = f"task {old_task} {current_statuses[old_task].value}"
            elif old_pair != new_pair:
                event = f"task {old_task}, pair {old_pair} completed"
            elif new_retry > old_retry:
                event = f"task {new_task}, pair {new_pair} retried"
            else:
                event = f"task {old_task}, pair {old_pair} {current_statuses[old_task].value}"
            events.append(
                f"env {env_id}: {event}; now task {new_task}, pair {new_pair}, "
                f"retry {new_retry}/{command.cfg.retries}"
            )

        current = [
            f"env {env_id}: task {task_id}, pair {current_pairs[env_id]}, "
            f"retry {current_retries[task_id]}/{command.cfg.retries}, status {current_statuses[task_id].value}"
            for env_id, task_id in enumerate(current_tasks)
        ]
        finished_tasks = {
            task_id: status.value
            for task_id, status in current_statuses.items()
            if status.value in {"completed", "failed"}
        }
        self._previous_tasks = current_tasks
        self._previous_pairs = current_pairs
        self._previous_retries = current_retries
        return {
            "task_progress/events": " | ".join(events),
            "task_progress/current": " | ".join(current),
            "task_progress/finished_tasks": str(finished_tasks),
        }

    def finish(self) -> None:
        """Finish the active W&B run, if tracking was enabled."""
        if self._run is not None:
            self._run.finish()
