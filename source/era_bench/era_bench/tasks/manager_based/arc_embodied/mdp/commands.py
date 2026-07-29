from __future__ import annotations
from collections.abc import Sequence
from dataclasses import MISSING
from enum import Enum
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from arc_utils import get_era_bench_tasks, sort_by_amount_of_output_pixels, GridPair, Pixel
from ...common.dataset_handler import AppendHDF5DatasetFileHandler

MAX_OBJECTS_PER_GRID = 64
MAX_OUTPUT_OBJECTS_PER_GRID = 40
OBJECT_FEATURES = 3  # x, y, color
COMMAND_HEADER_FEATURES = 2  # input object count, output object count

if TYPE_CHECKING:
    from ..era_bench_env import EraBenchEnv


class TaskStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@configclass
class LoadTaskCommandCfg(CommandTermCfg):
    """
    Configuration for the LoadTaskCommand. This command is responsible for loading ARC-AGI tasks for the environment.
    """

    class_type: type[CommandTerm] = MISSING  # pyright: ignore[reportAssignmentType]

    tasks_ids: list[str] | None = None
    task_pairs: list[list[int]] | None = None
    """Ordered raw pair indices to run for each task in ``tasks_ids``."""
    # TODO: Remove configurable selected-pair queues. ERA Bench runs are expected
    # to execute every raw pair of each task in order, so ``_task_pair_queue`` is
    # currently redundant with the task's complete train-and-test pair sequence.
    # Keep this compatibility surface until that broader refactor is worthwhile.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    retries: int = 3


class LoadTaskCommand(CommandTerm):
    """
    Command responsible for loading ARC-AGI tasks for the environment.
    The idea is to load one task pair per environment. A failed pair is retried for
    the configured number of retries; once those retries are exhausted, the task
    remains failed, its remaining pairs are skipped, and the environment advances
    to the next task. A completed pair advances to the next missing pair, and the
    task advances only after all selected pairs have completed.
    It requires that the task status is set externally by the done terms.
    """

    cfg: LoadTaskCommandCfg
    tasks = sort_by_amount_of_output_pixels(get_era_bench_tasks())
    tasks_status: dict[str, TaskStatus] = {}
    successful_task_pairs: dict[str, set[int]] = {}
    retries_per_task: dict[str, int] = {}
    current_tasks: list[str] = []
    current_task_pair: list[int] = []
    """Raw index of the active grid pair within each environment's current task."""
    _current_task_pair_position: list[int] = []
    """Internal zero-based position of the active pair in the configured per-task pair queue."""

    def __init__(self, cfg: LoadTaskCommandCfg, env: EraBenchEnv):
        super().__init__(cfg, env)
        self._tasks_by_id = {task.task_id: task for task in self.tasks}
        self._task_queue = self._resolve_task_queue(cfg.tasks_ids)
        if len(self._task_queue) < env.num_envs:
            raise ValueError(
                f"Number of environments ({env.num_envs}) exceeds the number of selected tasks ({len(self._task_queue)})."
            )

        self.tasks_status = {task.task_id: TaskStatus.NOT_STARTED for task in self.tasks}
        self.successful_task_pairs = {task.task_id: set() for task in self.tasks}
        self.retries_per_task = {task.task_id: 0 for task in self.tasks}
        self._task_pair_queue = self._resolve_task_pair_queue(cfg.task_pairs)
        self.current_tasks = self._task_queue[: self.num_envs]
        self._current_task_pair_position = [0] * self.num_envs
        self.current_task_pair = [self._task_pair_queue[task_id][0] for task_id in self.current_tasks]
        self._cube_size = env.cfg.scene.cubes_size[0]
        self._separation = env.cfg.scene.separation_distance
        command_dim = COMMAND_HEADER_FEATURES + 2 * MAX_OBJECTS_PER_GRID * OBJECT_FEATURES
        self._command = torch.zeros((self.num_envs, command_dim), device=self.device, dtype=torch.float32)
        self._command_state: tuple[tuple[str, int], ...] | None = None
        self._prepared_for_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self._refresh_command()

    def _initialize_from_dataset(self):
        file_handler = self._env.recorder_manager._dataset_file_handler

        if isinstance(file_handler, AppendHDF5DatasetFileHandler):
            tasks_ids_pairs_dict = file_handler.get_executed_task_ids_with_pairs()
            for task_id, pairs in tasks_ids_pairs_dict.items():
                if task_id in self.successful_task_pairs:
                    self.successful_task_pairs[task_id].update(pairs)

        for task_id in self._task_queue:
            self.tasks_status[task_id] = (
                TaskStatus.COMPLETED if self._is_task_completed(task_id) else TaskStatus.NOT_STARTED
            )
        self._assign_initial_tasks()

    def _is_task_completed(self, task_id: str) -> bool:
        """Return whether every configured pair for a task has succeeded at least once."""
        return all(pair_index in self.successful_task_pairs[task_id] for pair_index in self._task_pair_queue[task_id])

    def _get_next_unexecuted_pair(self, task_id: str) -> int | None:
        """Return the first configured pair that has not yet succeeded."""
        return next(
            (
                pair_index
                for pair_index in self._task_pair_queue[task_id]
                if pair_index not in self.successful_task_pairs[task_id]
            ),
            None,
        )

    def _assign_task_to_env(self, env_id: int, task_id: str) -> None:
        """Assign an unfinished task and its first missing pair to an environment."""
        next_pair = self._get_next_unexecuted_pair(task_id)
        if next_pair is None:
            raise ValueError(f"Cannot assign completed task '{task_id}' to environment {env_id}.")

        self.current_tasks[env_id] = task_id
        self._current_task_pair_position[env_id] = self._task_pair_queue[task_id].index(next_pair)
        self.current_task_pair[env_id] = next_pair
        self._set_task_in_progress(env_id)
        self.retries_per_task[task_id] = 0

    def _assign_initial_tasks(self) -> None:
        """Assign distinct unfinished tasks to environments after loading persisted successes."""
        unfinished_tasks = [
            task_id for task_id in self._task_queue if self.tasks_status[task_id] == TaskStatus.NOT_STARTED
        ]
        assigned_task_ids: set[str] = set()
        for env_id in range(self.num_envs):
            if env_id < len(unfinished_tasks):
                task_id = unfinished_tasks[env_id]
                self._assign_task_to_env(env_id, task_id)
            else:
                # The command has no idle-task representation. Keep any excess
                # environments on distinct completed tasks rather than assigning
                # one task to multiple environments.
                task_id = next(task_id for task_id in self._task_queue if task_id not in assigned_task_ids)
                self.current_tasks[env_id] = task_id
                self._current_task_pair_position[env_id] = 0
                self.current_task_pair[env_id] = self._task_pair_queue[task_id][0]
            assigned_task_ids.add(task_id)

    def _resolve_task_queue(self, task_ids: list[str] | None) -> list[str]:
        """Return the ordered, unique task IDs available to this command term."""
        task_queue = [task.task_id for task in self.tasks] if task_ids is None else list(task_ids)
        if len(task_queue) != len(set(task_queue)):
            raise ValueError("tasks_ids must not contain duplicate task IDs.")

        unknown_task_ids = [task_id for task_id in task_queue if task_id not in self._tasks_by_id]
        if unknown_task_ids:
            raise ValueError(f"tasks_ids contains unknown task IDs: {unknown_task_ids}.")
        return task_queue

    def _resolve_task_pair_queue(self, task_pairs: list[list[int]] | None) -> dict[str, list[int]]:
        """Return the ordered pair indices to run for every task in the task queue.

        This selected-pair abstraction is retained as technical debt. Current ERA
        Bench scheduling assumes that each queue contains every raw task pair in
        its original order.
        """
        if task_pairs is None:
            return {
                task_id: list(range(len(self._tasks_by_id[task_id].train + self._tasks_by_id[task_id].test)))
                for task_id in self._task_queue
            }
        if self.cfg.tasks_ids is None:
            raise ValueError("task_pairs requires tasks_ids so each pair list has an associated task.")
        if len(task_pairs) != len(self._task_queue):
            raise ValueError(
                f"Length of task_pairs ({len(task_pairs)}) must match the number of selected tasks ({len(self._task_queue)})."
            )

        task_pair_queue: dict[str, list[int]] = {}
        for task_id, pair_indices in zip(self._task_queue, task_pairs, strict=True):
            pair_count = len(self._tasks_by_id[task_id].train + self._tasks_by_id[task_id].test)
            if not pair_indices:
                raise ValueError(f"task_pairs for task '{task_id}' must contain at least one pair index.")
            if len(pair_indices) != len(set(pair_indices)):
                raise ValueError(f"task_pairs for task '{task_id}' must not contain duplicate pair indices.")
            invalid_pair_indices = [index for index in pair_indices if index < 0 or index >= pair_count]
            if invalid_pair_indices:
                raise ValueError(
                    f"task_pairs for task '{task_id}' contains invalid pair indices {invalid_pair_indices}; "
                    f"the task has {pair_count} pairs."
                )
            task_pair_queue[task_id] = list(pair_indices)
        return task_pair_queue

    def set_task_failed(self, env_id: int) -> None:
        """Mark the current task as failed."""
        self.tasks_status[self.current_tasks[env_id]] = TaskStatus.FAILED

    def set_task_completed(self, env_id: int) -> None:
        """Record success for the current pair and trigger its reset-time transition."""
        task_id = self.current_tasks[env_id]
        self.successful_task_pairs[task_id].add(self.current_task_pair[env_id])
        self.tasks_status[task_id] = TaskStatus.COMPLETED

    def _set_task_in_progress(self, env_id: int) -> None:
        """Mark the current task as in progress."""
        self.tasks_status[self.current_tasks[env_id]] = TaskStatus.IN_PROGRESS

    @property
    def command(self) -> torch.Tensor:
        """Return object commands with shape ``(num_envs, command_dim)``.

        Each row is encoded as ``[input_count, output_count, input_objects, output_objects]``.
        Every object occupies three values: ``(x, y, color)``. Unused object slots are zero-padded.
        """
        return self._command

    @property
    def current_pixel_pairs(self) -> list[tuple[list[Pixel], list[Pixel]]]:
        """Return the unpadded input/output pixel objects for every environment."""
        pixel_pairs: list[tuple[list[Pixel], list[Pixel]]] = []
        for grid_pair in self.current_grid_pairs:

            pixel_pairs.append(
                (
                    grid_pair.to_continuous_coordinates(
                        self._cube_size,
                        self._separation,
                        grid="input",
                    ),
                    grid_pair.to_continuous_coordinates(
                        self._cube_size,
                        self._separation,
                        grid="output",
                    ),
                )
            )
        return pixel_pairs

    @property
    def current_grid_pairs(self) -> list[GridPair]:
        result = []
        for env_id in range(self.num_envs):
            current_task_id = self.current_tasks[env_id]
            current_pair_index = self.current_task_pair[env_id]
            task = self._tasks_by_id[current_task_id]
            pair = (
                task.train[current_pair_index]
                if current_pair_index < len(task.train)
                else task.test[current_pair_index - len(task.train)]
            )
            result.append(pair)
        return result

    def get_current_task_grid_pairs(self, env_id: int) -> list[GridPair]:
        """Return every train and test pair for the environment's active task."""
        task = self._tasks_by_id[self.current_tasks[env_id]]
        return [*task.train, *task.test]

    def get_features(self, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the positions of the objects in the current task for the given environment ID."""
        input_amount = int(self.command[env_id, 0].item())
        output_amount = int(self.command[env_id, 1].item())
        input_positions = self.command[
            env_id, COMMAND_HEADER_FEATURES : COMMAND_HEADER_FEATURES + input_amount * OBJECT_FEATURES
        ].view(input_amount, OBJECT_FEATURES)
        output_positions = self.command[
            env_id,
            COMMAND_HEADER_FEATURES
            + MAX_OBJECTS_PER_GRID * OBJECT_FEATURES : COMMAND_HEADER_FEATURES
            + MAX_OBJECTS_PER_GRID * OBJECT_FEATURES
            + output_amount * OBJECT_FEATURES,
        ].view(output_amount, OBJECT_FEATURES)
        return input_positions, output_positions

    def compute(self, dt):
        self._update_command()

    def prepare_reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Advance task state before reset events consume the current command.

        Isaac Lab applies reset events before it calls ``CommandManager.reset``.
        Mark the selected environments so the following framework reset refreshes
        its timer and counter without resampling their task a second time.
        """
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return

        self._resample_command(env_ids_tensor)
        self._refresh_command()
        self._prepared_for_reset[env_ids_tensor] = True
        print(
            f"[INFO]: Tasks completed: {[task_id for task_id, status in self.tasks_status.items() if status == TaskStatus.COMPLETED]}; tasks failed: {[task_id for task_id, status in self.tasks_status.items() if status == TaskStatus.FAILED]}"
        )

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor):
        """Resample tasks pairs for the given environment IDs."""
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return

        already_prepared = self._prepared_for_reset[env_ids_tensor]
        if torch.any(already_prepared):
            self._prepared_for_reset[env_ids_tensor[already_prepared]] = False
            env_ids_tensor = env_ids_tensor[~already_prepared]

        for env_id in env_ids_tensor.tolist():
            current_task_id = self.current_tasks[env_id]
            if self.tasks_status[current_task_id] == TaskStatus.FAILED:
                if self.retries_per_task[current_task_id] >= self.cfg.retries:
                    self.tasks_status[current_task_id] = TaskStatus.FAILED
                    self._transition_to_next_task(env_id)
                    continue
                self._retry_current_pair(env_id)
            elif self.tasks_status[current_task_id] == TaskStatus.COMPLETED:
                self._transition_to_next_pair(env_id)

    def _transition_to_next_pair(self, env_id: int):
        """Transition to the next pair in the current task for the given environment ID."""
        task_id = self.current_tasks[env_id]
        next_pair = self._get_next_unexecuted_pair(task_id)
        if next_pair is None:
            self._transition_to_next_task(env_id)
            return
        self._current_task_pair_position[env_id] = self._task_pair_queue[task_id].index(next_pair)
        self.current_task_pair[env_id] = next_pair
        self.retries_per_task[task_id] = 0
        self._set_task_in_progress(env_id)
        print(
            f"[INFO]: Transitioned to next pair {self.current_task_pair[env_id]} of task '{task_id}' for environment {env_id}."
        )

    def _retry_current_pair(self, env_id: int):
        """Retry the current pair in the current task for the given environment ID."""
        self.retries_per_task[self.current_tasks[env_id]] += 1
        self._set_task_in_progress(env_id)
        print(
            f"[INFO]: Retrying task '{self.current_tasks[env_id]}' for environment {env_id}. Retry {self.retries_per_task[self.current_tasks[env_id]]}/{self.cfg.retries}."
        )

    def _transition_to_next_task(self, env_id: int):
        """Transition to the next task for the given environment ID."""
        self.tasks_status[self.current_tasks[env_id]] = (
            TaskStatus.FAILED
            if self.tasks_status[self.current_tasks[env_id]] == TaskStatus.FAILED
            else TaskStatus.COMPLETED
        )
        next_task_id = self._get_next_task_id()
        if next_task_id is not None:
            self._assign_task_to_env(env_id, next_task_id)
            print(f"[INFO]: Transitioned to next task '{next_task_id}' for environment {env_id}.")

    def _get_next_task_id(self) -> str | None:
        """Get the next unassigned task that has not been completed or failed."""
        active_task_ids = set(self.current_tasks)
        for task_id in self._task_queue:
            if task_id not in active_task_ids and self.tasks_status[task_id] == TaskStatus.NOT_STARTED:
                return task_id
        return None

    def _get_finished_env_ids(self) -> list[int]:
        """Get the IDs of environments that have finished their current task."""
        return [
            env_id
            for env_id in range(self.num_envs)
            if self.tasks_status[self.current_tasks[env_id]] in {TaskStatus.COMPLETED, TaskStatus.FAILED}
        ]

    def _update_command(self):
        """Update the command for each environment based on the current task and pair index."""
        resample_env_ids = []
        for env_id in range(self.num_envs):
            current_task_id = self.current_tasks[env_id]
            if self.tasks_status[current_task_id] == TaskStatus.NOT_STARTED:
                self._set_task_in_progress(env_id)
            elif self.tasks_status[current_task_id] in {TaskStatus.FAILED, TaskStatus.COMPLETED}:
                resample_env_ids.append(env_id)

        self._resample_command(resample_env_ids)
        self._refresh_command()

    def _refresh_command(self) -> None:
        """Encode the current ragged pixel objects into the fixed-width command tensor."""
        command_state = tuple(zip(self.current_tasks, self.current_task_pair, strict=True))
        if command_state == self._command_state:
            return

        self._command.zero_()
        input_offset = COMMAND_HEADER_FEATURES
        output_offset = input_offset + MAX_OBJECTS_PER_GRID * OBJECT_FEATURES
        for env_id, (input_objects, output_objects) in enumerate(self.current_pixel_pairs):

            if len(input_objects) > MAX_OBJECTS_PER_GRID or len(output_objects) > MAX_OBJECTS_PER_GRID:
                raise ValueError(
                    f"Task '{self.current_tasks[env_id]}' has more than {MAX_OBJECTS_PER_GRID} objects in a grid."
                )

            self._command[env_id, 0] = len(input_objects)
            self._command[env_id, 1] = len(output_objects)
            if input_objects:
                input_values = torch.tensor(
                    [(pixel.x, pixel.y, pixel.color) for pixel in input_objects],
                    device=self.device,
                    dtype=self._command.dtype,
                )
                self._command[env_id, input_offset : input_offset + input_values.numel()] = input_values.flatten()
            if output_objects:
                output_values = torch.tensor(
                    [(pixel.x, pixel.y, pixel.color) for pixel in output_objects],
                    device=self.device,
                    dtype=self._command.dtype,
                )
                self._command[env_id, output_offset : output_offset + output_values.numel()] = output_values.flatten()

        self._command_state = command_state

    def _update_metrics(self):
        pass

    def is_done(self, env_id: int) -> bool:
        """Check if the task for the given environment ID is done."""
        return self.tasks_status[self.current_tasks[env_id]] in {TaskStatus.COMPLETED, TaskStatus.FAILED}
