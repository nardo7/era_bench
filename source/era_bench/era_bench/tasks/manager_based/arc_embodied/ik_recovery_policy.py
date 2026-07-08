from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from .arc_state_machine import ArcStates
from .ik_watchdog import ArcIKWatchdogUpdate


class _CuroboPlannerLike(Protocol):
    def plan_grasp_center_goal_with_fixed_yaw_candidates(self, grasp_center_goal: torch.Tensor) -> bool: ...


def _normalize_watched_states(watched_states: ArcStates | Sequence[ArcStates]) -> tuple[ArcStates, ...]:
    if isinstance(watched_states, ArcStates):
        return (watched_states,)

    normalized_states = tuple(watched_states)
    if not normalized_states:
        raise ValueError("Expected at least one watched state.")
    if not all(isinstance(state, ArcStates) for state in normalized_states):
        raise TypeError("Expected watched_states to contain only ArcStates values.")
    return normalized_states


@dataclass(frozen=True)
class ArcIKRecoveryUpdate:
    """Structured output of :class:`ArcIKRecoveryPolicy.update`."""

    command: torch.Tensor
    active_mask: torch.Tensor
    watchdog_reset_env_ids: torch.Tensor
    execute_curobo_env_ids: torch.Tensor
    messages: tuple[str, ...]


class ArcIKRecoveryPolicy:
    """Runtime recovery policy for ARC IK divergence during placing-above moves.

    The policy is intentionally kept out of the state machine. It sits at the execution layer
    and only overrides the nominal command after the watchdog reports divergence.
    """

    _PHASE_IDLE = 0
    _PHASE_RETREAT = 1

    @staticmethod
    def _failure_mode_for_env(watchdog_update: ArcIKWatchdogUpdate, env_id: int) -> str:
        if watchdog_update.watchdog_result.diverged[env_id].item():
            return "divergence"
        if watchdog_update.watchdog_result.stalled[env_id].item():
            return "stall"
        return "issue"

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        watched_states: ArcStates | Sequence[ArcStates] = ArcStates.APPROACHING_PLACING_ABOVE,
        retreat_height: float = 0.05,
        retreat_height_step: float = 0.03,
        retreat_lateral_radius: float = 0.05,
        retreat_lateral_radius_step: float = 0.03,
        retreat_position_tolerance: float = 0.015,
        max_retreat_steps: int = 12,
        curobo_planner: _CuroboPlannerLike | None = None,
        log_path: str | None = None,
        session_label: str | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"Expected num_envs > 0, got {num_envs}.")
        if retreat_height <= 0.0:
            raise ValueError(f"Expected retreat_height > 0, got {retreat_height}.")
        if retreat_height_step < 0.0:
            raise ValueError(f"Expected retreat_height_step >= 0, got {retreat_height_step}.")
        if retreat_lateral_radius < 0.0:
            raise ValueError(f"Expected retreat_lateral_radius >= 0, got {retreat_lateral_radius}.")
        if retreat_lateral_radius_step < 0.0:
            raise ValueError(f"Expected retreat_lateral_radius_step >= 0, got {retreat_lateral_radius_step}.")
        if retreat_position_tolerance <= 0.0:
            raise ValueError(f"Expected retreat_position_tolerance > 0, got {retreat_position_tolerance}.")
        if max_retreat_steps <= 0:
            raise ValueError(f"Expected max_retreat_steps > 0, got {max_retreat_steps}.")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.watched_states = _normalize_watched_states(watched_states)
        self._watched_state_values = torch.tensor(
            [state.python_value for state in self.watched_states],
            dtype=torch.int64,
            device=self.device,
        )
        self.retreat_height = retreat_height
        self.retreat_height_step = retreat_height_step
        self.retreat_lateral_radius = retreat_lateral_radius
        self.retreat_lateral_radius_step = retreat_lateral_radius_step
        self.retreat_position_tolerance = retreat_position_tolerance
        self.max_retreat_steps = max_retreat_steps
        self._curobo_planner = curobo_planner
        self.log_path = log_path or self._default_log_path()
        self.session_label = session_label or "arc-ik-recovery"

        self._phase = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._attempt_idx = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._retreat_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._exhausted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._retreat_command = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.device)
        self._retry_command = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.device)

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8", buffering=1) as log_file:
            log_file.write(
                f"\n=== Arc IK recovery session start label={self.session_label} time={time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        self._phase[env_ids] = self._PHASE_IDLE
        self._attempt_idx[env_ids] = 0
        self._retreat_steps[env_ids] = 0
        self._exhausted[env_ids] = False
        self._retreat_command[env_ids] = 0.0
        self._retry_command[env_ids] = 0.0

    def update(
        self,
        arc_state: torch.Tensor,
        tcp_position: torch.Tensor,
        tcp_orientation: torch.Tensor,
        target_command: torch.Tensor,
        watchdog_update: ArcIKWatchdogUpdate,
        *,
        step: int | None = None,
        elapsed_s: float | None = None,
    ) -> ArcIKRecoveryUpdate:
        """Compute the command to execute for this step.

        The returned command always has the same shape as ``target_command`` and can be sent
        directly to ``env.step``.
        """
        if target_command.ndim != 2 or target_command.shape[0] != self.num_envs or target_command.shape[1] < 8:
            raise ValueError(f"Expected target_command shape ({self.num_envs}, 8+) got {tuple(target_command.shape)}.")

        command = target_command.clone()
        messages: list[str] = list(watchdog_update.warning_messages)
        watchdog_reset_env_ids: list[int] = []
        execute_curobo_env_ids: list[int] = []
        started_retreat_env_ids: list[int] = []

        active_state_mask = torch.isin(
            arc_state,
            self._watched_state_values.to(device=arc_state.device, dtype=arc_state.dtype),
        )
        inactive_env_ids = torch.nonzero(~active_state_mask, as_tuple=False).flatten()
        exiting_recovery_env_ids = inactive_env_ids[self._phase[inactive_env_ids] == self._PHASE_RETREAT]
        if exiting_recovery_env_ids.numel() > 0:
            watchdog_reset_env_ids.extend(exiting_recovery_env_ids.tolist())
        if inactive_env_ids.numel() > 0:
            self.reset(inactive_env_ids)

        trigger_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if watchdog_update.warning_env_ids.numel() > 0:
            trigger_mask[watchdog_update.warning_env_ids.to(device=self.device, dtype=torch.long)] = True

        for env_id in (
            torch.nonzero(trigger_mask & active_state_mask & ~self._exhausted, as_tuple=False).flatten().tolist()
        ):
            failure_mode = self._failure_mode_for_env(watchdog_update, env_id)
            if self._phase[env_id].item() == self._PHASE_IDLE:
                self._attempt_idx[env_id] = 0
                self._start_retreat(env_id, tcp_position, tcp_orientation, target_command)
                started_retreat_env_ids.append(env_id)
                messages.append(
                    "[WARNING]: Arc IK recovery started "
                    f"env={env_id} strategy=retreat-curobo trigger={failure_mode} attempt=0"
                )

        retreat_env_ids = torch.nonzero(self._phase == self._PHASE_RETREAT, as_tuple=False).flatten()
        if retreat_env_ids.numel() > 0:
            watchdog_reset_env_ids.extend(retreat_env_ids.tolist())
            command[retreat_env_ids] = self._retreat_command[retreat_env_ids]
            self._retreat_steps[retreat_env_ids] += 1
            retreat_reached = self._retreat_reached(retreat_env_ids, tcp_position)
            retreat_timed_out = self._retreat_steps[retreat_env_ids] >= self.max_retreat_steps
            planner_env_mask = retreat_reached | retreat_timed_out
            if started_retreat_env_ids:
                started_retreat_mask = torch.zeros_like(planner_env_mask)
                for started_env_id in started_retreat_env_ids:
                    started_matches = retreat_env_ids == started_env_id
                    started_retreat_mask |= started_matches
                planner_env_mask &= ~started_retreat_mask
            planner_env_ids = retreat_env_ids[planner_env_mask]
            for env_id in planner_env_ids.tolist():
                watchdog_reset_env_ids.append(env_id)
                self._phase[env_id] = self._PHASE_IDLE
                self._retry_command[env_id] = target_command[env_id]
                if self._attempt_curobo_plan(env_id, target_command[env_id, :7]):
                    execute_curobo_env_ids.append(env_id)
                    messages.append(
                        "[INFO]: Arc IK recovery escalating to curobo "
                        f"env={env_id} attempt={int(self._attempt_idx[env_id].item())} result=planned"
                    )
                else:
                    self._exhausted[env_id] = True
                    messages.append(
                        "[WARNING]: Arc IK recovery exhausted " f"env={env_id} attempts=1 planner=curobo result=failed"
                    )

        active_recovery_mask = self._phase != self._PHASE_IDLE
        self._log_messages(messages, arc_state=arc_state, step=step, elapsed_s=elapsed_s)
        return ArcIKRecoveryUpdate(
            command=command,
            active_mask=active_recovery_mask,
            watchdog_reset_env_ids=torch.tensor(watchdog_reset_env_ids, device=self.device, dtype=torch.long),
            execute_curobo_env_ids=torch.tensor(execute_curobo_env_ids, device=self.device, dtype=torch.long),
            messages=tuple(messages),
        )

    def _default_log_path(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "logs", "arc_ik_recovery.log")
        )

    def _log_messages(
        self,
        messages: list[str],
        *,
        arc_state: torch.Tensor,
        step: int | None,
        elapsed_s: float | None,
    ) -> None:
        if not messages:
            return
        step_value = -1 if step is None else step
        elapsed_value = -1.0 if elapsed_s is None else elapsed_s
        state_name = ArcStates(int(arc_state[0].item())).name
        with open(self.log_path, "a", encoding="utf-8", buffering=1) as log_file:
            for message in messages:
                log_file.write(f"elapsed_s={elapsed_value:.2f} step={step_value} state={state_name} {message}\n")

    def _start_retreat(
        self,
        env_id: int,
        tcp_position: torch.Tensor,
        tcp_orientation: torch.Tensor,
        target_command: torch.Tensor,
    ) -> None:
        self._phase[env_id] = self._PHASE_RETREAT
        self._retreat_steps[env_id] = 0
        self._retreat_command[env_id] = 0.0
        attempt_idx = int(self._attempt_idx[env_id].item())
        retreat_height = self.retreat_height + self.retreat_height_step * float(attempt_idx)
        retreat_lateral_radius = self.retreat_lateral_radius + self.retreat_lateral_radius_step * float(attempt_idx)
        retreat_offset = self._retreat_offset_for_attempt(
            attempt_idx=attempt_idx,
            retreat_height=retreat_height,
            retreat_lateral_radius=retreat_lateral_radius,
            dtype=target_command.dtype,
        )
        self._retreat_command[env_id, :3] = tcp_position[env_id] + retreat_offset
        self._retreat_command[env_id, 3:7] = tcp_orientation[env_id]
        self._retreat_command[env_id, 7] = target_command[env_id, 7]

    def _retreat_offset_for_attempt(
        self,
        *,
        attempt_idx: int,
        retreat_height: float,
        retreat_lateral_radius: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if retreat_lateral_radius <= 0.0:
            return torch.tensor([0.0, 0.0, retreat_height], device=self.device, dtype=dtype)

        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        azimuth = golden_angle * float(attempt_idx)
        lateral_x = retreat_lateral_radius * math.cos(azimuth)
        lateral_y = retreat_lateral_radius * math.sin(azimuth)
        return torch.tensor([lateral_x, lateral_y, retreat_height], device=self.device, dtype=dtype)

    def _attempt_curobo_plan(self, env_id: int, goal_command: torch.Tensor) -> bool:
        if self._curobo_planner is None:
            return False
        if env_id != 0:
            return False
        return bool(self._curobo_planner.plan_grasp_center_goal_with_fixed_yaw_candidates(goal_command))

    def _retreat_reached(self, env_ids: torch.Tensor, tcp_position: torch.Tensor) -> torch.Tensor:
        retreat_error = torch.linalg.vector_norm(
            self._retreat_command[env_ids, :3] - tcp_position[env_ids],
            dim=1,
        )
        return retreat_error <= self.retreat_position_tolerance
