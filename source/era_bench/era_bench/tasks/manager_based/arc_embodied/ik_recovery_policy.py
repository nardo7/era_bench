from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import torch

import isaaclab.utils.math as math_utils

from .arc_state_machine import ArcStates
from .ik_watchdog import ArcIKWatchdogUpdate


@dataclass(frozen=True)
class ArcIKRecoveryUpdate:
    """Structured output of :class:`ArcIKRecoveryPolicy.update`."""

    command: torch.Tensor
    active_mask: torch.Tensor
    messages: tuple[str, ...]


class ArcIKRecoveryPolicy:
    """Runtime recovery policy for ARC IK divergence during placing-above moves.

    The policy is intentionally kept out of the state machine. It sits at the execution layer
    and only overrides the nominal command after the watchdog reports divergence.
    """

    _PHASE_IDLE = 0
    _PHASE_RETREAT = 1
    _PHASE_RETRY = 2

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        watched_state: ArcStates = ArcStates.APPROACHING_PLACING_ABOVE,
        retreat_height: float = 0.12,
        retreat_position_tolerance: float = 0.015,
        max_retreat_steps: int = 12,
        retry_yaw_offsets_deg: tuple[float, ...] = (0.0, 15.0, -15.0),
        log_path: str | None = None,
        session_label: str | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"Expected num_envs > 0, got {num_envs}.")
        if retreat_height <= 0.0:
            raise ValueError(f"Expected retreat_height > 0, got {retreat_height}.")
        if retreat_position_tolerance <= 0.0:
            raise ValueError(f"Expected retreat_position_tolerance > 0, got {retreat_position_tolerance}.")
        if max_retreat_steps <= 0:
            raise ValueError(f"Expected max_retreat_steps > 0, got {max_retreat_steps}.")
        if not retry_yaw_offsets_deg:
            raise ValueError("Expected at least one retry yaw offset.")

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.watched_state = watched_state
        self.retreat_height = retreat_height
        self.retreat_position_tolerance = retreat_position_tolerance
        self.max_retreat_steps = max_retreat_steps
        self.retry_yaw_offsets_deg = retry_yaw_offsets_deg
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
        messages: list[str] = []

        active_state_mask = arc_state == self.watched_state.python_value
        inactive_env_ids = torch.nonzero(~active_state_mask, as_tuple=False).flatten()
        if inactive_env_ids.numel() > 0:
            self.reset(inactive_env_ids)

        trigger_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if watchdog_update.warning_env_ids.numel() > 0:
            trigger_mask[watchdog_update.warning_env_ids.to(device=self.device, dtype=torch.long)] = True

        for env_id in (
            torch.nonzero(trigger_mask & active_state_mask & ~self._exhausted, as_tuple=False).flatten().tolist()
        ):
            if self._phase[env_id].item() == self._PHASE_IDLE:
                self._attempt_idx[env_id] = 0
                self._start_retreat(env_id, tcp_position, tcp_orientation, target_command)
                messages.append(f"[WARNING]: Arc IK recovery started env={env_id} strategy=retreat-retry attempt=0")
            elif self._phase[env_id].item() == self._PHASE_RETRY:
                next_attempt = int(self._attempt_idx[env_id].item()) + 1
                if next_attempt < len(self.retry_yaw_offsets_deg):
                    self._attempt_idx[env_id] = next_attempt
                    self._start_retreat(env_id, tcp_position, tcp_orientation, target_command)
                    messages.append(
                        "[WARNING]: Arc IK recovery escalating "
                        f"env={env_id} strategy=retreat-retry attempt={next_attempt} "
                        f"yaw_offset_deg={self.retry_yaw_offsets_deg[next_attempt]:.1f}"
                    )
                else:
                    self._phase[env_id] = self._PHASE_IDLE
                    self._exhausted[env_id] = True
                    messages.append(
                        f"[WARNING]: Arc IK recovery exhausted env={env_id} attempts={len(self.retry_yaw_offsets_deg)}"
                    )

        retreat_env_ids = torch.nonzero(self._phase == self._PHASE_RETREAT, as_tuple=False).flatten()
        if retreat_env_ids.numel() > 0:
            command[retreat_env_ids] = self._retreat_command[retreat_env_ids]
            self._retreat_steps[retreat_env_ids] += 1
            retreat_reached = self._retreat_reached(retreat_env_ids, tcp_position)
            retreat_timed_out = self._retreat_steps[retreat_env_ids] >= self.max_retreat_steps
            retry_env_ids = retreat_env_ids[retreat_reached | retreat_timed_out]
            for env_id in retry_env_ids.tolist():
                self._start_retry(env_id, target_command)
                messages.append(
                    "[INFO]: Arc IK recovery switching to retry "
                    f"env={env_id} attempt={int(self._attempt_idx[env_id].item())} "
                    f"yaw_offset_deg={self.retry_yaw_offsets_deg[int(self._attempt_idx[env_id].item())]:.1f}"
                )

        retry_env_ids = torch.nonzero(self._phase == self._PHASE_RETRY, as_tuple=False).flatten()
        if retry_env_ids.numel() > 0:
            command[retry_env_ids] = self._retry_command[retry_env_ids]

        active_recovery_mask = self._phase != self._PHASE_IDLE
        self._log_messages(messages, arc_state=arc_state, step=step, elapsed_s=elapsed_s)
        return ArcIKRecoveryUpdate(command=command, active_mask=active_recovery_mask, messages=tuple(messages))

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
        self._retreat_command[env_id, :3] = tcp_position[env_id]
        self._retreat_command[env_id, 2] = tcp_position[env_id, 2] + self.retreat_height
        self._retreat_command[env_id, 3:7] = tcp_orientation[env_id]
        self._retreat_command[env_id, 7] = target_command[env_id, 7]

    def _start_retry(self, env_id: int, target_command: torch.Tensor) -> None:
        self._phase[env_id] = self._PHASE_RETRY
        self._retry_command[env_id] = target_command[env_id]
        yaw_offset_deg = self.retry_yaw_offsets_deg[int(self._attempt_idx[env_id].item())]
        if abs(yaw_offset_deg) < 1e-9:
            return

        yaw_tensor = torch.tensor([math.radians(yaw_offset_deg)], device=self.device, dtype=target_command.dtype)
        zeros = torch.zeros_like(yaw_tensor)
        yaw_quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw_tensor)
        base_quat = target_command[env_id, 3:7].unsqueeze(0)
        rotated_quat = math_utils.quat_mul(yaw_quat, base_quat)
        self._retry_command[env_id, 3:7] = rotated_quat[0]

    def _retreat_reached(self, env_ids: torch.Tensor, tcp_position: torch.Tensor) -> torch.Tensor:
        retreat_error = torch.linalg.vector_norm(
            self._retreat_command[env_ids, :3] - tcp_position[env_ids],
            dim=1,
        )
        return retreat_error <= self.retreat_position_tolerance
