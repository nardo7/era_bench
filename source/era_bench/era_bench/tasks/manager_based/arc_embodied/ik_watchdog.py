from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from ..common.progress_watchdog import ProgressWatchdog, ProgressWatchdogResult
from .arc_state_machine import ArcStates


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
class ArcIKWatchdogUpdate:
    """Structured output of :class:`ArcIKWatchdog.update`.

    The wrapper keeps the script-facing API small while still exposing enough detail for
    debugging or future recovery actions.
    """

    warning_messages: tuple[str, ...]
    warning_env_ids: torch.Tensor
    active_mask: torch.Tensor
    watchdog_result: ProgressWatchdogResult


class ArcIKWatchdog:
    """ARC-specific wrapper for monitoring IK progress during placing-above moves."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        watched_states: ArcStates | Sequence[ArcStates] = ArcStates.APPROACHING_PLACING_ABOVE,
        patience: int = 10,
        divergence_patience: int = 10,
        min_improvement: float = 1e-3,
        divergence_ratio: float = 1.25,
        divergence_abs_delta: float = 0.05,
        warmup_steps: int = 3,
        rewarn_ratio: float = 1.1,
        rewarn_abs_delta: float = 0.05,
    ) -> None:
        self._watched_states = _normalize_watched_states(watched_states)
        if divergence_patience <= 0:
            raise ValueError(f"Expected divergence_patience > 0, got {divergence_patience}.")
        self._divergence_patience = divergence_patience
        self._rewarn_ratio = rewarn_ratio
        self._rewarn_abs_delta = rewarn_abs_delta
        self._watchdog = ProgressWatchdog(
            num_envs=num_envs,
            device=device,
            patience=patience,
            min_improvement=min_improvement,
            divergence_ratio=divergence_ratio,
            divergence_abs_delta=divergence_abs_delta,
            warmup_steps=warmup_steps,
        )
        self._watched_state_values = torch.tensor(
            [state.python_value for state in self._watched_states],
            dtype=torch.int64,
            device=self._watchdog.device,
        )
        self._prev_stalled = torch.zeros(num_envs, dtype=torch.bool, device=self._watchdog.device)
        self._prev_diverged = torch.zeros(num_envs, dtype=torch.bool, device=self._watchdog.device)
        self._divergent_steps = torch.zeros(num_envs, dtype=torch.int64, device=self._watchdog.device)
        self._last_warned_error = torch.full((num_envs,), torch.inf, dtype=torch.float32, device=self._watchdog.device)

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        self._watchdog.reset(env_ids)
        if env_ids is None:
            self._prev_stalled.zero_()
            self._prev_diverged.zero_()
            self._divergent_steps.zero_()
            self._last_warned_error.fill_(torch.inf)
            return
        if isinstance(env_ids, torch.Tensor):
            resolved_env_ids = env_ids.to(device=self._watchdog.device, dtype=torch.long)
        else:
            resolved_env_ids = torch.tensor(env_ids, device=self._watchdog.device, dtype=torch.long)
        self._prev_stalled[resolved_env_ids] = False
        self._prev_diverged[resolved_env_ids] = False
        self._divergent_steps[resolved_env_ids] = 0
        self._last_warned_error[resolved_env_ids] = torch.inf

    def update(
        self,
        arc_state: torch.Tensor,
        tcp_position: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> ArcIKWatchdogUpdate:
        """Update the watchdog and return any newly triggered divergence warnings.

        Args:
            arc_state: Integer tensor shaped ``(num_envs,)``.
            tcp_position: Current TCP position shaped ``(num_envs, 3)``.
            target_pose: Target pose shaped ``(num_envs, 7+)``. Only xyz is used.
        """
        active_mask = torch.isin(
            arc_state,
            self._watched_state_values.to(device=arc_state.device, dtype=arc_state.dtype),
        )

        inactive_env_ids = torch.nonzero(~active_mask, as_tuple=False).flatten()
        if inactive_env_ids.numel() > 0:
            self.reset(inactive_env_ids)

        error = torch.full((self._watchdog.num_envs,), torch.inf, device=self._watchdog.device)
        if torch.any(active_mask):
            error[active_mask] = torch.linalg.vector_norm(
                target_pose[active_mask, :3] - tcp_position[active_mask], dim=1
            )

        watchdog_result = self._watchdog.update(error)
        new_stall = active_mask & watchdog_result.stalled & ~self._prev_stalled
        raw_divergence = active_mask & watchdog_result.diverged
        self._divergent_steps.copy_(
            torch.where(raw_divergence, self._divergent_steps + 1, torch.zeros_like(self._divergent_steps))
        )
        confirmed_divergence = raw_divergence & (self._divergent_steps >= self._divergence_patience)
        warning_watchdog_result = replace(watchdog_result, diverged=confirmed_divergence)
        new_divergence = confirmed_divergence & ~self._prev_diverged
        worsening_threshold = torch.maximum(
            self._last_warned_error * self._rewarn_ratio,
            self._last_warned_error + self._rewarn_abs_delta,
        )
        worsened_divergence = confirmed_divergence & self._prev_diverged & (watchdog_result.error > worsening_threshold)
        warning_env_ids = torch.nonzero(new_stall | new_divergence | worsened_divergence, as_tuple=False).flatten()

        warning_messages = tuple(
            self._format_warning(env_id=int(env_id), arc_state=arc_state, watchdog_result=warning_watchdog_result)
            for env_id in warning_env_ids.tolist()
        )

        if warning_env_ids.numel() > 0:
            self._last_warned_error[warning_env_ids] = watchdog_result.error[warning_env_ids]

        self._prev_stalled.copy_(
            torch.where(
                active_mask,
                watchdog_result.stalled,
                torch.zeros_like(self._prev_stalled),
            )
        )
        self._prev_diverged.copy_(
            torch.where(
                active_mask,
                confirmed_divergence,
                torch.zeros_like(self._prev_diverged),
            )
        )

        return ArcIKWatchdogUpdate(
            warning_messages=warning_messages,
            warning_env_ids=warning_env_ids,
            active_mask=active_mask,
            watchdog_result=watchdog_result,
        )

    def _format_warning(
        self,
        *,
        env_id: int,
        arc_state: torch.Tensor,
        watchdog_result: ProgressWatchdogResult,
    ) -> str:
        state_name = ArcStates(int(arc_state[env_id].item())).name
        if watchdog_result.diverged[env_id].item():
            failure_mode = "divergence"
        elif watchdog_result.stalled[env_id].item():
            failure_mode = "stall"
        else:
            failure_mode = "issue"
        return (
            f"[WARNING]: Arc IK watchdog detected {failure_mode} "
            f"env={env_id} state={state_name} "
            f"error={watchdog_result.error[env_id].item():.4f} "
            f"best_error={watchdog_result.best_error[env_id].item():.4f} "
            f"stagnant_steps={int(watchdog_result.stagnant_steps[env_id].item())} "
            f"stalled={bool(watchdog_result.stalled[env_id].item())} "
            f"diverged={bool(watchdog_result.diverged[env_id].item())}"
        )
