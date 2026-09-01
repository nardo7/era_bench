from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ProgressWatchdogResult:
    """Result returned by :class:`ProgressWatchdog.update`.

    All tensors are shaped ``(num_envs,)`` and live on the same device as the watchdog.
    """

    error: torch.Tensor
    improved: torch.Tensor
    stalled: torch.Tensor
    diverged: torch.Tensor
    stagnant_steps: torch.Tensor
    best_error: torch.Tensor

    @property
    def triggered(self) -> torch.Tensor:
        """Boolean mask for environments that should trigger a recovery action."""
        return self.stalled | self.diverged


class ProgressWatchdog:
    """Tracks per-environment progress toward a scalar error target.

    The watchdog is intentionally generic: callers provide a scalar error per environment
    such as Cartesian position error, pose error, or planner distance-to-goal. The watchdog
    then detects two common failure modes:

    - ``stalled``: the error does not improve enough for ``patience`` consecutive updates.
    - ``diverged``: the error increases beyond the best seen value by ``divergence_ratio``.

    This is useful for local IK solves where a command can occasionally enter a poor basin
    of attraction from an unlucky source pose.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        patience: int = 8,
        min_improvement: float = 1e-3,
        divergence_ratio: float | None = 1.5,
        divergence_abs_delta: float | None = None,
        warmup_steps: int = 0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"Expected num_envs > 0, got {num_envs}.")
        if patience <= 0:
            raise ValueError(f"Expected patience > 0, got {patience}.")
        if min_improvement < 0.0:
            raise ValueError(f"Expected min_improvement >= 0, got {min_improvement}.")
        if divergence_ratio is not None and divergence_ratio < 1.0:
            raise ValueError(f"Expected divergence_ratio >= 1.0, got {divergence_ratio}.")
        if divergence_abs_delta is not None and divergence_abs_delta < 0.0:
            raise ValueError(f"Expected divergence_abs_delta >= 0.0, got {divergence_abs_delta}.")
        if warmup_steps < 0:
            raise ValueError(f"Expected warmup_steps >= 0, got {warmup_steps}.")

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.patience = patience
        self.min_improvement = float(min_improvement)
        self.divergence_ratio = divergence_ratio
        self.divergence_abs_delta = divergence_abs_delta
        self.warmup_steps = warmup_steps

        self._step_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._stagnant_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._best_error = torch.full((self.num_envs,), torch.inf, dtype=torch.float32, device=self.device)
        self._last_error = torch.full((self.num_envs,), torch.inf, dtype=torch.float32, device=self.device)

    @property
    def best_error(self) -> torch.Tensor:
        return self._best_error

    @property
    def stagnant_steps(self) -> torch.Tensor:
        return self._stagnant_steps

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset internal state for all or a subset of environments."""
        resolved_env_ids = self._resolve_env_ids(env_ids)
        self._step_count[resolved_env_ids] = 0
        self._stagnant_steps[resolved_env_ids] = 0
        self._best_error[resolved_env_ids] = torch.inf
        self._last_error[resolved_env_ids] = torch.inf

    def update(self, error: torch.Tensor) -> ProgressWatchdogResult:
        """Update the watchdog with the current scalar error for each environment.

        Args:
            error: Tensor of shape ``(num_envs,)`` or ``(num_envs, 1)``.

        Returns:
            A :class:`ProgressWatchdogResult` with per-environment status flags.
        """
        error = self._normalize_error(error)

        self._step_count += 1

        improvement = self._best_error - error
        improved = improvement > self.min_improvement
        self._best_error.copy_(torch.minimum(self._best_error, error))

        self._stagnant_steps.copy_(
            torch.where(improved, torch.zeros_like(self._stagnant_steps), self._stagnant_steps + 1)
        )

        warmup_mask = self._step_count <= self.warmup_steps
        stalled = (~warmup_mask) & (self._stagnant_steps >= self.patience)

        if self.divergence_ratio is None:
            diverged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            finite_best = torch.isfinite(self._best_error) & (self._best_error > 0.0)
            diverged = (~warmup_mask) & finite_best & (error > self._best_error * self.divergence_ratio)
        if self.divergence_abs_delta is not None:
            finite_best = torch.isfinite(self._best_error)
            diverged = diverged | (
                (~warmup_mask) & finite_best & (error > self._best_error + self.divergence_abs_delta)
            )

        self._last_error.copy_(error)

        return ProgressWatchdogResult(
            error=error,
            improved=improved,
            stalled=stalled,
            diverged=diverged,
            stagnant_steps=self._stagnant_steps.clone(),
            best_error=self._best_error.clone(),
        )

    def _normalize_error(self, error: torch.Tensor) -> torch.Tensor:
        if error.ndim == 2 and error.shape[1] == 1:
            error = error.squeeze(1)
        if error.ndim != 1 or error.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected error shape ({self.num_envs},) or ({self.num_envs}, 1), got {tuple(error.shape)}."
            )
        if not torch.is_floating_point(error):
            error = error.float()
        return error.to(device=self.device, dtype=torch.float32)

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
