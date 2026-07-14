"""Environment-major storage for ARC task success targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from arc_utils import ARC_COLOR_TO_COLOR, Colors


_COLOR_TO_ARC_ID = {color: arc_color for arc_color, color in ARC_COLOR_TO_COLOR.items()}


@dataclass(frozen=True)
class SuccessPositionView:
    """The current success targets for one environment instance."""

    positions_w: torch.Tensor
    """World-frame target positions with shape ``(num_targets, 3)``."""

    color_ids: torch.Tensor
    """ARC color IDs with shape ``(num_targets,)``."""

    def positions_for_color(self, color: Colors) -> torch.Tensor:
        """Return this environment's target positions for ``color``."""
        return self.positions_w[self.color_ids == _COLOR_TO_ARC_ID[color]]


class SuccessPositionBuffer:
    """Fixed-capacity, environment-major ARC success-target buffer.

    Positions and colors are deliberately separate tensors: positions remain
    geometric ``(x, y, z)`` data while ARC color stays an integer label.  The
    first tensor dimension is always the Isaac Lab environment dimension.
    """

    def __init__(self, num_envs: int, max_targets: int, device: torch.device | str):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if max_targets <= 0:
            raise ValueError(f"max_targets must be positive, got {max_targets}.")

        self.positions_w = torch.zeros((num_envs, max_targets, 3), dtype=torch.float32, device=device)
        self.color_ids = torch.zeros((num_envs, max_targets), dtype=torch.long, device=device)
        self.counts = torch.zeros((num_envs,), dtype=torch.long, device=device)

    @property
    def num_envs(self) -> int:
        return self.positions_w.shape[0]

    @property
    def max_targets(self) -> int:
        return self.positions_w.shape[1]

    @property
    def active_targets(self) -> torch.Tensor:
        """Return shape ``(num_envs, max_targets)`` for currently populated slots."""
        target_indices = torch.arange(self.max_targets, device=self.positions_w.device)
        return target_indices.unsqueeze(0) < self.counts.unsqueeze(1)

    @property
    def has_targets(self) -> torch.Tensor:
        """Return one boolean per environment indicating whether it has targets."""
        return self.counts > 0

    def __getitem__(self, env_id: int) -> SuccessPositionView:
        """Return all valid target information for one environment."""
        self._validate_env_id(env_id)
        count = int(self.counts[env_id].item())
        return SuccessPositionView(
            positions_w=self.positions_w[env_id, :count],
            color_ids=self.color_ids[env_id, :count],
        )

    def color_mask(self, color: Colors) -> torch.Tensor:
        """Return an environment-major mask selecting populated targets of ``color``."""
        return self.active_targets & (self.color_ids == _COLOR_TO_ARC_ID[color])

    def replace(self, env_id: int, positions_w: torch.Tensor, color_ids: torch.Tensor) -> None:
        """Atomically replace every success target belonging to one environment."""
        self._validate_env_id(env_id)
        positions_w = positions_w.to(device=self.positions_w.device, dtype=self.positions_w.dtype)
        color_ids = color_ids.to(device=self.color_ids.device, dtype=self.color_ids.dtype)

        if positions_w.ndim != 2 or positions_w.shape[1] != 3:
            raise ValueError(f"positions_w must have shape (N, 3), got {tuple(positions_w.shape)}.")
        if color_ids.ndim != 1:
            raise ValueError(f"color_ids must have shape (N,), got {tuple(color_ids.shape)}.")
        if positions_w.shape[0] != color_ids.shape[0]:
            raise ValueError(
                f"positions_w and color_ids must have the same length, got {positions_w.shape[0]} and {color_ids.shape[0]}."
            )
        if positions_w.shape[0] > self.max_targets:
            raise ValueError(
                f"Environment {env_id} has {positions_w.shape[0]} success targets, exceeding capacity {self.max_targets}."
            )
        valid_arc_ids = torch.tensor(tuple(ARC_COLOR_TO_COLOR), device=self.color_ids.device, dtype=self.color_ids.dtype)
        if color_ids.numel() > 0 and not bool(torch.isin(color_ids, valid_arc_ids).all()):
            raise ValueError("color_ids must contain supported, non-background ARC color IDs.")

        self.positions_w[env_id].zero_()
        self.color_ids[env_id].zero_()
        target_count = positions_w.shape[0]
        if target_count > 0:
            self.positions_w[env_id, :target_count].copy_(positions_w)
            self.color_ids[env_id, :target_count].copy_(color_ids)
        self.counts[env_id] = target_count

    def clear(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        """Clear the targets for selected environments, or all environments."""
        if env_ids is None:
            self.positions_w.zero_()
            self.color_ids.zero_()
            self.counts.zero_()
            return

        env_ids_tensor = torch.as_tensor(env_ids, device=self.positions_w.device, dtype=torch.long)
        self.positions_w[env_ids_tensor] = 0.0
        self.color_ids[env_ids_tensor] = 0
        self.counts[env_ids_tensor] = 0

    def _validate_env_id(self, env_id: int) -> None:
        if env_id < 0 or env_id >= self.num_envs:
            raise IndexError(f"Environment ID {env_id} is outside [0, {self.num_envs}).")
