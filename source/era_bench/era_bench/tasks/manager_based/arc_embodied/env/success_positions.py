"""Environment-major colored spatial state for ARC targets and reserve stacks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from arc_utils import ARC_COLOR_TO_COLOR, Colors


_COLOR_TO_ARC_ID = {color: arc_color for arc_color, color in ARC_COLOR_TO_COLOR.items()}
_NUM_ARC_COLORS = max(ARC_COLOR_TO_COLOR) + 1


@dataclass(frozen=True)
class ColoredSpatialView:
    """Colored spatial items belonging to one environment."""

    values_w: torch.Tensor
    """World-frame values with an item dimension first."""

    color_ids: torch.Tensor
    """ARC color ID for every item."""

    def values_for_color_id(self, arc_color_id: int) -> torch.Tensor:
        return self.values_w[self.color_ids == arc_color_id]

    def values_for_color(self, color: Colors) -> torch.Tensor:
        return self.values_for_color_id(_COLOR_TO_ARC_ID[color])


class ColoredSpatialBuffer:
    """Fixed-capacity environment-major storage shared by colored spatial data."""

    def __init__(
        self,
        num_envs: int,
        max_items: int,
        item_shape: tuple[int, ...],
        device: torch.device | str,
    ):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if max_items <= 0:
            raise ValueError(f"max_items must be positive, got {max_items}.")

        self.values_w = torch.zeros((num_envs, max_items, *item_shape), dtype=torch.float32, device=device)
        self.color_ids = torch.zeros((num_envs, max_items), dtype=torch.long, device=device)
        self.counts = torch.zeros((num_envs,), dtype=torch.long, device=device)

    @property
    def num_envs(self) -> int:
        return self.values_w.shape[0]

    @property
    def max_items(self) -> int:
        return self.values_w.shape[1]

    @property
    def active_items(self) -> torch.Tensor:
        item_indices = torch.arange(self.max_items, device=self.values_w.device)
        return item_indices.unsqueeze(0) < self.counts.unsqueeze(1)

    def _replace(self, env_id: int, values_w: torch.Tensor, color_ids: torch.Tensor) -> None:
        self._validate_env_id(env_id)
        values_w = values_w.to(device=self.values_w.device, dtype=self.values_w.dtype)
        color_ids = color_ids.to(device=self.color_ids.device, dtype=self.color_ids.dtype)

        if values_w.ndim != self.values_w.ndim - 1 or tuple(values_w.shape[1:]) != tuple(self.values_w.shape[2:]):
            raise ValueError(
                f"values_w must have shape (N, {tuple(self.values_w.shape[2:])}), got {tuple(values_w.shape)}."
            )
        if color_ids.ndim != 1 or values_w.shape[0] != color_ids.shape[0]:
            raise ValueError("values_w and color_ids must have matching first dimensions.")
        if values_w.shape[0] > self.max_items:
            raise ValueError(f"Environment {env_id} has {values_w.shape[0]} items, exceeding capacity {self.max_items}.")
        valid_arc_ids = torch.tensor(tuple(ARC_COLOR_TO_COLOR), device=self.color_ids.device, dtype=self.color_ids.dtype)
        if color_ids.numel() > 0 and not bool(torch.isin(color_ids, valid_arc_ids).all()):
            raise ValueError("color_ids must contain supported, non-background ARC color IDs.")

        self.values_w[env_id].zero_()
        self.color_ids[env_id].zero_()
        count = values_w.shape[0]
        if count:
            self.values_w[env_id, :count].copy_(values_w)
            self.color_ids[env_id, :count].copy_(color_ids)
        self.counts[env_id] = count

    def clear(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        if env_ids is None:
            self.values_w.zero_()
            self.color_ids.zero_()
            self.counts.zero_()
            return
        env_ids_tensor = torch.as_tensor(env_ids, device=self.values_w.device, dtype=torch.long)
        self.values_w[env_ids_tensor] = 0.0
        self.color_ids[env_ids_tensor] = 0
        self.counts[env_ids_tensor] = 0

    def _validate_env_id(self, env_id: int) -> None:
        if env_id < 0 or env_id >= self.num_envs:
            raise IndexError(f"Environment ID {env_id} is outside [0, {self.num_envs}).")


@dataclass(frozen=True)
class SuccessPositionView(ColoredSpatialView):
    """The current success targets for one environment instance."""

    @property
    def positions_w(self) -> torch.Tensor:
        return self.values_w

    def positions_for_color(self, color: Colors) -> torch.Tensor:
        return self.values_for_color(color)


class SuccessPositionBuffer(ColoredSpatialBuffer):
    """Fixed-capacity environment-major ARC success-target buffer."""

    def __init__(self, num_envs: int, max_targets: int, device: torch.device | str):
        super().__init__(num_envs=num_envs, max_items=max_targets, item_shape=(3,), device=device)

    @property
    def positions_w(self) -> torch.Tensor:
        return self.values_w

    @property
    def max_targets(self) -> int:
        return self.max_items

    @property
    def active_targets(self) -> torch.Tensor:
        return self.active_items

    @property
    def has_targets(self) -> torch.Tensor:
        return self.counts > 0

    def __getitem__(self, env_id: int) -> SuccessPositionView:
        self._validate_env_id(env_id)
        count = int(self.counts[env_id].item())
        return SuccessPositionView(self.positions_w[env_id, :count], self.color_ids[env_id, :count])

    def color_mask(self, color: Colors) -> torch.Tensor:
        return self.active_targets & (self.color_ids == _COLOR_TO_ARC_ID[color])

    def replace(self, env_id: int, positions_w: torch.Tensor, color_ids: torch.Tensor) -> None:
        self._replace(env_id, positions_w, color_ids)


@dataclass(frozen=True)
class ReserveCubeStackView(ColoredSpatialView):
    """Fixed reset-time poses and ARC colors for one environment's physical stacks."""

    @property
    def poses_w(self) -> torch.Tensor:
        return self.values_w

    def stack_index_for_color_id(self, arc_color_id: int) -> int:
        matches = torch.nonzero(self.color_ids == arc_color_id, as_tuple=False).flatten()
        if matches.numel() != 1:
            raise KeyError(f"ARC color ID {arc_color_id} does not identify exactly one reserve stack.")
        return int(matches.item())


class ReserveCubeStackBuffer(ColoredSpatialBuffer):
    """Environment-major reserve-stack snapshots with a color-to-stack index."""

    def __init__(self, num_envs: int, num_stacks: int, cubes_per_stack: int, device: torch.device | str):
        super().__init__(num_envs, num_stacks, (cubes_per_stack, 7), device)
        self.color_to_stack = torch.full((num_envs, _NUM_ARC_COLORS), -1, dtype=torch.long, device=device)

    @property
    def num_stacks(self) -> int:
        return self.max_items

    @property
    def cubes_per_stack(self) -> int:
        return self.values_w.shape[2]

    @property
    def poses_w(self) -> torch.Tensor:
        return self.values_w

    def __getitem__(self, env_id: int) -> ReserveCubeStackView:
        self._validate_env_id(env_id)
        return ReserveCubeStackView(self.poses_w[env_id], self.color_ids[env_id])

    def replace(self, env_id: int, poses_w: torch.Tensor, color_ids: torch.Tensor) -> None:
        if poses_w.shape[0] != self.num_stacks or color_ids.shape != (self.num_stacks,):
            raise ValueError(f"Reserve stacks require exactly {self.num_stacks} poses and color IDs.")
        if len(torch.unique(color_ids)) != self.num_stacks:
            raise ValueError("Every physical reserve stack must have a distinct ARC color ID.")
        self._replace(env_id, poses_w, color_ids)
        self.color_to_stack[env_id].fill_(-1)
        self.color_to_stack[env_id, color_ids.to(device=self.color_to_stack.device)] = torch.arange(
            self.num_stacks, device=self.color_to_stack.device
        )

    def clear(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        super().clear(env_ids)
        if env_ids is None:
            self.color_to_stack.fill_(-1)
        else:
            self.color_to_stack[torch.as_tensor(env_ids, device=self.color_to_stack.device, dtype=torch.long)] = -1
