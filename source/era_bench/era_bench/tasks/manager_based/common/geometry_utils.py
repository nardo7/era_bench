import torch


def intersect_line_rectangle(
    line_points: torch.Tensor,
    line_dirs: torch.Tensor,
    rect_centers: torch.Tensor,
    rect_sizes: torch.Tensor,
    x_margin_offset: float = 0.5,
    y_margin_offset: float = 0.8,
) -> torch.Tensor:
    """
    Computes the intersection point of a line and a rectangle in 2D. It works for env batches.
    """
    x_margin = rect_sizes[..., 0] - x_margin_offset
    y_margin = rect_sizes[..., 1] - y_margin_offset
    x_min = rect_centers[..., 0] - x_margin / 2 + 0.3  # add offset to avoid to close to robot
    x_max = rect_centers[..., 0] + x_margin / 2 - 0.3  # add offset to avoid to close to robot
    y_min = rect_centers[..., 1] - y_margin / 2
    y_max = rect_centers[..., 1] + y_margin / 2

    polygon = torch.stack(
        (
            torch.stack((x_min, x_max), dim=-1),
            torch.stack((y_min, y_max), dim=-1),
        ),
        dim=-2,
    )

    # polygon = pxolygon.permute(2, 0, 1)  # (B, 2, 2) where B is the batch size

    # repeat coordinates to match the shape of polygin i.e. duplicate x and y in a new dim at the end
    line_points = line_points.unsqueeze(-1).expand(*line_points.shape, 2)  # (B, 2, 2)
    line_dirs = line_dirs.unsqueeze(-1).expand(*line_dirs.shape, 2)  # (B, 2, 2)

    t = polygon - line_points
    t = t / line_dirs

    t_mins = torch.min(t, dim=-1).values  # (B, 2)
    t_maxs = torch.max(t, dim=-1).values  # (B, 2)

    t_min = torch.max(t_mins, dim=-1).values  # (B,)
    t_max = torch.min(t_maxs, dim=-1).values  # (B,)

    return torch.stack((t_min, t_max), dim=-1)  # (B, 2)


def normal_vectors(line_dirs: torch.Tensor) -> torch.Tensor:
    """
    Computes the normal vectors to the line directions in 2D.
    """
    # For a direction vector (dx, dy), a normal vector can be (-dy, dx) or (dy, -dx)
    # We can choose one of them consistently, e.g., (-dy, dx)
    normal_x = -line_dirs[..., 1]  # -dy
    normal_y = line_dirs[..., 0]  # dx
    normals = torch.stack((normal_x, normal_y), dim=-1)  # (B, 2)
    return normals


def _sample_ts(
    env_ids: list[int],
    t_boundaries: torch.Tensor,
    number_of_objects: int,
    min_separation: float = 0.08,
):
    batch_size = len(env_ids)
    num_slots = torch.floor((t_boundaries[:, 1] - t_boundaries[:, 0]) / min_separation).long() + 1

    if (num_slots < number_of_objects).any():
        raise ValueError(
            f"Not enough space to place {number_of_objects} objects with min_separation={min_separation}. "
            f"Max available slots: {num_slots.max().item()}"
        )

    max_slots = int(num_slots.max().item())
    slot_range = torch.arange(max_slots, device=t_boundaries.device).unsqueeze(0)
    valid_slots = slot_range < num_slots.unsqueeze(1)

    rand = torch.rand((batch_size, max_slots), device=t_boundaries.device)
    rand = torch.where(valid_slots, rand, torch.full_like(rand, float("inf")))
    slot_indices = torch.argsort(rand, dim=1)[:, :number_of_objects]

    return t_boundaries[:, 0].unsqueeze(-1) + slot_indices.to(t_boundaries.dtype) * min_separation


def sample_points_on_half_plane(
    env_ids: list[int],
    line_pos_xy: torch.Tensor,
    line_dir_xy: torch.Tensor,
    table_pos: torch.Tensor,
    table_size_xy: torch.Tensor,
    number_of_objects: int,
    min_separation: float = 0.08,
) -> torch.Tensor:
    """
    Samples points on the half-plane defined by the line and the table boundaries.
    """

    n_directions = normal_vectors(line_dir_xy)

    # Restrict line samples to positions whose positive-normal half-plane still has
    # at least min_separation of clearance inside the table.
    offset_line_pos_xy = line_pos_xy + min_separation * n_directions
    t_boundaries = intersect_line_rectangle(offset_line_pos_xy, line_dir_xy, table_pos, table_size_xy)

    if (t_boundaries[:, 1] < t_boundaries[:, 0]).any():
        raise ValueError(
            "Not enough room to sample points with the requested min_separation from the line "
            "on the selected half-plane."
        )

    t = _sample_ts(env_ids, t_boundaries, number_of_objects, min_separation)  # (B, C)
    sampled_points = line_pos_xy.unsqueeze(1) + t.unsqueeze(-1) * line_dir_xy.unsqueeze(1)  # B, C, 2

    n_boundaries = intersect_line_rectangle(sampled_points.permute(1, 0, 2), n_directions, table_pos, table_size_xy)
    n_boundaries = n_boundaries.permute(1, 0, 2)
    n_boundaries[..., 0] = torch.clamp_min(n_boundaries[..., 0], min_separation)

    invalid_mask = n_boundaries[..., 1] < n_boundaries[..., 0]
    if invalid_mask.any():
        feasible_upper = torch.clamp_min(n_boundaries[..., 1], 0.0)
        n_boundaries[..., 0] = torch.where(invalid_mask, feasible_upper, n_boundaries[..., 0])
        n_boundaries[..., 1] = torch.where(invalid_mask, feasible_upper, n_boundaries[..., 1])

    n = (
        torch.rand((len(env_ids), number_of_objects), device=table_pos.device)
        * (n_boundaries[..., 1] - n_boundaries[..., 0])
        + n_boundaries[..., 0]
    )
    sampled_points = sampled_points + n.unsqueeze(-1) * n_directions.unsqueeze(1)

    return sampled_points  # B, C, 2


def mirror_points_across_line(
    points_xy: torch.Tensor, line_pos_xy: torch.Tensor, line_dir_xy: torch.Tensor
) -> torch.Tensor:
    """Mirrors points across a 2D line defined by a point and unit direction."""
    points_to_line = points_xy - line_pos_xy.unsqueeze(1)
    parallel_mag = torch.sum(points_to_line * line_dir_xy.unsqueeze(1), dim=-1, keepdim=True)
    proj_parallel = parallel_mag * line_dir_xy.unsqueeze(1)
    perpendicular = points_to_line - proj_parallel
    return line_pos_xy.unsqueeze(1) + proj_parallel - perpendicular
