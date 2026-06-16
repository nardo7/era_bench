import torch


def intersect_line_rectangle(
    line_points: torch.Tensor, line_dirs: torch.Tensor, rect_centers: torch.Tensor, rect_sizes: torch.Tensor
) -> torch.Tensor:
    """
    Computes the intersection point of a line and a rectangle in 2D. It works for env batches.
    """

    x_min = rect_centers[..., 0] - rect_sizes[..., 0] / 2
    x_max = rect_centers[..., 0] + rect_sizes[..., 0] / 2
    y_min = rect_centers[..., 1] - rect_sizes[..., 1] / 2
    y_max = rect_centers[..., 1] + rect_sizes[..., 1] / 2

    polygon = torch.stack(
        (
            torch.stack((x_min, x_max), dim=-1),
            torch.stack((y_min, y_max), dim=-1),
        ),
        dim=-2,
    )

    # polygon = pxolygon.permute(2, 0, 1)  # (B, 2, 2) where B is the batch size

    # repeat coordinates to match the shape of polygin i.e. duplicate x and y in a new dim at the end
    line_points = line_points.unsqueeze(-1).expand(-1, -1, 2)  # (B, 2, 2)
    line_dirs = line_dirs.unsqueeze(-1).expand(-1, -1, 2)  # (B, 2, 2)

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
