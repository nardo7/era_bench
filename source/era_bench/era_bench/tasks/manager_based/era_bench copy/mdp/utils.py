import torch


def line_direction_xy_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Returns the 2D unit direction of the line from its world quaternion."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]

    # Yaw from quaternion (w, x, y, z): line direction in XY plane.
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    direction_xy = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
    return torch.nn.functional.normalize(direction_xy, dim=1)


def line_direction_xy_from_quat_unnormalized(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Returns the 2D unit direction of the line from its world quaternion."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]

    # Yaw from quaternion (w, x, y, z): line direction in XY plane.
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    direction_xy = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
    return direction_xy
