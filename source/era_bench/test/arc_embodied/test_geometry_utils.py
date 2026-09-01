import sys
from pathlib import Path

# Add the common task directory to sys.path so geometry_utils can be imported directly
# without triggering the era_bench package __init__.py chain.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "era_bench" / "tasks" / "manager_based" / "common"),
)

import torch
from geometry_utils import _sample_ts, mirror_points_across_line


class TestSampleTs:
    def test_enforces_minimum_separation_between_sorted_samples(self):
        """Sorted t samples should stay at least min_separation apart within each env."""
        torch.manual_seed(0)
        min_separation = 0.1
        t_boundaries = torch.tensor([[0.0, 0.5], [1.0, 1.5]], dtype=torch.float32)

        ts = _sample_ts(
            env_ids=[0, 1],
            t_boundaries=t_boundaries,
            number_of_objects=5,
            min_separation=min_separation,
        )

        sorted_ts = torch.sort(ts, dim=-1).values
        gaps = torch.diff(sorted_ts, dim=-1)

        assert ts.shape == (2, 5)
        assert (gaps >= min_separation - 1e-6).all()


class TestMirrorPointsAcrossLine:
    """Test suite for mirror_points_across_line function."""

    def test_point_on_line_unchanged(self):
        """A point on the line should remain unchanged after mirroring."""
        # Single batch, single point on the line y=0
        points = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)  # (B=1, N=1, 2) - on the line
        line_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # (B=1, 2)
        line_dir = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # (B=1, 2) horizontal line at y=0

        mirrored = mirror_points_across_line(points, line_pos, line_dir)

        assert mirrored.shape == (1, 1, 2)
        assert torch.allclose(mirrored, points, atol=1e-5)

    def test_perpendicular_mirror_symmetric(self):
        """Mirror a point directly perpendicular to the line."""
        # Line is horizontal at y=0, point at (0, 1), should mirror to (0, -1)
        points = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)  # (B=1, N=1, 2)
        line_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        line_dir = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        mirrored = mirror_points_across_line(points, line_pos, line_dir)
        expected = torch.tensor([[[0.0, -1.0]]], dtype=torch.float32)

        assert torch.allclose(mirrored, expected, atol=1e-5)

    def test_diagonal_line_mirror(self):
        """Mirror across a diagonal line (45-degree slope)."""
        # Line along y=x from origin, point at (1, 0), should mirror to (0, 1)
        points = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        line_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        line_dir = torch.tensor([[1.0, 1.0]]) / torch.sqrt(torch.tensor(2.0))  # normalized direction

        mirrored = mirror_points_across_line(points, line_pos, line_dir)
        expected = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)

        assert torch.allclose(mirrored, expected, atol=1e-5)

    def test_multiple_points_same_line(self):
        """Mirror multiple points across the same line."""
        points = torch.tensor([[[0.0, 1.0], [1.0, 2.0], [-1.0, 1.0]]], dtype=torch.float32)  # (B=1, N=3, 2)
        line_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        line_dir = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # horizontal line at y=0

        mirrored = mirror_points_across_line(points, line_pos, line_dir)
        expected = torch.tensor([[[0.0, -1.0], [1.0, -2.0], [-1.0, -1.0]]], dtype=torch.float32)

        assert mirrored.shape == (1, 3, 2)
        assert torch.allclose(mirrored, expected, atol=1e-5)

    def test_batch_different_lines(self):
        """Test batched operation with different lines per environment."""
        # Batch of 2 environments, 2 points each
        points = torch.tensor(
            [[[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32
        )  # (B=2, N=2, 2)

        # First env: horizontal line at y=0
        # Second env: vertical line at x=0
        line_pos = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)  # (B=2, 2)
        line_dir = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)  # (B=2, 2)

        mirrored = mirror_points_across_line(points, line_pos, line_dir)

        # First env: (0,1)→(0,-1), (1,1)→(1,-1)
        # Second env: (0,1)→(0,1), (1,1)→(-1,1)
        expected = torch.tensor([[[0.0, -1.0], [1.0, -1.0]], [[0.0, 1.0], [-1.0, 1.0]]], dtype=torch.float32)

        assert mirrored.shape == (2, 2, 2)
        assert torch.allclose(mirrored, expected, atol=1e-5)

    def test_mirror_is_involution(self):
        """Mirroring twice should give back the original point."""
        points = torch.tensor([[[2.0, 3.0]]], dtype=torch.float32)
        line_pos = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        line_dir = torch.tensor([[1.0, 1.0]]) / torch.sqrt(torch.tensor(2.0))

        mirrored_once = mirror_points_across_line(points, line_pos, line_dir)
        mirrored_twice = mirror_points_across_line(mirrored_once, line_pos, line_dir)

        assert torch.allclose(mirrored_twice, points, atol=1e-5)

    def test_equidistant_from_line(self):
        """Original and mirrored points should be equidistant from the line."""
        points = torch.tensor([[[2.0, 3.0]]], dtype=torch.float32)  # (B=1, N=1, 2)
        line_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        line_dir = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # horizontal line at y=0

        mirrored = mirror_points_across_line(points, line_pos, line_dir)

        # Distance from point to line (y=0) is |y| = 3
        # Distance from mirrored point should also be 3
        point_dist = torch.abs(points[..., 1] - line_pos[:, 1:2].unsqueeze(1))
        mirrored_dist = torch.abs(mirrored[..., 1] - line_pos[:, 1:2].unsqueeze(1))

        assert torch.allclose(point_dist, mirrored_dist, atol=1e-5)

    def test_mirrored_points_on_opposite_side(self):
        """Mirrored points should be on the opposite side of the line."""
        points = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]], dtype=torch.float32)  # (B=1, N=2, 2)
        line_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        line_dir = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # horizontal line

        mirrored = mirror_points_across_line(points, line_pos, line_dir)

        # Original: (0,1) has y>0, (0,-1) has y<0
        # Mirrored: should have opposite signs
        assert torch.sign(points[0, 0, 1]) != torch.sign(mirrored[0, 0, 1])
        assert torch.sign(points[0, 1, 1]) != torch.sign(mirrored[0, 1, 1])

    def test_line_offset_from_origin(self):
        """Test mirroring with a line not passing through the origin."""
        # Line at y=2 (horizontal), point at (0, 3), should mirror to (0, 1)
        points = torch.tensor([[[0.0, 3.0]]], dtype=torch.float32)
        line_pos = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
        line_dir = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        mirrored = mirror_points_across_line(points, line_pos, line_dir)
        expected = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)

        assert torch.allclose(mirrored, expected, atol=1e-5)
