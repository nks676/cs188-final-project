"""Unit tests for taskb/spatial.py."""
import math

import numpy as np
import pytest

from taskb.stubs import _WS_MIN, _WS_MAX
from taskb.spatial import (
    get_corner_pos,
    get_midpoint,
    get_point_offset,
    get_side_pos,
    make_circle_positions,
    make_line_positions,
)

# Workspace from stubs: x ∈ [-0.5, 0.5], y ∈ [-0.4, 0.4], z_table = 0.82
WS_MIN = _WS_MIN
WS_MAX = _WS_MAX
Z = WS_MIN[2]


# ── get_corner_pos ────────────────────────────────────────────────────────────

class TestGetCornerPos:
    def test_top_left(self):
        p = get_corner_pos("top left")
        assert p[0] == pytest.approx(WS_MIN[0])
        assert p[1] == pytest.approx(WS_MAX[1])
        assert p[2] == pytest.approx(Z)

    def test_top_right(self):
        p = get_corner_pos("top right")
        assert p[0] == pytest.approx(WS_MAX[0])
        assert p[1] == pytest.approx(WS_MAX[1])
        assert p[2] == pytest.approx(Z)

    def test_bottom_left(self):
        p = get_corner_pos("bottom left")
        assert p[0] == pytest.approx(WS_MIN[0])
        assert p[1] == pytest.approx(WS_MIN[1])
        assert p[2] == pytest.approx(Z)

    def test_bottom_right(self):
        p = get_corner_pos("bottom right")
        assert p[0] == pytest.approx(WS_MAX[0])
        assert p[1] == pytest.approx(WS_MIN[1])
        assert p[2] == pytest.approx(Z)

    def test_invalid(self):
        with pytest.raises(ValueError):
            get_corner_pos("center")

    def test_returns_ndarray(self):
        assert isinstance(get_corner_pos("top left"), np.ndarray)


# ── get_side_pos ──────────────────────────────────────────────────────────────

class TestGetSidePos:
    def test_left(self):
        p = get_side_pos("left")
        assert p[0] == pytest.approx(WS_MIN[0])
        assert p[1] == pytest.approx((WS_MIN[1] + WS_MAX[1]) / 2)
        assert p[2] == pytest.approx(Z)

    def test_right(self):
        p = get_side_pos("right")
        assert p[0] == pytest.approx(WS_MAX[0])

    def test_top(self):
        p = get_side_pos("top")
        assert p[1] == pytest.approx(WS_MAX[1])
        assert p[0] == pytest.approx((WS_MIN[0] + WS_MAX[0]) / 2)

    def test_bottom(self):
        p = get_side_pos("bottom")
        assert p[1] == pytest.approx(WS_MIN[1])

    def test_z_preserved(self):
        for side in ("left", "right", "top", "bottom"):
            assert get_side_pos(side)[2] == pytest.approx(Z)

    def test_invalid(self):
        with pytest.raises(ValueError):
            get_side_pos("front")


# ── get_midpoint ──────────────────────────────────────────────────────────────

class TestGetMidpoint:
    def test_basic(self):
        mp = get_midpoint([0, 0, 0], [2, 4, 6])
        np.testing.assert_allclose(mp, [1, 2, 3])

    def test_same_point(self):
        mp = get_midpoint([1, 2, 3], [1, 2, 3])
        np.testing.assert_allclose(mp, [1, 2, 3])

    def test_negative_coords(self):
        mp = get_midpoint([-1, -1, 0], [1, 1, 0])
        np.testing.assert_allclose(mp, [0, 0, 0])

    def test_returns_ndarray(self):
        assert isinstance(get_midpoint([0, 0, 0], [1, 1, 1]), np.ndarray)

    def test_workspace_midpoint(self):
        mp = get_midpoint(WS_MIN, WS_MAX)
        assert mp[0] == pytest.approx(0.0)  # (-0.5 + 0.5) / 2
        assert mp[1] == pytest.approx(0.0)  # (-0.4 + 0.4) / 2


# ── get_point_offset ──────────────────────────────────────────────────────────

class TestGetPointOffset:
    BASE = [0.1, 0.2, 0.82]

    def test_left(self):
        p = get_point_offset(self.BASE, "left", 10)
        np.testing.assert_allclose(p, [0.1 - 0.10, 0.2, 0.82])

    def test_right(self):
        p = get_point_offset(self.BASE, "right", 20)
        np.testing.assert_allclose(p, [0.1 + 0.20, 0.2, 0.82])

    def test_forward(self):
        p = get_point_offset(self.BASE, "forward", 5)
        np.testing.assert_allclose(p, [0.1, 0.2 + 0.05, 0.82])

    def test_backward(self):
        p = get_point_offset(self.BASE, "backward", 5)
        np.testing.assert_allclose(p, [0.1, 0.2 - 0.05, 0.82])

    def test_up(self):
        p = get_point_offset(self.BASE, "up", 10)
        np.testing.assert_allclose(p, [0.1, 0.2, 0.82 + 0.10])

    def test_down(self):
        p = get_point_offset(self.BASE, "down", 10)
        np.testing.assert_allclose(p, [0.1, 0.2, 0.82 - 0.10])

    def test_cm_to_metres_conversion(self):
        p = get_point_offset([0.0, 0.0, 0.0], "right", 100)
        np.testing.assert_allclose(p, [1.0, 0.0, 0.0])

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            get_point_offset(self.BASE, "diagonal", 5)

    def test_returns_ndarray(self):
        assert isinstance(get_point_offset(self.BASE, "left", 1), np.ndarray)


# ── make_line_positions ───────────────────────────────────────────────────────

class TestMakeLinePositions:
    def test_two_points(self):
        positions = make_line_positions([0, 0, 0], [1, 0, 0], 2)
        assert len(positions) == 2
        np.testing.assert_allclose(positions[0], [0, 0, 0])
        np.testing.assert_allclose(positions[1], [1, 0, 0])

    def test_three_points_midpoint(self):
        positions = make_line_positions([0, 0, 0], [2, 0, 0], 3)
        assert len(positions) == 3
        np.testing.assert_allclose(positions[1], [1, 0, 0])

    def test_five_points_evenly_spaced(self):
        positions = make_line_positions([0, 0, 0], [4, 0, 0], 5)
        xs = [p[0] for p in positions]
        assert xs == pytest.approx([0, 1, 2, 3, 4])

    def test_n_equals_one(self):
        positions = make_line_positions([1, 2, 3], [9, 9, 9], 1)
        assert len(positions) == 1
        np.testing.assert_allclose(positions[0], [1, 2, 3])

    def test_z_preserved(self):
        positions = make_line_positions([0, 0, 0.82], [1, 1, 0.82], 4)
        for p in positions:
            assert p[2] == pytest.approx(0.82)

    def test_returns_list_of_ndarrays(self):
        positions = make_line_positions([0, 0, 0], [1, 1, 1], 3)
        assert isinstance(positions, list)
        assert all(isinstance(p, np.ndarray) for p in positions)


# ── make_circle_positions ─────────────────────────────────────────────────────

class TestMakeCirclePositions:
    CENTER = [0.0, 0.0, 0.82]

    def test_count(self):
        positions = make_circle_positions(self.CENTER, 20, 4)
        assert len(positions) == 4

    def test_radius(self):
        positions = make_circle_positions(self.CENTER, 20, 8)  # radius = 0.20 m
        for p in positions:
            r = math.sqrt((p[0] - self.CENTER[0])**2 + (p[1] - self.CENTER[1])**2)
            assert r == pytest.approx(0.20, abs=1e-9)

    def test_z_preserved(self):
        positions = make_circle_positions(self.CENTER, 15, 6)
        for p in positions:
            assert p[2] == pytest.approx(0.82)

    def test_evenly_spaced_angles(self):
        positions = make_circle_positions(self.CENTER, 10, 4)
        angles = [math.atan2(p[1] - self.CENTER[1], p[0] - self.CENTER[0])
                  for p in positions]
        diffs = [(angles[i+1] - angles[i]) % (2*math.pi) for i in range(3)]
        assert all(d == pytest.approx(math.pi / 2) for d in diffs)

    def test_two_points_are_opposite(self):
        positions = make_circle_positions(self.CENTER, 10, 2)
        mid = (positions[0] + positions[1]) / 2
        np.testing.assert_allclose(mid[:2], [0.0, 0.0], atol=1e-9)

    def test_returns_list_of_ndarrays(self):
        positions = make_circle_positions(self.CENTER, 10, 3)
        assert isinstance(positions, list)
        assert all(isinstance(p, np.ndarray) for p in positions)
