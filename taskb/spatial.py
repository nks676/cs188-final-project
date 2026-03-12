"""Pure-math spatial helper functions for tabletop robot tasks."""
import numpy as np

# Lazy import to avoid circular dependency at module load time
_get_workspace_bounds = None
CORNER_INSET_RATIO = 0.20


def _ws_bounds():
    global _get_workspace_bounds
    if _get_workspace_bounds is None:
        try:
            from taska.api import get_workspace_bounds
        except ImportError:
            from taskb.stubs import get_workspace_bounds
        _get_workspace_bounds = get_workspace_bounds
    return _get_workspace_bounds()


def get_corner_pos(corner: str) -> np.ndarray:
    """
    Return the [x, y, z] position of a workspace corner.

    corner: "top left" | "top right" | "bottom left" | "bottom right"
    z is the table surface height (ws_min[2]).

    Uses a 10% inward inset from workspace edges so targets are reachable and
    less likely to fail near the table boundary.

    Perspective convention:
    - Commands are interpreted in viewer terms, then mapped into the opposite
      robot frame orientation used by the environment.
    - This means:
      - "top" maps to higher y in workspace coordinates
      - "bottom" maps to lower y in workspace coordinates
      - "left" maps to higher x in workspace coordinates
      - "right" maps to lower x in workspace coordinates
    """
    ws_min, ws_max = _ws_bounds()
    z = ws_min[2]
    x_span = ws_max[0] - ws_min[0]
    y_span = ws_max[1] - ws_min[1]
    x_left = ws_max[0] - CORNER_INSET_RATIO * x_span
    x_right = ws_min[0] + CORNER_INSET_RATIO * x_span
    y_top = ws_max[1] - CORNER_INSET_RATIO * y_span
    y_bottom = ws_min[1] + CORNER_INSET_RATIO * y_span
    if corner == "top left":
        return np.array([x_left, y_top, z])
    elif corner == "top right":
        return np.array([x_right, y_top, z])
    elif corner == "bottom left":
        return np.array([x_left, y_bottom, z])
    elif corner == "bottom right":
        return np.array([x_right, y_bottom, z])
    else:
        raise ValueError(f"Unknown corner: {corner!r}. Use 'top left', 'top right', 'bottom left', 'bottom right'.")


def get_side_pos(side: str) -> np.ndarray:
    """
    Return the midpoint position along a workspace side.

    side: "left" | "right" | "top" | "bottom"
    z is the table surface height.

    "top" / "bottom" and "left" / "right" are mapped into the opposite robot
    frame orientation used by the environment.
    """
    ws_min, ws_max = _ws_bounds()
    z = ws_min[2]
    cx = (ws_min[0] + ws_max[0]) / 2
    cy = (ws_min[1] + ws_max[1]) / 2
    if side == "left":
        return np.array([ws_max[0], cy, z])
    elif side == "right":
        return np.array([ws_min[0], cy, z])
    elif side == "top":
        return np.array([cx, ws_max[1], z])
    elif side == "bottom":
        return np.array([cx, ws_min[1], z])
    else:
        raise ValueError(f"Unknown side: {side!r}. Use 'left', 'right', 'top', 'bottom'.")


def get_midpoint(pos1, pos2) -> np.ndarray:
    """
    Return the component-wise midpoint between two positions.

    pos1, pos2: array-like of length 3.
    """
    return (np.array(pos1) + np.array(pos2)) / 2.0


def get_point_offset(pos, direction: str, dist_cm: float) -> np.ndarray:
    """
    Return pos shifted by dist_cm centimetres in the given direction.

    direction: "left" | "right" | "up" | "down" | "forward" | "backward"
      - left/right: ±x axis
      - forward/backward: ±y axis
      - up/down: ±z axis

    dist_cm: distance in centimetres (converted to metres internally).
    """
    dist_m = dist_cm / 100.0
    p = np.array(pos, dtype=float)
    deltas = {
        "left":     np.array([-dist_m, 0.0, 0.0]),
        "right":    np.array([ dist_m, 0.0, 0.0]),
        "forward":  np.array([0.0,  dist_m, 0.0]),
        "backward": np.array([0.0, -dist_m, 0.0]),
        "up":       np.array([0.0, 0.0,  dist_m]),
        "down":     np.array([0.0, 0.0, -dist_m]),
    }
    if direction not in deltas:
        raise ValueError(f"Unknown direction: {direction!r}.")
    return p + deltas[direction]


def make_line_positions(start, end, n: int) -> list[np.ndarray]:
    """
    Return n evenly-spaced positions from start to end (inclusive).

    start, end: array-like of length 3.
    n: number of positions (n >= 2).
    """
    s = np.array(start, dtype=float)
    e = np.array(end, dtype=float)
    if n == 1:
        return [s.copy()]
    return [s + (e - s) * i / (n - 1) for i in range(n)]


def make_circle_positions(center, radius_cm: float, n: int) -> list[np.ndarray]:
    """
    Return n evenly-spaced positions on a circle around center.

    center: array-like of length 3 (z is preserved for all points).
    radius_cm: radius in centimetres.
    n: number of positions.
    """
    c = np.array(center, dtype=float)
    r = radius_cm / 100.0
    angles = [2 * np.pi * i / n for i in range(n)]
    return [np.array([c[0] + r * np.cos(a), c[1] + r * np.sin(a), c[2]]) for a in angles]
