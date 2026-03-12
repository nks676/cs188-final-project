"""Mock implementations of Task A functions for dev/testing."""
import copy
import logging

import numpy as np

logger = logging.getLogger("taskb.stubs")

STUB_SCENE = [
    {"id": 0, "color": "red",    "size": "large",  "pos": [0.10,  0.20,  0.82], "height": 0.050},
    {"id": 1, "color": "blue",   "size": "small",  "pos": [0.30,  0.10,  0.82], "height": 0.040},
    {"id": 2, "color": "green",  "size": "large",  "pos": [-0.10, 0.15,  0.82], "height": 0.050},
    {"id": 3, "color": "yellow", "size": "small",  "pos": [0.20, -0.10,  0.82], "height": 0.040},
]

# Mutable scene state so pick_and_place actually updates positions
_scene: list[dict] = copy.deepcopy(STUB_SCENE)

# Workspace bounds: [x_min, y_min, z_table], [x_max, y_max, z_max]
_WS_MIN = np.array([-0.50, -0.40, 0.82])
_WS_MAX = np.array([ 0.50,  0.40, 1.50])


def reset_scene() -> None:
    """Reset stub scene to initial state (useful before each eval episode)."""
    global _scene
    _scene = copy.deepcopy(STUB_SCENE)


def get_scene_state() -> list[dict]:
    """Return list of block dicts: {id, color, size, pos, height}."""
    result = copy.deepcopy(_scene)
    logger.debug("get_scene_state() -> %s", result)
    return result


def get_workspace_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return (ws_min, ws_max) as numpy arrays."""
    logger.debug("get_workspace_bounds() -> %s, %s", _WS_MIN, _WS_MAX)
    return _WS_MIN.copy(), _WS_MAX.copy()


def pick_and_place(source_id: int, target) -> bool:
    """
    Move source block to target.

    target can be:
      - list/array of length 3: absolute [x, y, z] position
      - int: target block id (stack on top of that block)

    Returns True on success.
    """
    global _scene

    src = next((b for b in _scene if b["id"] == source_id), None)
    if src is None:
        logger.warning("pick_and_place: source_id %s not found", source_id)
        return False

    if isinstance(target, (int, np.integer)):
        tgt = next((b for b in _scene if b["id"] == int(target)), None)
        if tgt is None:
            logger.warning("pick_and_place: target_id %s not found", target)
            return False
        new_z = tgt["pos"][2] + tgt["height"] + src["height"] / 2
        new_pos = [tgt["pos"][0], tgt["pos"][1], new_z]
        logger.info("pick_and_place(%s, id=%s) -> stack at %s", source_id, int(target), new_pos)
    else:
        new_pos = list(target)
        logger.info("pick_and_place(%s, pos=%s)", source_id, new_pos)

    src["pos"] = new_pos
    return True
