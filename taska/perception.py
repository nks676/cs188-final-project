"""
Perception functions that read state from the live robosuite simulator.

These are the two Task-A perception APIs exposed to Task B via MCP:
  - get_scene_state(env)  -> list[dict]
  - get_workspace_bounds(env) -> tuple[np.ndarray, np.ndarray]
"""

import numpy as np

from taska.config import SIZE_MAP, TABLE_FULL_SIZE, TABLE_OFFSET

# Inset margin (meters) so computed bounds keep blocks safely on the table.
_TABLE_MARGIN = 0.05


def get_scene_state(env):
    """
    Return the complete environment state: one dict per block.

    Each dict contains:
        "id"     : int          — stable block identifier (0–5)
        "color"  : str          — e.g. "red", "blue"
        "size"   : str          — "small" | "large"
        "pos"    : list[float]  — [x, y, z] current 3-D position
        "height" : float        — full z-extent of the block (2 × half-extent)

    Args:
        env: A BlockManipulationEnv instance (must be reset before calling).

    Returns:
        list[dict]: One entry per block in the scene.
    """
    state = []
    for meta in env.block_meta:
        body_id = env.block_body_ids[meta["id"]]
        pos = env.sim.data.body_xpos[body_id].copy()

        half_z = SIZE_MAP[meta["size"]][2]
        height = half_z * 2.0

        state.append(
            {
                "id": meta["id"],
                "color": meta["color"],
                "size": meta["size"],
                "pos": pos.tolist(),
                "height": height,
            }
        )
    return state


def get_workspace_bounds(env=None):
    """
    Return (lower_corner, upper_corner) of the usable table surface.

    A small margin is applied so that targets near the edge won't cause
    blocks to fall off.  The z component is set to the table surface height.

    Args:
        env: Optional. Not used currently (bounds derived from config constants),
             but accepted for API consistency.

    Returns:
        tuple[np.ndarray, np.ndarray]: (lower_corner, upper_corner), each shape (3,).
    """
    half_x = TABLE_FULL_SIZE[0] / 2.0 - _TABLE_MARGIN
    half_y = TABLE_FULL_SIZE[1] / 2.0 - _TABLE_MARGIN
    table_z = TABLE_OFFSET[2]

    lower = np.array([TABLE_OFFSET[0] - half_x, TABLE_OFFSET[1] - half_y, table_z])
    upper = np.array([TABLE_OFFSET[0] + half_x, TABLE_OFFSET[1] + half_y, table_z])

    return lower, upper
