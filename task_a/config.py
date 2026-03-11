"""Block definitions, sizes, colors, and environment constants."""

import numpy as np

# Fixed set of 6 blocks — colors never change, sizes are randomized per episode.
BLOCK_COLORS = [
    {"name": "red_block",    "color": "red",    "rgba": [1.0, 0.0, 0.0, 1.0]},
    {"name": "blue_block",   "color": "blue",   "rgba": [0.0, 0.0, 1.0, 1.0]},
    {"name": "green_block",  "color": "green",  "rgba": [0.0, 0.8, 0.0, 1.0]},
    {"name": "yellow_block", "color": "yellow", "rgba": [1.0, 1.0, 0.0, 1.0]},
    {"name": "purple_block", "color": "purple", "rgba": [0.5, 0.0, 0.5, 1.0]},
    {"name": "orange_block", "color": "orange", "rgba": [1.0, 0.5, 0.0, 1.0]},
]

# Half-extents in meters (MuJoCo box geoms use half-sizes).
SIZE_MAP = {
    "small": [0.020, 0.020, 0.020],   # 4cm cube
    "large": [0.025, 0.025, 0.025],   # 5cm cube
}

SIZE_CATEGORIES = ["small", "large"]

# Table surface dimensions and offset
TABLE_FULL_SIZE = (0.8, 0.8, 0.05)
TABLE_OFFSET = np.array([0.0, 0.0, 0.8])

# Block placement range (relative to table center) — wide enough for 6 blocks
PLACEMENT_X_RANGE = [-0.15, 0.15]
PLACEMENT_Y_RANGE = [-0.15, 0.15]
PLACEMENT_Z_OFFSET = 0.01
