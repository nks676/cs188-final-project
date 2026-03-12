"""System prompt, API reference, few-shot examples, and prompt builder."""

SYSTEM_MESSAGE = """\
You are a Python code generator for a tabletop robot. Given a natural-language instruction,
write a short Python program using only the provided API functions. You MUST call
get_scene_state() at the start to discover the scene — no scene information is given to you.
Return ONLY the code block with no explanation.\
"""

API_REFERENCE = """\
# === AVAILABLE API ===

# --- Task A: Environment ---

def get_scene_state() -> list[dict]:
    \"\"\"
    Return the current scene as a list of block dicts.
    Each dict has keys: id (int), color (str), size (str), pos (list[3] x/y/z), height (float).
    Example:
      [
        {"id": 0, "color": "red",   "size": "large", "pos": [0.10, 0.20, 0.82], "height": 0.050},
        {"id": 1, "color": "blue",  "size": "small", "pos": [0.30, 0.10, 0.82], "height": 0.040},
        {"id": 2, "color": "green", "size": "large", "pos": [-0.1, 0.15, 0.82], "height": 0.050},
        {"id": 3, "color": "yellow","size": "small", "pos": [0.20,-0.10, 0.82], "height": 0.040},
      ]
    \"\"\"

def get_workspace_bounds() -> tuple:
    \"\"\"Return (ws_min, ws_max) as numpy arrays of shape (3,): [x,y,z].\"\"\"

def pick_and_place(source_id: int, target) -> bool:
    \"\"\"
    Move block with source_id to target.
    target: list/array [x,y,z] for absolute placement, or int block id for stacking.
    Returns True on success.
    \"\"\"

# --- Task B: Spatial helpers ---

def get_corner_pos(corner: str) -> np.ndarray:
    \"\"\"corner: 'top left' | 'top right' | 'bottom left' | 'bottom right'. Returns [x,y,z].\"\"\"

def get_side_pos(side: str) -> np.ndarray:
    \"\"\"side: 'left' | 'right' | 'top' | 'bottom'. Returns midpoint of that workspace side.\"\"\"

def get_midpoint(pos1, pos2) -> np.ndarray:
    \"\"\"Return component-wise midpoint of two positions.\"\"\"

def get_point_offset(pos, direction: str, dist_cm: float) -> np.ndarray:
    \"\"\"
    Shift pos by dist_cm centimetres in direction.
    direction: 'left'|'right'|'up'|'down'|'forward'|'backward'
    \"\"\"

def make_line_positions(start, end, n: int) -> list:
    \"\"\"Return n evenly-spaced positions from start to end (inclusive).\"\"\"

def make_circle_positions(center, radius_cm: float, n: int) -> list:
    \"\"\"Return n evenly-spaced positions on a circle of radius_cm around center.\"\"\"

def say(msg: str) -> None:
    \"\"\"Print a message (for debugging/narration).\"\"\"
"""

_FEW_SHOT_EXAMPLES = [
    # --- A: Direct Placement ---
    (
        "Put the red block in the top right corner.",
        """\
scene = get_scene_state()
red = next(b for b in scene if b["color"] == "red")
corner = get_corner_pos("top right")
pick_and_place(red["id"], corner)
""",
    ),
    (
        "Move the blue block 10cm to the left of the green block.",
        """\
scene = get_scene_state()
blue = next(b for b in scene if b["color"] == "blue")
green = next(b for b in scene if b["color"] == "green")
target = get_point_offset(green["pos"], "left", 10)
pick_and_place(blue["id"], target)
""",
    ),
    # --- B: Stacking ---
    (
        "Stack the green block on the red block.",
        """\
scene = get_scene_state()
green = next(b for b in scene if b["color"] == "green")
red = next(b for b in scene if b["color"] == "red")
pick_and_place(green["id"], red["id"])
""",
    ),
    (
        "Build a tower with blue on bottom, then red, then green on top.",
        """\
scene = get_scene_state()
blue   = next(b for b in scene if b["color"] == "blue")
red    = next(b for b in scene if b["color"] == "red")
green  = next(b for b in scene if b["color"] == "green")
# Place blue first to anchor the tower
ws_min, ws_max = get_workspace_bounds()
center = get_midpoint(ws_min, ws_max)
pick_and_place(blue["id"], center)
pick_and_place(red["id"],   blue["id"])
pick_and_place(green["id"], red["id"])
""",
    ),
    # --- C: Spatial Arrangements ---
    (
        "Line up all the blocks along the right side.",
        """\
scene = get_scene_state()
n = len(scene)
ws_min, ws_max = get_workspace_bounds()
start = get_corner_pos("bottom right")
end   = get_corner_pos("top right")
positions = make_line_positions(start, end, n)
for block, pos in zip(scene, positions):
    pick_and_place(block["id"], pos)
""",
    ),
    (
        "Arrange the red and blue blocks in a circle in the center.",
        """\
scene = get_scene_state()
targets = [b for b in scene if b["color"] in ("red", "blue")]
ws_min, ws_max = get_workspace_bounds()
center = get_midpoint(ws_min, ws_max)
positions = make_circle_positions(center, 15, len(targets))
for block, pos in zip(targets, positions):
    pick_and_place(block["id"], pos)
""",
    ),
    # --- D: Conditional / Relational ---
    (
        "If the red block is to the left of the blue block, swap them.",
        """\
scene = get_scene_state()
red  = next(b for b in scene if b["color"] == "red")
blue = next(b for b in scene if b["color"] == "blue")
if red["pos"][0] < blue["pos"][0]:
    red_pos  = list(red["pos"])
    blue_pos = list(blue["pos"])
    pick_and_place(red["id"],  blue_pos)
    pick_and_place(blue["id"], red_pos)
""",
    ),
    (
        "Move whichever block is closest to the center to the top left corner.",
        """\
scene = get_scene_state()
ws_min, ws_max = get_workspace_bounds()
center = get_midpoint(ws_min, ws_max)
closest = min(scene, key=lambda b: abs(b["pos"][0] - center[0]) + abs(b["pos"][1] - center[1]))
target = get_corner_pos("top left")
pick_and_place(closest["id"], target)
""",
    ),
    # --- E: Composite Multi-Step ---
    (
        "Put all the large blocks on the left side, then stack the small blocks on the right.",
        """\
scene = get_scene_state()
large_blocks = [b for b in scene if b["size"] == "large"]
small_blocks = [b for b in scene if b["size"] == "small"]
left_start = get_corner_pos("bottom left")
left_end   = get_corner_pos("top left")
left_positions = make_line_positions(left_start, left_end, max(len(large_blocks), 1))
for block, pos in zip(large_blocks, left_positions):
    pick_and_place(block["id"], pos)
if small_blocks:
    right_center = get_side_pos("right")
    pick_and_place(small_blocks[0]["id"], right_center)
    for block in small_blocks[1:]:
        pick_and_place(block["id"], small_blocks[0]["id"])
""",
    ),
    (
        "Sort all blocks by color from left to right: red, green, blue.",
        """\
scene = get_scene_state()
color_order = ["red", "green", "blue"]
ordered = []
for color in color_order:
    match = next((b for b in scene if b["color"] == color), None)
    if match:
        ordered.append(match)
# Add remaining blocks not in the explicit order
remaining = [b for b in scene if b["color"] not in color_order]
ordered.extend(remaining)
ws_min, ws_max = get_workspace_bounds()
start = get_side_pos("left")
end   = get_side_pos("right")
positions = make_line_positions(start, end, len(ordered))
for block, pos in zip(ordered, positions):
    pick_and_place(block["id"], pos)
""",
    ),
    # --- F: Rejection ---
    (
        "Cut the block in half.",
        """\
say("I cannot cut blocks; the robot can only pick and place objects.")
""",
    ),
    (
        "Paint the block red.",
        """\
say("Painting is not a supported operation; the robot can only pick and place objects.")
""",
    ),
]


def build_prompt(instruction: str) -> str:
    """Assemble the full prompt for the LLM."""
    parts = [SYSTEM_MESSAGE, "\n\n", API_REFERENCE, "\n\n# === FEW-SHOT EXAMPLES ===\n\n"]
    for instr, code in _FEW_SHOT_EXAMPLES:
        parts.append(f"# INSTRUCTION: {instr}\n```python\n{code}```\n\n")
    parts.append(f"# INSTRUCTION: {instruction}\n")
    return "".join(parts)
