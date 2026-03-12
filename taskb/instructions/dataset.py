"""Test instruction dataset: 30-50 instructions across categories A-F."""

INSTRUCTIONS = [
    # ── A: Direct Placement ──────────────────────────────────────────────────
    {
        "instruction": "Put the red block in the top right corner.",
        "category": "A",
        "difficulty": "easy",
    },
    {
        "instruction": "Move the blue block to the bottom left corner.",
        "category": "A",
        "difficulty": "easy",
    },
    {
        "instruction": "Place the green block along the left side of the workspace.",
        "category": "A",
        "difficulty": "easy",
    },
    {
        "instruction": "Move the yellow block to the top side of the workspace.",
        "category": "A",
        "difficulty": "easy",
    },
    {
        "instruction": "Put the blue block in the exact center of the workspace.",
        "category": "A",
        "difficulty": "easy",
    },
    {
        "instruction": "Move the red block 15cm to the right of the blue block.",
        "category": "A",
        "difficulty": "medium",
    },
    {
        "instruction": "Place the green block 20cm forward from its current position.",
        "category": "A",
        "difficulty": "medium",
    },
    {
        "instruction": "Move the yellow block halfway between the red and green blocks.",
        "category": "A",
        "difficulty": "medium",
    },
    {
        "instruction": "Put the blue block 10cm to the left and 10cm forward of the yellow block.",
        "category": "A",
        "difficulty": "hard",
    },
    {
        "instruction": "Move the red block to a position 5cm from the right edge of the workspace.",
        "category": "A",
        "difficulty": "hard",
    },

    # ── B: Stacking ──────────────────────────────────────────────────────────
    {
        "instruction": "Stack the green block on the red block.",
        "category": "B",
        "difficulty": "easy",
    },
    {
        "instruction": "Put the blue block on top of the yellow block.",
        "category": "B",
        "difficulty": "easy",
    },
    {
        "instruction": "Stack the small blue block on the large red block.",
        "category": "B",
        "difficulty": "easy",
    },
    {
        "instruction": "Build a tower with blue on bottom, then red, then green on top.",
        "category": "B",
        "difficulty": "medium",
    },
    {
        "instruction": "Stack all the small blocks on top of the largest block.",
        "category": "B",
        "difficulty": "medium",
    },
    {
        "instruction": "Stack all blocks into a single tower, largest on the bottom.",
        "category": "B",
        "difficulty": "hard",
    },
    {
        "instruction": "Make two separate stacks: red on blue, and yellow on green.",
        "category": "B",
        "difficulty": "hard",
    },

    # ── C: Spatial Arrangements ──────────────────────────────────────────────
    {
        "instruction": "Line up all the blocks along the right side.",
        "category": "C",
        "difficulty": "easy",
    },
    {
        "instruction": "Arrange all blocks in a row across the top of the workspace.",
        "category": "C",
        "difficulty": "easy",
    },
    {
        "instruction": "Spread all blocks evenly from the left side to the right side.",
        "category": "C",
        "difficulty": "easy",
    },
    {
        "instruction": "Arrange the red and blue blocks in a small circle in the center.",
        "category": "C",
        "difficulty": "medium",
    },
    {
        "instruction": "Place all four blocks in a circle with a radius of 20cm around the center.",
        "category": "C",
        "difficulty": "medium",
    },
    {
        "instruction": "Line up all blocks diagonally from the top left to the bottom right corner.",
        "category": "C",
        "difficulty": "hard",
    },

    # ── D: Conditional / Relational ──────────────────────────────────────────
    {
        "instruction": "If the red block is to the left of the blue block, swap them.",
        "category": "D",
        "difficulty": "easy",
    },
    {
        "instruction": "Move whichever block is closest to the center to the top left corner.",
        "category": "D",
        "difficulty": "easy",
    },
    {
        "instruction": "Move the block furthest from the center to the bottom right corner.",
        "category": "D",
        "difficulty": "medium",
    },
    {
        "instruction": "If any block is in the bottom half of the workspace, move it to the top.",
        "category": "D",
        "difficulty": "medium",
    },
    {
        "instruction": "Move the two blocks that are closest to each other so they are 20cm apart.",
        "category": "D",
        "difficulty": "hard",
    },
    {
        "instruction": "Swap the positions of the leftmost and rightmost blocks.",
        "category": "D",
        "difficulty": "hard",
    },

    # ── E: Composite Multi-Step ───────────────────────────────────────────────
    {
        "instruction": "Put all the large blocks on the left side, then stack the small blocks on the right.",
        "category": "E",
        "difficulty": "easy",
    },
    {
        "instruction": "Sort the blocks by color from left to right: red, green, blue, yellow.",
        "category": "E",
        "difficulty": "easy",
    },
    {
        "instruction": "First move all blocks to the top half of the workspace, then line them up.",
        "category": "E",
        "difficulty": "medium",
    },
    {
        "instruction": "Group blocks by size: large blocks form a stack on the left, small blocks in a row on the right.",
        "category": "E",
        "difficulty": "medium",
    },
    {
        "instruction": "Move the red block to the center, then stack all other blocks on top of it in order of size, smallest last.",
        "category": "E",
        "difficulty": "hard",
    },
    {
        "instruction": "Rearrange all blocks: place them in a circle, then stack each block on the one clockwise from it.",
        "category": "E",
        "difficulty": "hard",
    },

    # ── F: Rejection ─────────────────────────────────────────────────────────
    {
        "instruction": "Cut the block in half.",
        "category": "F",
        "difficulty": "easy",
    },
    {
        "instruction": "Paint the red block blue.",
        "category": "F",
        "difficulty": "easy",
    },
    {
        "instruction": "Spin the blue block 90 degrees.",
        "category": "F",
        "difficulty": "easy",
    },
    {
        "instruction": "Throw the yellow block off the table.",
        "category": "F",
        "difficulty": "medium",
    },
    {
        "instruction": "Make the green block float in mid-air.",
        "category": "F",
        "difficulty": "medium",
    },
    {
        "instruction": "Teleport the red block to the corner.",
        "category": "F",
        "difficulty": "hard",
    },
]
