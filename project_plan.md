# Project Plan: Code-as-Policies for Robosuite Block Manipulation

## Current Task B Status

Task B is implemented as a standalone Python pipeline under `taskb/`. The current code path:

- builds a prompt with a fixed API reference and few-shot examples
- calls Gemini directly via the `google-genai` SDK
- validates generated Python with the sandbox in `taskb/sandbox.py`
- executes against either `taska.env` or deterministic stubs in `taskb/stubs.py`
- evaluates outcomes and logs each episode to `logs/episodes.jsonl`

This means Task B currently uses a direct SDK call plus a local execution sandbox, not MCP tool wiring.

## 1) Goal

Build a system where an LLM receives natural-language instructions and **generates executable Python code** (not just JSON action labels) that composes low-level robot primitives to manipulate blocks in a robosuite tabletop environment. Inspired by [Code as Policies](https://code-as-policies.github.io), the LLM acts as a policy writer — it outputs Python snippets that call perception utilities, compute spatial targets, and sequence gripper actions.

---

## 2) Core Idea: LLM Generates Code, Not Just Labels

### Current plan (too simple)
The LLM outputs a JSON blob choosing between `move_block` and `stack_block`. This is basically a classifier — there is no reasoning, no spatial computation, no composition.

### New plan (CaP-inspired)
The LLM is prompted with:
- A library of **low-level primitive APIs** (perception + control), exposed in the Python execution sandbox.
- A set of **few-shot examples** showing how natural-language commands map to Python code that calls those APIs.

The LLM receives **only the user instruction** — no scene state is injected into the prompt. Instead, the generated code must **discover the scene** by calling MCP perception tools (e.g., `get_obj_names()`, `get_obj_pos(...)`).

Given a new instruction, the LLM writes a short Python program that:
1. Queries the scene (e.g., `get_obj_pos('red block')`).
2. Computes spatial targets (e.g., midpoints, offsets, corners).
3. Calls control primitives in sequence (e.g., `pick_and_place('red block', target_pos)`).
4. Uses standard Python (loops, conditionals, NumPy) for multi-step logic.

The generated code is then **executed** against the real environment APIs.

---

## 3) Primitive API Library

The LLM can call any of the following functions. These are the building blocks that the generated code composes.

### Perception Utilities
```python
get_scene_state() -> list[dict]
# Returns the complete environment state: one dict per block in the scene.
# Each dict contains:
#   "id"    : int        — unique block identifier (stable within an episode)
#   "color" : str        — e.g. "red", "blue", "green", "yellow", "purple", "orange"
#   "size"  : str        — "small" | "large"
#   "pos"   : list[float]  — [x, y, z] current 3D position
#   "height": float      — z-extent of the block (used internally by pick_and_place stacking)
#
# Example return value:
# [
#   {"id": 0, "color": "red",    "size": "large", "pos": [0.10, 0.20, 0.82], "height": 0.050},
#   {"id": 1, "color": "blue",   "size": "small", "pos": [0.30, 0.10, 0.82], "height": 0.040},
#   {"id": 2, "color": "green",  "size": "large", "pos": [-0.1, 0.15, 0.82], "height": 0.050},
# ]
#
# Call this once at the start of a program, then use the returned ids in pick_and_place.
# No separate get_obj_pos / get_obj_color / get_obj_height calls are needed.

get_workspace_bounds() -> tuple[np.ndarray, np.ndarray]
# Returns (lower_corner, upper_corner) of the workspace table surface.
```

`get_scene_state()` is exposed through the Task B execution environment. The LLM calls it once at the start of generated code to discover all block attributes and their IDs — no scene context is injected into the prompt.

### Control Primitives
```python
pick_and_place(source_id: int, target: np.ndarray | int) -> bool
# Picks up the block identified by source_id and either:
#   - Places it at an absolute position, if target is an np.ndarray [x, y, z].
#   - Stacks it on top of the block identified by target, if target is an int (block id).
#     In stacking mode the controller automatically computes the correct z-offset
#     (no manual height arithmetic needed in generated code).
#
# Internally runs: approach → descend → grasp → lift → move → descend → release → retreat.
# Returns True on success.
#
# Examples:
#   pick_and_place(0, np.array([0.2, 0.1, 0.82]))   # place block 0 at absolute pos
#   pick_and_place(2, 0)                              # stack block 2 on top of block 0

say(message: str)
# Speaks/logs a message to the user.
```

> **Design choice**: Stacking is just `pick_and_place(source_id, target_id)` — the primitive handles the height offset internally. Placement to an explicit position uses `pick_and_place(source_id, target_pos)`. The LLM distinguishes the two cases by type: pass an int to stack, pass an array to place.

### Spatial Helper Functions
```python
get_corner_pos(corner: str) -> np.ndarray
# corner in {"top left", "top right", "bottom left", "bottom right"}

get_side_pos(side: str) -> np.ndarray
# side in {"left", "right", "top", "bottom"}

get_midpoint(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray
# Returns the midpoint between two positions.

get_point_offset(pos: np.ndarray, direction: str, dist_cm: float) -> np.ndarray
# direction in {"left", "right", "up", "down", "forward", "backward"}
# Returns a new position offset from pos by dist_cm in the given direction.

make_line_positions(start: np.ndarray, end: np.ndarray, n: int) -> list[np.ndarray]
# Returns n evenly-spaced positions along the line from start to end.

make_circle_positions(center: np.ndarray, radius_cm: float, n: int) -> list[np.ndarray]
# Returns n positions arranged in a circle around center.
```

---

## 4) Supported Instructions (with example generated code)

Instructions should be **lower-level and spatially specific** — the user gives directions that require the LLM to reason about positions, distances, and spatial relationships.

### A) Direct Placement with Spatial Targets

Instructions that name a specific spatial destination (corner, side, relative position).

Example instructions:
- "Put the red block in the top right corner."
- "Move the blue block 10cm to the left of the green block."
- "Place the small yellow block between the red and blue blocks."

Example generated code:
```python
# "Put the red block in the top right corner."
say("Moving the red block to the top right corner")
blocks = get_scene_state()
red = next(b for b in blocks if b["color"] == "red")
corner = get_corner_pos("top right")
pick_and_place(red["id"], corner)
```

```python
# "Move the blue block 10cm to the left of the green block."
say("Moving the blue block to the left of the green block")
blocks = get_scene_state()
blue  = next(b for b in blocks if b["color"] == "blue")
green = next(b for b in blocks if b["color"] == "green")
target = get_point_offset(np.array(green["pos"]), "left", 10)
pick_and_place(blue["id"], target)
```

```python
# "Place the small yellow block between the red and blue blocks."
say("Placing the yellow block between the red and blue blocks")
blocks = get_scene_state()
yellow = next(b for b in blocks if b["color"] == "yellow" and b["size"] == "small")
red    = next(b for b in blocks if b["color"] == "red")
blue   = next(b for b in blocks if b["color"] == "blue")
mid = get_midpoint(np.array(red["pos"]), np.array(blue["pos"]))
pick_and_place(yellow["id"], mid)
```

### B) Stacking with Order Constraints

Instructions that specify stacking and sometimes ordering.

Example instructions:
- "Stack the green block on the red block."
- "Build a tower with blue on bottom, then red, then green on top."
- "Stack all the small blocks with the darkest color on the bottom."

Example generated code:
```python
# "Stack the green block on the red block."
say("Stacking the green block on the red block")
blocks = get_scene_state()
green = next(b for b in blocks if b["color"] == "green")
red   = next(b for b in blocks if b["color"] == "red")
pick_and_place(green["id"], red["id"])  # pass target id → stacking mode
```

```python
# "Build a tower with blue on bottom, then red, then green on top."
say("Building a tower: blue, red, green from bottom to top")
blocks = get_scene_state()
by_color = {b["color"]: b for b in blocks}
color_order = ["blue", "red", "green"]
for i in range(1, len(color_order)):
    bottom = by_color[color_order[i - 1]]
    top    = by_color[color_order[i]]
    pick_and_place(top["id"], bottom["id"])  # stack top onto bottom
```

### C) Spatial Arrangement Patterns

Instructions that ask the robot to arrange blocks into geometric patterns.

Example instructions:
- "Line up all the blocks along the right side."
- "Make a row of blocks across the top, sorted by size."
- "Arrange the red and blue blocks in a circle in the center."

Example generated code:
```python
# "Line up all the blocks along the right side."
say("Lining up all blocks on the right side")
blocks = get_scene_state()
right_top    = get_corner_pos("top right")
right_bottom = get_corner_pos("bottom right")
positions = make_line_positions(right_top, right_bottom, len(blocks))
for block, pos in zip(blocks, positions):
    pick_and_place(block["id"], pos)
```

```python
# "Arrange the red and blue blocks in a circle in the center."
say("Arranging red and blue blocks in a circle")
blocks = get_scene_state()
targets = [b for b in blocks if b["color"] in ("red", "blue")]
ws_lo, ws_hi = get_workspace_bounds()
center = (ws_lo + ws_hi) / 2
circle_pts = make_circle_positions(center, 8, len(targets))
for block, pos in zip(targets, circle_pts):
    pick_and_place(block["id"], pos)
```

### D) Conditional / Relational Instructions

Instructions that require checking the scene state and branching.

Example instructions:
- "If the red block is to the left of the blue block, swap them."
- "Move whichever block is closest to the center to the top left corner."
- "Stack the two blocks that are farthest apart."

Example generated code:
```python
# "If the red block is to the left of the blue block, swap them."
blocks   = get_scene_state()
by_color = {b["color"]: b for b in blocks}
red  = by_color["red"]
blue = by_color["blue"]
red_pos  = np.array(red["pos"])
blue_pos = np.array(blue["pos"])
if red_pos[0] < blue_pos[0]:
    say("Red is left of blue — swapping them")
    pick_and_place(red["id"],  blue_pos.copy())
    pick_and_place(blue["id"], red_pos.copy())
else:
    say("Red is already right of blue — no swap needed")
```

```python
# "Move whichever block is closest to the center to the top left corner."
blocks = get_scene_state()
ws_lo, ws_hi = get_workspace_bounds()
center  = (ws_lo + ws_hi) / 2
closest = min(blocks, key=lambda b: np.linalg.norm(np.array(b["pos"])[:2] - center[:2]))
say(f"The {closest['color']} block is closest to center — moving it to top left")
pick_and_place(closest["id"], get_corner_pos("top left"))
```

### E) Multi-Step Composite Instructions

Instructions that combine several sub-tasks.

Example instructions:
- "Put all the large blocks on the left side, then stack the small blocks on the right."
- "Move the red block next to the blue block, then stack the green block on top of both."
- "Sort all blocks by color from left to right: red, green, blue."

Example generated code:
```python
# "Sort all blocks by color from left to right: red, green, blue."
say("Sorting blocks left to right: red, green, blue")
blocks      = get_scene_state()
color_order = ["red", "green", "blue"]
left  = get_side_pos("left")
right = get_side_pos("right")
slots    = make_line_positions(left, right, len(color_order))
by_color = {b["color"]: b for b in blocks}
for color, pos in zip(color_order, slots):
    if color in by_color:
        pick_and_place(by_color[color]["id"], pos)
```

### F) Rejection / Unsupported

The LLM should still reject impossible or unsupported instructions:
- "Cut the block in half." → `say("I can only pick, place, and stack blocks — I can't cut them.")`
- "Paint the block red." → `say("I can't change block colors — I can only move them.")`

---

## 5) Language → Code → Control Pipeline

### Step 1: Prompt Construction

Build the LLM prompt with:
1. **System message**: "You are a Python code generator for a tabletop robot. Given a natural language instruction, write a short Python program using only the provided MCP API functions. You must call `get_scene_state()` at the start to discover the scene — no scene information is provided to you."
2. **API documentation**: All perception, control, and spatial helper function signatures and docstrings (the MCP tool schemas).
3. **Few-shot examples**: 8–12 examples covering categories A–F above, showing instruction → code pairs. Every example starts with `blocks = get_scene_state()` and selects blocks by attribute before calling `pick_and_place` with their IDs.
4. **User instruction**: The new command to execute.

**No scene state is injected.** The LLM must generate code that calls `get_scene_state()` once to discover all block attributes and IDs, then uses those IDs in `pick_and_place`. This is the key design choice — the LLM must reason about which blocks match the instruction and look up their IDs before acting.

### Step 2: Code Generation

The LLM returns a Python code string. Before execution:
- **Parse check**: The code must be valid Python (AST parse).
- **Safety check**: Only allowed function calls (whitelist). No `import os`, `exec`, `eval`, file I/O, network, etc.
- **Scope check**: All referenced function calls must be in the allowed whitelist.

### Step 3: Code Execution

Execute the sanitized code against the live robosuite environment. Each `pick_and_place` call runs the scripted low-level controller:
- approach → descend → grasp → lift → move → descend → release → retreat

Perception calls (`get_obj_names`, `get_obj_pos`, etc.) query the live simulator state via MCP.

### Step 4: Success Verification

After execution, check task-specific success criteria:
- Object final positions match intended targets (within tolerance).
- Stack stability (objects remain stacked for N timesteps).

---

## 6) Environment Design

- **Objects**: 4–6 blocks of different colors (`red`, `blue`, `green`, `yellow`, `purple`, `orange`) and two sizes (`small`, `large`).
- **Spawn**: Positions randomized every reset within the workspace bounds.
- **Workspace**: Bounded rectangular table surface.
- No fixed zones needed — spatial targets are computed dynamically by the generated code.

- One instruction is executed per reset.

---

## 7) Evaluation Plan

### Primary Metric
- **One-shot episode success rate**: Did the generated code achieve the intent of the instruction?

### Breakdown by Category
| Category | Example | Target |
|---|---|---|
| Direct placement | "Put X in corner Y" | ≥90% |
| Stacking | "Stack X on Y" | ≥85% |
| Spatial arrangements | "Line up blocks on right" | ≥75% |
| Conditional | "If X then do Y" | ≥70% |
| Composite | "Do A then B" | ≥70% |
| Rejection | "Cut the block" | ≥95% |

### Overall MVP Target
- ≥80% total success across all categories on randomized initial states.

### Secondary Metrics
- Code parse validity rate (LLM outputs valid Python).
- Code safety check pass rate (no disallowed calls).
- Per-primitive success rate (`pick_and_place`).
- Perception call accuracy (LLM correctly discovers and uses scene info).

---

## 8) Failure Handling

- **Unsupported instruction** → LLM generates `say(...)` rejection, no actions executed.
- **Code parse failure** → Log error, mark episode failed, do not execute.
- **Safety violation** → Log violation, mark episode failed, do not execute.
- **Reference to nonexistent object** → Caught at runtime when perception call returns error.
- **Primitive failure** (grasp miss, unstable placement) → Stop execution, log failure trace.

---

## 9) Logging (for report + demo)

For each episode:
1. Raw instruction.
2. LLM-generated Python code.
3. Safety/parse check result.
4. MCP tool call trace (perception queries + action calls with args and results).
5. Scene state snapshots (before and after execution).
6. Final outcome + failure reason (if any).

This provides the transparent **language → code → control** mapping that the project requires.

---

## 10) Work Split — Task A / Task B

### Task A: Robosuite Environment Functions

Everything that **directly touches the robosuite simulator**. This is the critical path — nothing works end-to-end until these are reliable.

#### Perception (read sim state)
- `get_scene_state()` — query sim for all blocks, returning a list of dicts with `id`, `color`, `size`, `pos`, and `height` for each block. Single call gives the LLM everything it needs to reason about the scene.
- `get_workspace_bounds()` — read table bounds from sim

#### Control (act in sim)
- `pick_and_place(source_id, target)` — the scripted motion controller (approach → descend → grasp → lift → move → descend → release → retreat). `target` is either an `np.ndarray` for absolute placement or an `int` block id for stacking (controller computes height offset automatically). This is the hardest part — requires controller tuning, grasp reliability, stacking stability.

#### Environment setup
- Robosuite env configuration: table, 4–6 blocks, randomized spawns, camera
- Block size/color definitions

---

### Task B: MCP Server + Code Pipeline + Everything Else

Everything that **does not require a working robosuite env** — can be developed and tested with stub/mock functions for Task A.

#### MCP Server
- Expose all Task A functions as MCP tools with proper schemas
- Expose spatial helpers and `say()` as MCP tools

#### Spatial Helper Functions (pure math, no sim)
- `get_corner_pos(corner)` — compute from workspace bounds
- `get_side_pos(side)` — compute from workspace bounds
- `get_midpoint(pos1, pos2)` — pure NumPy
- `get_point_offset(pos, direction, dist_cm)` — pure NumPy
- `make_line_positions(start, end, n)` — pure NumPy
- `make_circle_positions(center, radius, n)` — pure NumPy
- `say(message)` — log/print

#### LLM Prompting Pipeline
- System prompt + API docs section + few-shot examples (8–12 covering categories A–F)
- LLM API integration (OpenAI/Anthropic client)
- Code extraction from LLM response

#### Code Safety & Execution
- AST parse validation
- Whitelist-based safety checker (block `import os`, `exec`, `eval`, etc.)
- Execution sandbox (`exec()` with restricted globals)

#### Evaluation & Logging
- Success verification (position tolerance, stack stability)
- Evaluation harness (batch runner over instruction set)
- Per-episode logging (instruction, code, call trace, before/after state, outcome)
- Test instruction dataset (30–50 instructions across categories A–F)
- Metrics aggregation (success rates by category)

---

## 11) Milestones

1. Perception MCP tools + spatial helpers implemented and tested.
2. `pick_and_place` primitive working end-to-end (including stacking via height computation).
3. Prompt template + few-shot examples authored.
4. Code generation + safety checking pipeline working.
5. Single-instruction episodes running end-to-end.
6. Evaluation harness + report-ready metrics and traces.

---

## 12) What Changed vs. Previous Plan

| Aspect | Old Plan | New Plan |
|---|---|---|
| LLM output | JSON action label | Executable Python code |
| Scene info | Injected into prompt | LLM discovers via single `get_scene_state()` MCP call |
| Perception API | Multiple per-attribute calls (`get_obj_pos`, `get_obj_color`, …) | Single `get_scene_state()` returns all block info (id, color, size, pos, height) |
| Block addressing | String names (`"red block"`) | Integer IDs returned by `get_scene_state()` |
| Primitives | 2 high-level (`move_block`, `stack_block`) | Single `pick_and_place(source_id, target)` + `get_scene_state()` + spatial helpers |
| Stacking | Dedicated `stack_block` primitive | `pick_and_place(source_id, target_id)` — controller handles height offset automatically |
| Placement | N/A | `pick_and_place(source_id, target_pos)` — pass an `np.ndarray` |
| Spatial reasoning | None (fixed zones only) | LLM computes positions using NumPy + helper functions |
| Instruction types | Move to zone / stack in zone | Corners, offsets, midpoints, arrangements, conditionals, composites |
| Composability | Flat list of tool calls | LLM writes loops, conditionals, variable assignments |
| Destinations | Fixed `MOVE_ZONE` / `STACK_ZONE` | Any computed position on the workspace |
| Complexity | Classifier + planner | Code generation + execution (CaP-style) |
