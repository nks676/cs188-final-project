# Project Plan: Code-as-Policies for Robosuite Block Manipulation

## 1) Goal

Build a system where an LLM receives natural-language instructions and **generates executable Python code** (not just JSON action labels) that composes low-level robot primitives to manipulate blocks in a robosuite tabletop environment. Inspired by [Code as Policies](https://code-as-policies.github.io), the LLM acts as a policy writer — it outputs Python snippets that call perception utilities, compute spatial targets, and sequence gripper actions.

---

## 2) Core Idea: LLM Generates Code, Not Just Labels

### Current plan (too simple)
The LLM outputs a JSON blob choosing between `move_block` and `stack_block`. This is basically a classifier — there is no reasoning, no spatial computation, no composition.

### New plan (CaP-inspired)
The LLM is prompted with:
- A library of **low-level primitive APIs** (perception + control), exposed as MCP tools.
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
get_obj_names() -> list[str]
# Returns names of all objects in the scene.

get_obj_pos(name: str) -> np.ndarray  # [x, y, z]
# Returns the current 3D position of the named object.

get_obj_height(name: str) -> float
# Returns the height (z-extent) of the named object. Useful for computing
# stacking targets: place on top = get_obj_pos(bottom)  + [0, 0, get_obj_height(bottom)].

get_obj_size(name: str) -> str  # "small" | "large"
# Returns the size category of the named object.

get_obj_color(name: str) -> str
# Returns the color of the named object.

get_workspace_bounds() -> tuple[np.ndarray, np.ndarray]
# Returns (lower_corner, upper_corner) of the workspace.

is_on_top(top: str, bottom: str) -> bool
# Returns True if 'top' object is stably on 'bottom' object.
```

All perception utilities are exposed as **MCP tools**. The LLM's generated code calls these to discover the scene — no scene context is provided in the prompt itself.

### Control Primitives
```python
pick_and_place(obj_name: str, target_pos: np.ndarray) -> bool
# Picks up the named object and places it at target_pos.
# Internally runs: approach → descend → grasp → lift → move → descend → release → retreat.
# Returns True on success.
# This is the ONLY action primitive. Stacking is achieved by computing the right
# target height: target = get_obj_pos(bottom) + [0, 0, get_obj_height(bottom)]

say(message: str)
# Speaks/logs a message to the user.
```

> **Design choice**: There is no `stack_on` primitive. Stacking is just `pick_and_place` with a computed z-offset. This forces the LLM to reason about heights in code, which is the whole point of code-as-policies.

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
corner = get_corner_pos("top right")
pick_and_place("red block", corner)
```

```python
# "Move the blue block 10cm to the left of the green block."
say("Moving the blue block to the left of the green block")
green_pos = get_obj_pos("green block")
target = get_point_offset(green_pos, "left", 10)
pick_and_place("blue block", target)
```

```python
# "Place the small yellow block between the red and blue blocks."
say("Placing the yellow block between the red and blue blocks")
red_pos = get_obj_pos("red block")
blue_pos = get_obj_pos("blue block")
mid = get_midpoint(red_pos, blue_pos)
pick_and_place("small yellow block", mid)
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
red_pos = get_obj_pos("red block")
target = red_pos + np.array([0, 0, get_obj_height("red block")])
pick_and_place("green block", target)
```

```python
# "Build a tower with blue on bottom, then red, then green on top."
say("Building a tower: blue, red, green from bottom to top")
order = ["blue block", "red block", "green block"]
for i in range(1, len(order)):
    bottom = order[i - 1]
    top = order[i]
    bottom_pos = get_obj_pos(bottom)
    target = bottom_pos + np.array([0, 0, get_obj_height(bottom)])
    pick_and_place(top, target)
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
block_names = [n for n in get_obj_names() if "block" in n]
right_top = get_corner_pos("top right")
right_bottom = get_corner_pos("bottom right")
positions = make_line_positions(right_top, right_bottom, len(block_names))
for name, pos in zip(block_names, positions):
    pick_and_place(name, pos)
```

```python
# "Arrange the red and blue blocks in a circle in the center."
say("Arranging red and blue blocks in a circle")
ws_lo, ws_hi = get_workspace_bounds()
center = (ws_lo + ws_hi) / 2
targets = [n for n in get_obj_names() if "red" in n or "blue" in n]
circle_pts = make_circle_positions(center, 8, len(targets))
for name, pos in zip(targets, circle_pts):
    pick_and_place(name, pos)
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
red_pos = get_obj_pos("red block")
blue_pos = get_obj_pos("blue block")
if red_pos[0] < blue_pos[0]:
    say("Red is left of blue — swapping them")
    pick_and_place("red block", blue_pos.copy())
    pick_and_place("blue block", red_pos.copy())
else:
    say("Red is already right of blue — no swap needed")
```

```python
# "Move whichever block is closest to the center to the top left corner."
ws_lo, ws_hi = get_workspace_bounds()
center = (ws_lo + ws_hi) / 2
block_names = [n for n in get_obj_names() if "block" in n]
closest = min(block_names, key=lambda n: np.linalg.norm(get_obj_pos(n)[:2] - center[:2]))
say(f"The {closest} is closest to center — moving it to top left")
pick_and_place(closest, get_corner_pos("top left"))
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
color_order = ["red", "green", "blue"]
left = get_side_pos("left")
right = get_side_pos("right")
slots = make_line_positions(left, right, len(color_order))
for color, pos in zip(color_order, slots):
    matches = [n for n in get_obj_names() if color in n and "block" in n]
    for name in matches:
        pick_and_place(name, pos)
```

### F) Rejection / Unsupported

The LLM should still reject impossible or unsupported instructions:
- "Cut the block in half." → `say("I can only pick, place, and stack blocks — I can't cut them.")`
- "Paint the block red." → `say("I can't change block colors — I can only move them.")`

---

## 5) Language → Code → Control Pipeline

### Step 1: Prompt Construction

Build the LLM prompt with:
1. **System message**: "You are a Python code generator for a tabletop robot. Given a natural language instruction, write a short Python program using only the provided MCP API functions. You must call perception tools to discover the scene — no scene information is provided to you."
2. **API documentation**: All perception, control, and spatial helper function signatures and docstrings (the MCP tool schemas).
3. **Few-shot examples**: 8–12 examples covering categories A–F above, showing instruction → code pairs. Every example calls perception tools to discover the scene first.
4. **User instruction**: The new command to execute.

**No scene state is injected.** The LLM must generate code that calls `get_obj_names()`, `get_obj_pos()`, etc. to discover the environment. This is a key design choice — it tests whether the LLM can write code that gathers information before acting.

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
- `get_obj_names()` — query sim for all object names
- `get_obj_pos(name)` — read object 3D position from sim
- `get_obj_height(name)` — read object z-extent from sim
- `get_obj_size(name)` — read object size category from sim
- `get_obj_color(name)` — read object color from sim
- `get_workspace_bounds()` — read table bounds from sim
- `is_on_top(top, bottom)` — check stacking state in sim

#### Control (act in sim)
- `pick_and_place(obj_name, target_pos)` — the scripted motion controller (approach → descend → grasp → lift → move → descend → release → retreat). This is the hardest part — requires controller tuning, grasp reliability, stacking stability.

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
| Scene info | Injected into prompt | LLM discovers via MCP perception calls |
| Primitives | 2 high-level (`move_block`, `stack_block`) | Single `pick_and_place` + perception MCP tools + spatial helpers |
| Stacking | Dedicated `stack_block` primitive | LLM computes target height via `get_obj_pos` + `get_obj_height` |
| Spatial reasoning | None (fixed zones only) | LLM computes positions using NumPy + helper functions |
| Instruction types | Move to zone / stack in zone | Corners, offsets, midpoints, arrangements, conditionals, composites |
| Composability | Flat list of tool calls | LLM writes loops, conditionals, variable assignments |
| Destinations | Fixed `MOVE_ZONE` / `STACK_ZONE` | Any computed position on the workspace |
| Complexity | Classifier + planner | Code generation + execution (CaP-style) |
