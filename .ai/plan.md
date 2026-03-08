## Plan: MCP Language-to-Robot Control (Move + Stack)

This scope is intentionally minimal: one instruction per episode, two executable tools (`move_block`, `stack_block`), and two fixed destination areas (`MOVE_ZONE`, `STACK_ZONE`). The LLM uses MCP context to parse instruction intent, decide if the request is achievable, and then emits a sequence of primitive tool calls. Multi-block behavior is achieved by calling these primitives multiple times.

## Project Scope

- Environment: single-table robosuite scene with randomized initial block poses.
- Instruction budget: exactly one natural-language instruction per environment reset.
- Supported executable tools only:
   - `move_block(block_pos, final_dest)`
   - `stack_block(bottom_block, top_block)`
- Block references allowed in language: color + size only (e.g., "small red block", "large blue block").
- Out of scope: multi-step conversation loops, dynamic obstacle planning, free-form destination requests.

## Task Definitions

### 1) `move_block`

Goal: relocate one block to a destination slot in `MOVE_ZONE`.

Examples:
- "Move the small red block to the move area."
- "Move all small red blocks to the move area."
- "Move three large blue blocks to the move area."
- "Put the large blue block in the move zone."

Success condition:
- The moved block ends within the requested `MOVE_ZONE` slot and gripper is released.

Multi-block interpretation:
- For requests like "move all red blocks", the planner emits repeated calls to `move_block(...)`, once per selected block, until selection is exhausted or capacity is reached.

### 2) `stack_block`

Goal: place one block on top of another in `STACK_ZONE`.

Examples:
- "Stack the small green block on the large yellow block."
- "Stack all red blocks in the stack area."
- "Stack three small blocks in the stack area."
- "Place the large red block on top of the small blue block in the stack area."

Success condition:
- `top_block` is stably on `bottom_block` (height + XY alignment thresholds) and both are within `STACK_ZONE`.

Multi-block interpretation:
- For requests like "stack all red blocks", the planner emits repeated pairwise calls:
   1. Choose base block in `STACK_ZONE`.
   2. For each next block, call `stack_block(current_top, next_block)`.
   3. Update `current_top = next_block` and continue until done or height limit reached.

## Language → Plan → Control Contract

### A) Language (LLM via MCP)

Input:
- Raw user instruction.
- Scene context from MCP tools (list of blocks with `id`, `color`, `size`, `pose`; zone states).
- Allowed tool schema and argument types.

Output schema (strict JSON):

```json
{
   "action": "move_block | stack_block | reject",
   "selection": {
      "color": "red | any",
      "size": "small | large | any",
      "count": "all | integer"
   },
   "ordering": "size_asc | size_desc | arbitrary",
   "tool_calls": [
      {
         "name": "move_block",
         "args": {"block_pos": [0.0, 0.0, 0.0], "final_dest": [0.0, 0.0, 0.0]}
      },
      {
         "name": "stack_block",
         "args": {"bottom_block": "block_id", "top_block": "block_id"}
      }
   ],
   "reason": "if reject, explain why"
}
```

Rules:
- Return `reject` if instruction is outside supported actions.
- Return `reject` if requested count exceeds available matching blocks.
- Return `reject` if selected block count exceeds action capacity limits.
- If ordering is not specified, default to deterministic ordering by block id.
- `tool_calls` must contain only supported primitives with valid arguments.

### B) Plan (Symbolic subgoals)

`move_block` plan:
1. Resolve candidate set from (`color`, `size`).
2. Select K blocks based on `count` (`all` or integer).
3. Reserve K free slots in `MOVE_ZONE`.
4. Emit K primitive calls: `move_block(block_pos_i, slot_i)`.

`stack_block` plan:
1. Resolve candidate set from (`color`, `size`).
2. Select K blocks based on `count` (`all` or integer).
3. Order selected blocks using `ordering` (default deterministic by id).
4. Assign `STACK_ZONE` base pose.
5. Emit primitive sequence:
   - If needed, first place base with `move_block(base_pos, stack_base_dest)`.
   - Then for i from 2..K: `stack_block(bottom=block_{i-1}, top=block_i)`.

### C) Control (Low-level primitives)

Each subgoal compiles to deterministic primitives:
- `move_pregrasp(block)`
- `descend_and_grasp(block)`
- `lift_clearance()`
- `move_to_pose(goal_pose)`
- `descend_and_release()`
- `retreat()`

No learned controller is required for MVP; controller is scripted and repeatable.

## MCP Tooling Design

MCP read tools:
- `get_scene_state()` → all block attributes and poses.
- `get_zone_state()` → zone bounds, slot occupancy.
- `resolve_block(color, size)` → matching candidates.

MCP action tools:
- `move_block(block_pos, final_dest)` → executes pick/place for one block.
- `stack_block(bottom_block, top_block)` → executes one pairwise stack operation.
- `check_success(action, context)` → boolean + diagnostics.

MCP guardrails:
- Reject unsupported verbs/intents.
- Reject empty selections.
- Reject requests above zone or stack capacity.
- Reject impossible stack due to stability limits.
- Reject any `tool_calls` containing unsupported function names or invalid arg shapes.

## Environment Setup

- Blocks: multiple colors and two sizes (`small`, `large`).
- Initial poses: randomized each episode within spawn bounds.
- Fixed zones:
   - `MOVE_ZONE`: flat area with pre-defined placement slots (`MAX_MOVE_SLOTS`).
   - `STACK_ZONE`: area with one base stack pose and stack verification region (`MAX_STACK_HEIGHT`).
- One instruction is sampled/entered after reset, then single execution attempt.

## Evaluation Plan

Primary metric:
- Episode success rate for one-shot instruction execution.

Breakdown:
- `move_block` success rate over N randomized seeds.
- `stack_block` success rate over N randomized seeds.
- Parse validity rate (valid schema from LLM).
- Feasibility decision accuracy (correctly reject impossible/unsupported instructions).

Suggested MVP target:
- >=80% success overall across the two actions on randomized initial states.

## Failure Handling

- If parser returns `reject`, system returns a clear reason and does not execute.
- If requested count is greater than available matching blocks, reject.
- If requested count exceeds `MAX_MOVE_SLOTS` or `MAX_STACK_HEIGHT`, reject with limit reason.
- If a primitive call fails, stop execution for that episode and return failure trace.
- If execution fails (grasp miss, unstable stack), mark episode failed with trace logs.

## Logging and Traceability

For every episode, log:
1. Raw instruction.
2. Parsed JSON intent from LLM.
3. Feasibility decision.
4. Ordered primitive tool calls.
5. Primitive action sequence.
6. Final success/failure and reason.

This provides the required transparent mapping: language → plan → control.

## Team-of-3 Execution Split

- Member 1 (Language + MCP): parser prompt/schema, feasibility checks, MCP read/action wrappers.
- Member 2 (Planner): block resolution, slot assignment, subgoal generation for move/stack.
- Member 3 (Control + Eval): primitive controller integration, success checks, randomized evaluation scripts.

## Implementation Milestones

1. Scene + zones + random initialization working.
2. MCP read tools expose complete environment context.
3. LLM parser outputs strict schema and reject path.
4. `move_block` end-to-end success.
5. `stack_block` end-to-end success.
6. Evaluation harness + report-ready metrics and traces.