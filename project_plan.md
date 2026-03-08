# Project Plan: Language-to-Robot Control in Robosuite (MCP + LLM)

## 1) Goal

Build an agent that takes one natural-language instruction and executes it in robosuite using only two primitive actions:

- `move_block(block_pos, final_dest)`
- `stack_block(bottom_block, top_block)`

The LLM (through MCP) parses the instruction, checks whether it is achievable with current scene context, and outputs a sequence of these primitive calls.

---

## 2) Scope (What We Will and Won’t Do)

### In Scope
- Single-table robosuite environment.
- Randomized initial block positions each episode.
- Exactly **one instruction per environment reset**.
- Block references by **color + size** only.
- Two fixed destination regions:
  - `MOVE_ZONE`
  - `STACK_ZONE`
- Multi-block commands are supported by issuing primitive actions repeatedly.

### Out of Scope
- Multi-turn dialogue/planning loops.
- Free-form arbitrary destinations.
- Dynamic obstacle reasoning beyond fixed zones/slots.
- Learned low-level controller (MVP uses scripted primitives).

---

## 3) Supported User Intents

## A) Move

Intent meaning: move one or more selected blocks into `MOVE_ZONE`.

Example commands:
- “Move the small red block to the move area.”
- “Move all small red blocks to the move area.”
- “Move three large blue blocks to the move area.”

Execution style:
- For each selected block, emit one `move_block(block_pos, final_dest)` call.

Success criteria:
- Every requested block is in a valid `MOVE_ZONE` slot and released.

## B) Stack

Intent meaning: build a stack in `STACK_ZONE`.

Example commands:
- “Stack the small green block on the large yellow block.”
- “Stack all red blocks in the stack area.”
- “Stack three small blocks in the stack area.”

Execution style:
- Emit repeated pairwise stack operations:
  - choose base block,
  - then call `stack_block(current_top, next_block)` until done.

Success criteria:
- Required stack relations hold (`top` stably on `bottom`) and stack lies within `STACK_ZONE`.

---

## 4) Limits and Capacity Rules

The system can operate on **any number of blocks up to limits**:

- Move limit: `MAX_MOVE_SLOTS`
- Stack limit: `MAX_STACK_HEIGHT`

If a request exceeds available matching blocks or exceeds these limits, the system rejects the instruction with a clear reason.

---

## 5) Language → Plan → Control Pipeline

## Step 1: Language Understanding (LLM via MCP)

Inputs:
- User instruction.
- Scene context (`id`, `color`, `size`, pose, zone occupancy).
- Allowed tool signatures.

Output (strict JSON):

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

## Step 2: Planning

- Resolve candidate blocks from color/size filters.
- Select `count` blocks (or all).
- Allocate destination slots / stack order deterministically.
- Produce ordered primitive tool calls.

## Step 3: Control Execution

Each tool call runs scripted robot primitives:
- pregrasp
- grasp
- lift
- move
- place/release
- retreat

---

## 6) MCP Tool Interface

### Read tools
- `get_scene_state()`
- `get_zone_state()`
- `resolve_block(color, size)`

### Action tools
- `move_block(block_pos, final_dest)`
- `stack_block(bottom_block, top_block)`
- `check_success(action, context)`

### Guardrails
- Reject unsupported intents.
- Reject empty/ambiguous/impossible selections.
- Reject requests above capacity.
- Reject malformed tool calls.

---

## 7) Environment Design

- Blocks contain attributes: `id`, `color`, `size`, pose.
- Spawn positions are randomized every reset.
- `MOVE_ZONE`: fixed set of placement slots.
- `STACK_ZONE`: fixed base area for stacks + verification region.
- One instruction is executed per reset.

---

## 8) Evaluation Plan

Primary metric:
- One-shot episode success rate.

Secondary metrics:
- `move` success rate over randomized seeds.
- `stack` success rate over randomized seeds.
- Parse validity rate.
- Correct reject/feasibility accuracy.

MVP target:
- >=80% total success on randomized initial states.

---

## 9) Failure Behavior

- Unsupported or impossible request → `reject` with reason.
- Count exceeds available matching blocks → reject.
- Exceeds `MAX_MOVE_SLOTS` or `MAX_STACK_HEIGHT` → reject.
- Primitive/tool failure during execution → stop episode and log failure trace.

---

## 10) Logging (for report + demo)

For each episode log:
1. Raw instruction
2. Parsed JSON
3. Feasibility/reject decision
4. Ordered primitive tool calls
5. Execution trace
6. Final outcome + failure reason (if any)

This gives a transparent language → plan → control mapping.

---

## 11) Team Split (3 Members)

- Member 1: LLM prompt/schema + MCP parser/guardrails
- Member 2: planner (selection, ordering, slot assignment)
- Member 3: control integration + evaluation harness

---

## 12) Milestones

1. Scene randomization + zone setup complete
2. MCP read tools complete
3. LLM parsing + reject path complete
4. Move end-to-end complete
5. Stack end-to-end complete
6. Evaluation + report-ready plots/logs complete
