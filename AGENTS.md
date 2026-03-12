# Repository Guide

## Structure
- `taska/`: real robosuite implementation plus the Task B adapter API.
- `taskb/`: Gemini prompt/LLM/sandbox/evaluator/CLI pipeline.
- `tests/`: unit and integration tests for Task A + Task B wiring.
- `README.md`: run, test, and environment setup.

## Backend boundary
- `taska.api` is the function-based API Task B imports.
- `taska.BlockEnvironment` is the underlying class-based robosuite backend.
- Keep these interfaces aligned:
  - `reset_env()`
  - `get_scene_state()`
  - `get_workspace_bounds()`
  - `pick_and_place(source_id, target)`

## Runtime defaults
- Primary mode is real Task A backend.
- Stubs are for explicit testing only (`TASKB_USE_STUBS=1`).
- Use `TASKB_REQUIRE_REAL_TASKA=1` in integration runs to fail fast if real backend is unavailable.

## Core commands
- Single instruction: `python -m taskb.main run "put the red block in the top right corner"`
- Batch eval: `python -m taskb.main eval --n 10`
- Fast tests: `./.venv/bin/pytest -q -m "not robosuite"`
- Robosuite tests: `TASKB_REQUIRE_REAL_TASKA=1 ./.venv/bin/pytest -q -m robosuite`

## Engineering notes
- When changing tolerances/spatial logic, update corresponding tests in the same change.
- Prefer minimal, compatibility-preserving refactors on this branch.
