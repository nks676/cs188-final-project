# Repository Guide

## Structure
- `task_a/`: real robosuite implementation (`BlockEnvironment`, control, perception).
- `taska/`: compatibility adapter exposing function API (`taska.env`) that Task B imports.
- `taskb/`: Gemini prompt/LLM/sandbox/evaluator/CLI pipeline.
- `tests/`: unit and integration tests for Task A + Task B wiring.
- `README.md`: run, test, and environment setup.

## Why both `task_a/` and `taska/`
- `task_a` is the real backend package.
- `taska.env` is a thin adapter kept for compatibility with Task B imports and monkeypatching in tests.
- Do not delete `taska/` unless all Task B imports and tests are migrated together.

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
- Keep interface contract stable for:
  - `reset_env()`
  - `get_scene_state()`
  - `get_workspace_bounds()`
  - `pick_and_place(source_id, target)`
- When changing tolerances/spatial logic, update corresponding tests in the same change.
- Prefer minimal, compatibility-preserving refactors on this branch.
