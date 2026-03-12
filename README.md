# CS188 Final Project: Task A + Task B Integration

Task B takes natural language, gets Gemini-generated Python code, validates it in a sandbox, and executes actions on the real robosuite backend.

## Repo layout

- `task_a/`: real robosuite backend (`BlockEnvironment` and low-level control/perception).
- `taska/`: compatibility adapter (`taska.env`) used by Task B.
- `taskb/`: prompt, LLM call, sandbox, evaluator, CLI.
- `tests/`: unit and integration tests.

`task_a` and `taska` both exist intentionally: `taska.env` is a thin bridge API while `task_a` contains the actual implementation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-robosuite.txt
```

Set API key (choose one):

```bash
export GEMINI_API_KEY=your_key_here
```

or `.env` at repo root:

```bash
GEMINI_API_KEY=your_key_here
```

## Run

Headless:

```bash
python -m taskb.main run "put the red block in the top right corner"
```

Batch evaluation:

```bash
python -m taskb.main eval --n 10
```

Visual run on macOS (required for MuJoCo passive viewer):

```bash
TASKA_RENDER=1 TASKB_REQUIRE_REAL_TASKA=1 ./.venv/bin/mjpython -m taskb.main run "put the red block in the top right corner"
```

Visual run on Linux/Windows (regular `python` is usually fine):

```bash
TASKA_RENDER=1 TASKB_REQUIRE_REAL_TASKA=1 python -m taskb.main run "put the red block in the top right corner"
```

Generated Gemini code is printed in the terminal under `--- Generated code ---`.

## Test

Fast tests (no robosuite):

```bash
./.venv/bin/pytest -q -m "not robosuite"
```

Real robosuite tests:

```bash
TASKB_REQUIRE_REAL_TASKA=1 ./.venv/bin/pytest -q -m robosuite
```

Targeted integration check:

```bash
TASKB_REQUIRE_REAL_TASKA=1 ./.venv/bin/pytest -q tests/test_integration.py::test_task_a_full_workflow
```

## Flags

- `TASKB_REQUIRE_REAL_TASKA=1`: require real backend; fail fast if unavailable.
- `TASKB_USE_STUBS=1`: force stubs (testing only).
- `TASKA_RENDER=1`: enable robosuite viewer.

Default mode is real Task A backend. Stubs are only used when explicitly requested.

## Logs

Runs are logged to `logs/episodes.jsonl` with instruction, generated code, safety checks, call trace, before/after scene state, and success/failure reason.
