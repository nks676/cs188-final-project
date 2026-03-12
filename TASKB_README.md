# Task B: Code-as-Policies Block Manipulation

This branch contains the Task B pipeline for turning natural-language block manipulation instructions into executable Python code. The code generator uses Gemini, the generated code runs through a sandbox, and the default runtime path uses the real Task A robosuite backend when it is installed.

## Setup

Create and activate a project virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the base dependencies:

```bash
pip install -r requirements.txt
```

For real robosuite integration runs, install the extra physics dependencies in the same venv:

```bash
pip install -r requirements-robosuite.txt
```

If robosuite installation fails under your local interpreter, switch to a Python version supported by robosuite and recreate `.venv`.

Set your Gemini API key before running live generation:

```bash
export GEMINI_API_KEY=your_key_here
```

You can also create a local `.env` file at the repo root with:

```bash
GEMINI_API_KEY=your_key_here
```

## Run

Run one instruction through the full pipeline:

```bash
python -m taskb.main run "put the red block in the top right corner"
```

To run the same Gemini-generated flow with the robosuite viewer enabled:

```bash
TASKA_RENDER=1 TASKB_REQUIRE_REAL_TASKA=1 python -m taskb.main run "put the red block in the top right corner"
```

Run batch evaluation on a random sample:

```bash
python -m taskb.main eval --n 10
```

By default, Task B uses the real `taska.env` backend.

To force stub behavior for test-only workflows, set:

```bash
export TASKB_USE_STUBS=1
```

To require the real backend and raise immediately if it is unavailable, set:

```bash
export TASKB_REQUIRE_REAL_TASKA=1
```

## Tests

Run the fast suite without robosuite:

```bash
pytest -q -m "not robosuite"
```

These tests force the stub backend internally so they stay deterministic even when robosuite is installed.

Run the robosuite-backed integration suite in the same venv:

```bash
TASKB_REQUIRE_REAL_TASKA=1 pytest -q -m robosuite
```

The suite covers spatial helpers, sandbox behavior, evaluator logic, prompt construction, LLM response handling, stub-backed integration flow, and real Task A integration when robosuite is installed.

## Logs

Each episode is appended to `logs/episodes.jsonl` with:

- instruction and category
- generated code
- parse/safety status
- call trace
- scene snapshots before and after execution
- success/failure outcome

## Repo Map

- `taskb/main.py`: CLI entrypoint
- `taskb/prompt.py`: prompt text and few-shot examples
- `taskb/llm.py`: Gemini client, response extraction, retry handling
- `taskb/sandbox.py`: AST safety checks and execution harness
- `taskb/evaluator.py`: success verification
- `taskb/stubs.py`: deterministic mock environment
- `tests/`: unit and integration coverage
