# Task B: Code-as-Policies Block Manipulation

This branch contains the Task B pipeline for turning natural-language block manipulation instructions into executable Python code. The code generator uses Gemini, the generated code runs through a sandbox, and the default development path uses deterministic stubs instead of a live robosuite environment.

## Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

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

Run batch evaluation on a random sample:

```bash
python -m taskb.main eval --n 10
```

If `taska.env` is unavailable, the CLI automatically falls back to `taskb.stubs`.

## Tests

Run the Task B test suite with:

```bash
pytest -q
```

The suite covers spatial helpers, sandbox behavior, evaluator logic, prompt construction, LLM response handling, and stub-backed integration flow.

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
