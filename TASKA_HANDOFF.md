# Task A Integration Handoff

Task B is ready to run against the real `taska.api` backend by default. Stubs now exist only for deterministic tests, so the main requirement is that Task A matches the existing environment contract exactly.

## Required `taska.api` API

Implement these three functions:

```python
def get_scene_state() -> list[dict]: ...
def get_workspace_bounds() -> tuple[np.ndarray, np.ndarray]: ...
def pick_and_place(source_id: int, target) -> bool: ...
```

Expected `get_scene_state()` item shape:

```python
{
    "id": 0,
    "color": "red",
    "size": "large",
    "pos": [0.10, 0.20, 0.82],
    "height": 0.050,
}
```

Requirements:

- `id` must be stable within an episode.
- `color` and `size` must be strings the prompt can reason about.
- `pos` must be a length-3 position in `[x, y, z]` form.
- `height` must be the block height used for stack verification.
- `get_workspace_bounds()` must return `(ws_min, ws_max)` as NumPy arrays.
- `pick_and_place(source_id, target)` must accept:
  - an `int` target block id for stacking
  - a length-3 array/list target for absolute placement
- `pick_and_place(...)` should return `True` on success and `False` on failure.

## How Task B Chooses Real Env vs Stubs

`taskb.main` tries:

```python
from taska.api import get_scene_state, get_workspace_bounds, pick_and_place
```

Task B now uses the real backend by default. Stub behavior should only be enabled deliberately for tests via `TASKB_USE_STUBS=1`.

## Smoke Tests After Integration

Run these after wiring `taska.api`:

```bash
source .venv/bin/activate
pytest -q -m "not robosuite"
TASKB_REQUIRE_REAL_TASKA=1 pytest -q -m robosuite
TASKB_REQUIRE_REAL_TASKA=1 python -m taskb.main run "Put the red block in the top right corner"
TASKB_REQUIRE_REAL_TASKA=1 python -m taskb.main run "Stack the green block on the red block."
TASKB_REQUIRE_REAL_TASKA=1 python -m taskb.main run "If the red block is to the left of the blue block, swap them."
TASKB_REQUIRE_REAL_TASKA=1 python -m taskb.main run "Cut the block in half."
```

What to watch for:

- placement and stacking should return `[SUCCESS]`
- rejection instructions should call `say(...)` and avoid motion
- if `TASKB_REQUIRE_REAL_TASKA=1`, any silent fallback to stubs should become an error
- `logs/episodes.jsonl` should contain a complete episode record
- if real-env evaluation fails, inspect `scene_before`, `scene_after`, and `call_trace` in the log

## Integration Notes

- Keep the Task A API compatible with the stub behavior in `taskb/stubs.py`.
- Avoid changing the Task B CLI or prompt contract just to fit Task A.
- If the real environment uses different object metadata internally, adapt it inside `taska.api` before returning values to Task B.
