# Task A Integration Handoff

Task B is ready to integrate against a real `taska.env` module. The Task B side already works end to end with Gemini plus deterministic stubs, so the main requirement is that Task A matches the existing environment contract exactly.

## Required `taska.env` API

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
from taska.env import get_scene_state, get_workspace_bounds, pick_and_place
```

If that import fails, it falls back to `taskb.stubs`. Once `taska.env` exists and imports cleanly, Task B will automatically use it.

## Smoke Tests After Integration

Run these after adding `taska.env`:

```bash
pytest -q
python -m taskb.main run "Put the red block in the top right corner"
python -m taskb.main run "Stack the green block on the red block."
python -m taskb.main run "If the red block is to the left of the blue block, swap them."
python -m taskb.main run "Cut the block in half."
```

What to watch for:

- placement and stacking should return `[SUCCESS]`
- rejection instructions should call `say(...)` and avoid motion
- `logs/episodes.jsonl` should contain a complete episode record
- if real-env evaluation fails, inspect `scene_before`, `scene_after`, and `call_trace` in the log

## Integration Notes

- Keep the Task A API compatible with the stub behavior in `taskb/stubs.py`.
- Avoid changing the Task B CLI or prompt contract just to fit Task A.
- If the real environment uses different object metadata internally, adapt it inside `taska.env` before returning values to Task B.
