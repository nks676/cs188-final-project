"""Per-episode structured JSON logging."""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_logger = logging.getLogger("taskb.logger")

LOGS_DIR = Path(__file__).parent.parent / "logs"


def _ensure_logs_dir() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR


def _to_jsonable(value):
    """Recursively convert NumPy containers/scalars into JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def log_episode(
    *,
    instruction: str,
    category: str,
    generated_code: str,
    parse_ok: bool,
    safety_ok: bool,
    call_trace: list,
    scene_before: list,
    scene_after: list,
    success: bool,
    failure_reason: str | None = None,
) -> str:
    """
    Append a JSON episode record to logs/episodes.jsonl.
    Returns the episode_id (UUID4 string).
    """
    episode_id = str(uuid.uuid4())
    record = {
        "episode_id": episode_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instruction": instruction,
        "category": category,
        "generated_code": generated_code,
        "parse_ok": parse_ok,
        "safety_ok": safety_ok,
        "call_trace": _to_jsonable(call_trace),
        "scene_before": _to_jsonable(scene_before),
        "scene_after": _to_jsonable(scene_after),
        "success": success,
        "failure_reason": _to_jsonable(failure_reason),
    }

    log_path = _ensure_logs_dir() / "episodes.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    _logger.info("Episode %s logged (success=%s)", episode_id, success)
    return episode_id
