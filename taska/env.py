"""Function-based adapter over task_a.BlockEnvironment for Task B integration."""

from __future__ import annotations

import atexit
import os

_ENV = None
_USE_STUBS = None
REQUIRE_REAL_TASKA_ENV = "TASKB_REQUIRE_REAL_TASKA"
USE_STUBS_ENV = "TASKB_USE_STUBS"
RENDER_ENV = "TASKA_RENDER"


def _env_flag(name: str) -> bool:
    """Parse common truthy values from environment flags."""
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _task_a_available():
    """Return True when the real Task A environment can be constructed."""
    global _USE_STUBS
    if _USE_STUBS is None:
        try:
            from task_a import BlockEnvironment  # noqa: F401
        except ImportError:
            _USE_STUBS = True
        else:
            _USE_STUBS = False
    return not _USE_STUBS


def _require_real_task_a() -> bool:
    """Return True when the caller requires the real Task A backend."""
    return _env_flag(REQUIRE_REAL_TASKA_ENV)


def _force_stubs() -> bool:
    """Return True when tests explicitly request the stub backend."""
    return _env_flag(USE_STUBS_ENV)


def using_stub_fallback() -> bool:
    """Return True when Task B should use stub behavior instead of the real env."""
    if _force_stubs():
        return True

    if _task_a_available():
        return False

    if _require_real_task_a():
        raise RuntimeError(
            f"{REQUIRE_REAL_TASKA_ENV}=1 but the real Task A environment is unavailable. "
            "Install robosuite and Task A dependencies in the project venv."
        )

    raise RuntimeError(
        "The real Task A environment is unavailable. "
        f"Install robosuite and Task A dependencies, or set {USE_STUBS_ENV}=1 for stub-only tests."
    )


def _ensure_env():
    global _ENV
    if using_stub_fallback():
        raise RuntimeError("Task A runtime dependencies are unavailable.")
    if _ENV is None:
        from task_a import BlockEnvironment

        # Optional viewer for interactive Task B runs (disabled by default).
        render = _env_flag(RENDER_ENV)
        _ENV = BlockEnvironment(has_renderer=render, has_offscreen_renderer=False)
    return _ENV


def reset_env():
    """Reset the shared Task A environment and return the initial scene state."""
    if using_stub_fallback():
        from taskb.stubs import get_scene_state, reset_scene

        reset_scene()
        return get_scene_state()
    env = _ensure_env()
    return env.reset()


def get_scene_state():
    """Return scene state, lazily resetting the env on first use."""
    if using_stub_fallback():
        from taskb.stubs import get_scene_state as _get_scene_state

        return _get_scene_state()
    env = _ensure_env()
    try:
        return env.get_scene_state()
    except RuntimeError:
        env.reset()
        return env.get_scene_state()


def get_workspace_bounds():
    """Return workspace bounds from the shared Task A environment."""
    if using_stub_fallback():
        from taskb.stubs import get_workspace_bounds as _get_workspace_bounds

        return _get_workspace_bounds()
    return _ensure_env().get_workspace_bounds()


def pick_and_place(source_id: int, target):
    """Execute a pick-and-place action on the shared Task A environment."""
    if using_stub_fallback():
        from taskb.stubs import pick_and_place as _pick_and_place

        return _pick_and_place(source_id, target)
    env = _ensure_env()
    try:
        return env.pick_and_place(source_id, target)
    except RuntimeError:
        env.reset()
        return env.pick_and_place(source_id, target)


def close_env():
    """Close and clear the shared Task A environment."""
    global _ENV
    if _ENV is not None:
        _ENV.close()
        _ENV = None


atexit.register(close_env)
