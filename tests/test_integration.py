"""
Integration smoke tests: full pipeline with stubs, no LLM call.

These tests exercise: stubs → sandbox.run_code → evaluator.verify_episode → logger.log_episode.
The LLM step is replaced by pre-written code strings (as if the LLM returned them).
"""
import copy

import numpy as np
import pytest

from taskb.evaluator import verify_episode
from taskb.logger import log_episode
from taskb.sandbox import run_code
from taskb.spatial import (
    get_corner_pos,
    get_midpoint,
    get_point_offset,
    get_side_pos,
    make_circle_positions,
    make_line_positions,
)
from taskb.stubs import (
    get_scene_state,
    get_workspace_bounds,
    pick_and_place,
    reset_scene,
)
from taskb.say import say


def make_env_api():
    return {
        "get_scene_state": get_scene_state,
        "get_workspace_bounds": get_workspace_bounds,
        "pick_and_place": pick_and_place,
        "get_corner_pos": get_corner_pos,
        "get_side_pos": get_side_pos,
        "get_midpoint": get_midpoint,
        "get_point_offset": get_point_offset,
        "make_line_positions": make_line_positions,
        "make_circle_positions": make_circle_positions,
        "say": say,
    }


@pytest.fixture(autouse=True)
def fresh_scene():
    reset_scene()
    yield
    reset_scene()


# ── Category A: Direct Placement ─────────────────────────────────────────────

def test_cat_a_corner_placement():
    code = '''\
scene = get_scene_state()
red = next(b for b in scene if b["color"] == "red")
target = get_corner_pos("top right")
pick_and_place(red["id"], target)
'''
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    scene_after = get_scene_state()
    ok, reason = verify_episode(result["call_trace"], scene_after, use_stub=True)
    assert ok, reason


def test_cat_a_offset_placement():
    code = '''\
scene = get_scene_state()
blue = next(b for b in scene if b["color"] == "blue")
green = next(b for b in scene if b["color"] == "green")
target = get_point_offset(green["pos"], "left", 10)
pick_and_place(blue["id"], target)
'''
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    ok, reason = verify_episode(result["call_trace"], get_scene_state(), use_stub=True)
    assert ok, reason


# ── Category B: Stacking ─────────────────────────────────────────────────────

def test_cat_b_simple_stack():
    code = '''\
scene = get_scene_state()
green = next(b for b in scene if b["color"] == "green")
red   = next(b for b in scene if b["color"] == "red")
pick_and_place(green["id"], red["id"])
'''
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    ok, reason = verify_episode(result["call_trace"], get_scene_state(), use_stub=True)
    assert ok, reason
    # Verify the stub actually moved green on top of red
    scene = get_scene_state()
    red = next(b for b in scene if b["color"] == "red")
    green = next(b for b in scene if b["color"] == "green")
    assert green["pos"][2] > red["pos"][2]


def test_cat_b_three_stack():
    code = '''\
scene = get_scene_state()
blue  = next(b for b in scene if b["color"] == "blue")
red   = next(b for b in scene if b["color"] == "red")
green = next(b for b in scene if b["color"] == "green")
ws_min, ws_max = get_workspace_bounds()
center = get_midpoint(ws_min, ws_max)
pick_and_place(blue["id"], center)
pick_and_place(red["id"],  blue["id"])
pick_and_place(green["id"], red["id"])
'''
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    pnp_calls = [e for e in result["call_trace"] if e["fn"] == "pick_and_place"]
    assert len(pnp_calls) == 3


# ── Category C: Spatial Arrangements ─────────────────────────────────────────

def test_cat_c_line_all_blocks():
    code = '''\
scene = get_scene_state()
n = len(scene)
start = get_corner_pos("bottom right")
end   = get_corner_pos("top right")
positions = make_line_positions(start, end, n)
for block, pos in zip(scene, positions):
    pick_and_place(block["id"], pos)
'''
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    pnp_calls = [e for e in result["call_trace"] if e["fn"] == "pick_and_place"]
    assert len(pnp_calls) == 4  # all 4 blocks moved


def test_cat_c_circle():
    code = '''\
scene = get_scene_state()
ws_min, ws_max = get_workspace_bounds()
center = get_midpoint(ws_min, ws_max)
positions = make_circle_positions(center, 15, len(scene))
for block, pos in zip(scene, positions):
    pick_and_place(block["id"], pos)
'''
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    pnp_calls = [e for e in result["call_trace"] if e["fn"] == "pick_and_place"]
    assert len(pnp_calls) == 4


# ── Category D: Conditional ───────────────────────────────────────────────────

def test_cat_d_conditional_swap():
    code = '''\
scene = get_scene_state()
red  = next(b for b in scene if b["color"] == "red")
blue = next(b for b in scene if b["color"] == "blue")
if red["pos"][0] < blue["pos"][0]:
    rp = list(red["pos"])
    bp = list(blue["pos"])
    pick_and_place(red["id"],  bp)
    pick_and_place(blue["id"], rp)
'''
    # Stubs: red x=0.10, blue x=0.30 → red IS left of blue → swap happens
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    pnp_calls = [e for e in result["call_trace"] if e["fn"] == "pick_and_place"]
    assert len(pnp_calls) == 2


def test_cat_d_no_action_when_condition_false():
    code = '''\
scene = get_scene_state()
red  = next(b for b in scene if b["color"] == "red")
blue = next(b for b in scene if b["color"] == "blue")
if red["pos"][0] > blue["pos"][0]:  # false in stub scene
    pick_and_place(red["id"], list(blue["pos"]))
    pick_and_place(blue["id"], list(red["pos"]))
'''
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    pnp_calls = [e for e in result["call_trace"] if e["fn"] == "pick_and_place"]
    assert len(pnp_calls) == 0  # condition was false


# ── Category E: Composite ─────────────────────────────────────────────────────

def test_cat_e_sort_large_left_small_right():
    code = '''\
scene = get_scene_state()
large = [b for b in scene if b["size"] == "large"]
small = [b for b in scene if b["size"] == "small"]
left_start = get_corner_pos("bottom left")
left_end   = get_corner_pos("top left")
positions = make_line_positions(left_start, left_end, max(len(large), 1))
for block, pos in zip(large, positions):
    pick_and_place(block["id"], pos)
if small:
    right_center = get_side_pos("right")
    pick_and_place(small[0]["id"], right_center)
    for block in small[1:]:
        pick_and_place(block["id"], small[0]["id"])
'''
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    pnp_calls = [e for e in result["call_trace"] if e["fn"] == "pick_and_place"]
    # 2 large + 2 small (first to right, second stacked on first)
    assert len(pnp_calls) == 4


# ── Category F: Rejection ─────────────────────────────────────────────────────

def test_cat_f_rejection_say_only():
    code = 'say("I cannot cut blocks; the robot can only pick and place objects.")'
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    pnp_calls = [e for e in result["call_trace"] if e["fn"] == "pick_and_place"]
    assert len(pnp_calls) == 0  # no physical action taken


# ── Safety guardrails ─────────────────────────────────────────────────────────

def test_blocked_code_does_not_execute():
    """Ensure malicious codegen can't run even if injected."""
    code = "import os\nos.system('echo pwned')"
    result = run_code(code, make_env_api())
    assert result["success"] is False
    assert result["call_trace"] == []


# ── Logger integration ────────────────────────────────────────────────────────

def test_logger_writes_without_error():
    """log_episode should not raise even with minimal data."""
    episode_id = log_episode(
        instruction="test",
        category="A",
        generated_code="x=1",
        parse_ok=True,
        safety_ok=True,
        call_trace=[],
        scene_before=[],
        scene_after=[],
        success=True,
        failure_reason=None,
    )
    assert isinstance(episode_id, str)
    assert len(episode_id) == 36  # UUID4


def test_logger_serializes_numpy_values():
    """log_episode should serialize ndarray/scalar payloads from call traces."""
    episode_id = log_episode(
        instruction="numpy test",
        category="A",
        generated_code="pick_and_place(0, target)",
        parse_ok=True,
        safety_ok=True,
        call_trace=[
            {
                "fn": "pick_and_place",
                "args": [0, np.array([0.5, 0.4, 0.82])],
                "result": np.bool_(True),
            }
        ],
        scene_before=[{"id": 0, "pos": np.array([0.1, 0.2, 0.82])}],
        scene_after=[{"id": 0, "pos": np.array([0.5, 0.4, 0.82])}],
        success=True,
        failure_reason=None,
    )
    assert isinstance(episode_id, str)
