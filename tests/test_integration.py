"""
Integration tests for both Task B's stub pipeline and Task A's BlockEnvironment.
"""

import copy
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("TASKB_USE_STUBS", "1")

from taskb.evaluator import verify_episode
from taskb.logger import log_episode
from taskb.sandbox import run_code
import taskb.spatial as spatial
from taskb.spatial import (
    get_corner_pos,
    get_midpoint,
    get_point_offset,
    get_side_pos,
    make_circle_positions,
    make_line_positions,
)
from taskb.say import say
from taskb.stubs import (
    get_scene_state,
    get_workspace_bounds,
    pick_and_place,
    reset_scene,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_env_api():
    spatial._get_workspace_bounds = get_workspace_bounds
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
    spatial._get_workspace_bounds = get_workspace_bounds
    yield
    reset_scene()
    spatial._get_workspace_bounds = None


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
    assert len(pnp_calls) == 4


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
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    pnp_calls = [e for e in result["call_trace"] if e["fn"] == "pick_and_place"]
    assert len(pnp_calls) == 2


def test_cat_d_no_action_when_condition_false():
    code = '''\
scene = get_scene_state()
red  = next(b for b in scene if b["color"] == "red")
blue = next(b for b in scene if b["color"] == "blue")
if red["pos"][0] > blue["pos"][0]:
    pick_and_place(red["id"], list(blue["pos"]))
    pick_and_place(blue["id"], list(red["pos"]))
'''
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    pnp_calls = [e for e in result["call_trace"] if e["fn"] == "pick_and_place"]
    assert len(pnp_calls) == 0


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
    assert len(pnp_calls) == 4


def test_cat_f_rejection_say_only():
    code = 'say("I cannot cut blocks; the robot can only pick and place objects.")'
    result = run_code(code, make_env_api())
    assert result["success"], result["error"]
    pnp_calls = [e for e in result["call_trace"] if e["fn"] == "pick_and_place"]
    assert len(pnp_calls) == 0


def test_blocked_code_does_not_execute():
    code = "import os\nos.system('echo pwned')"
    result = run_code(code, make_env_api())
    assert result["success"] is False
    assert result["call_trace"] == []


def test_logger_writes_without_error():
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
    assert len(episode_id) == 36


def test_logger_serializes_numpy_values():
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


try:
    import robosuite  # noqa: F401
except ImportError:
    BlockEnvironment = None
else:
    from task_a import BlockEnvironment

RENDER_STEPS = 200


def _settle(env, steps=RENDER_STEPS):
    for _ in range(steps):
        env.step(np.zeros(env.action_dim))


def _safe_real_env_target(lower, upper, block):
    """Choose a placement target that matches the known-good Task A control tests."""
    return np.array([
        (lower[0] + upper[0]) / 2 + 0.05,
        (lower[1] + upper[1]) / 2 + 0.05,
        lower[2] + block["height"] / 2,
    ])


@pytest.fixture
def require_real_task_a(monkeypatch):
    monkeypatch.setenv("TASKB_REQUIRE_REAL_TASKA", "1")


def test_require_real_task_a_raises_without_robosuite(monkeypatch):
    import taska.env as taska_env

    monkeypatch.delenv("TASKB_USE_STUBS", raising=False)
    monkeypatch.setenv("TASKB_REQUIRE_REAL_TASKA", "1")
    monkeypatch.setattr(taska_env, "_USE_STUBS", True)

    with pytest.raises(RuntimeError, match="TASKB_REQUIRE_REAL_TASKA=1"):
        taska_env.get_workspace_bounds()


@pytest.mark.skipif(BlockEnvironment is None, reason="robosuite not installed")
@pytest.mark.robosuite
def test_task_a_full_workflow(require_real_task_a):
    env = BlockEnvironment(has_renderer=False, has_offscreen_renderer=False)
    state = env.reset()
    assert len(state) == 6

    lower, upper = env.get_workspace_bounds()
    _settle(env)

    block_0 = state[0]
    target_pos = _safe_real_env_target(lower, upper, block_0)
    ok = env.pick_and_place(block_0["id"], target_pos)
    _settle(env)

    state = env.get_scene_state()
    block_2 = next(b for b in state if b["id"] == 2)
    ok2 = env.pick_and_place(block_2["id"], block_0["id"])
    _settle(env)

    final_state = env.get_scene_state()
    env.close()

    final_block_0 = next(b for b in final_state if b["id"] == block_0["id"])
    final_block_2 = next(b for b in final_state if b["id"] == block_2["id"])

    placement_xy_error = np.linalg.norm(np.array(final_block_0["pos"][:2]) - target_pos[:2])
    stack_xy_error = np.linalg.norm(
        np.array(final_block_2["pos"][:2]) - np.array(final_block_0["pos"][:2])
    )
    expected_z = final_block_0["pos"][2] + final_block_0["height"] / 2 + final_block_2["height"] / 2
    stack_z_error = abs(final_block_2["pos"][2] - expected_z)

    assert ok or placement_xy_error < 0.04
    assert ok2 or (stack_xy_error < 0.04 and stack_z_error < 0.02)


@pytest.mark.skipif(BlockEnvironment is None, reason="robosuite not installed")
@pytest.mark.robosuite
def test_task_a_error_handling(require_real_task_a):
    env = BlockEnvironment(has_renderer=False, has_offscreen_renderer=False)

    with pytest.raises(RuntimeError):
        env.get_scene_state()

    env.reset()

    with pytest.raises(ValueError):
        env.pick_and_place(99, np.array([0.0, 0.0, 0.82]))
    with pytest.raises(ValueError):
        env.pick_and_place(0, 99)
    with pytest.raises(ValueError):
        env.pick_and_place(0, 0)
    with pytest.raises(ValueError):
        env.pick_and_place(0, np.array([1.0, 2.0]))

    env.close()


@pytest.mark.skipif(BlockEnvironment is None, reason="robosuite not installed")
@pytest.mark.robosuite
def test_taska_adapter_uses_real_env(require_real_task_a):
    import taska.env as taska_env

    os.environ.pop("TASKB_USE_STUBS", None)
    taska_env.close_env()
    taska_env.reset_env()
    scene = taska_env.get_scene_state()
    lower, upper = taska_env.get_workspace_bounds()

    assert not taska_env.using_stub_fallback()
    assert len(scene) == 6
    assert lower.shape == (3,)
    assert upper.shape == (3,)


@pytest.mark.skipif(BlockEnvironment is None, reason="robosuite not installed")
@pytest.mark.robosuite
def test_task_b_pipeline_against_real_task_a(require_real_task_a):
    import taska.env as taska_env

    os.environ.pop("TASKB_USE_STUBS", None)
    taska_env.close_env()
    taska_env.reset_env()
    env_api = {
        "get_scene_state": taska_env.get_scene_state,
        "get_workspace_bounds": taska_env.get_workspace_bounds,
        "pick_and_place": taska_env.pick_and_place,
        "get_corner_pos": get_corner_pos,
        "get_side_pos": get_side_pos,
        "get_midpoint": get_midpoint,
        "get_point_offset": get_point_offset,
        "make_line_positions": make_line_positions,
        "make_circle_positions": make_circle_positions,
        "say": say,
    }
    code = '''\
scene = get_scene_state()
red = next(b for b in scene if b["color"] == "red")
lower, upper = get_workspace_bounds()
target = np.array([
    (lower[0] + upper[0]) / 2 + 0.05,
    (lower[1] + upper[1]) / 2 + 0.05,
    lower[2] + red["height"] / 2,
])
pick_and_place(red["id"], target)
'''
    result = run_code(code, env_api)
    assert result["success"], result["error"]
    _settle(taska_env._ensure_env())
    scene_after = copy.deepcopy(taska_env.get_scene_state())
    ok, reason = verify_episode(result["call_trace"], scene_after, use_stub=False)
    assert ok, reason
