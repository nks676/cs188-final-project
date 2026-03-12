"""
Tests for perception functions: get_scene_state() and get_workspace_bounds().

Run from project root:
    mjpython -m tests.test_perception
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

pytestmark = pytest.mark.robosuite
pytest.importorskip("robosuite")

from task_a.env import BlockManipulationEnv
from task_a.perception import get_scene_state, get_workspace_bounds

RENDER_STEPS = 200


def make_env():
    return BlockManipulationEnv(has_renderer=False, has_offscreen_renderer=False)


def test_get_scene_state():
    """Verify get_scene_state returns correct block info."""
    print("Testing get_scene_state()...")
    env = make_env()
    env.reset()

    state = get_scene_state(env)

    assert len(state) == 6, f"Expected 6 blocks, got {len(state)}"

    print("\nScene state:")
    print("-" * 70)
    for block in state:
        print(
            f"  ID={block['id']}  color={block['color']:>6s}  "
            f"size={block['size']:>5s}  height={block['height']:.3f}  "
            f"pos=[{block['pos'][0]:+.3f}, {block['pos'][1]:+.3f}, {block['pos'][2]:.3f}]"
        )

    for block in state:
        # Check required keys
        for key in ("id", "color", "size", "pos", "height"):
            assert key in block, f"Missing key '{key}' in block {block}"

        # Check types
        assert isinstance(block["id"], int)
        assert isinstance(block["color"], str)
        assert isinstance(block["size"], str) and block["size"] in ("small", "large")
        assert isinstance(block["pos"], list) and len(block["pos"]) == 3
        assert isinstance(block["height"], float) and block["height"] > 0

        # Height should match size category
        expected_height = 0.040 if block["size"] == "small" else 0.050
        assert abs(block["height"] - expected_height) < 1e-6, (
            f"Block {block['color']}: height {block['height']} != expected {expected_height}"
        )

        # Block should be above the table
        assert block["pos"][2] > 0.79, (
            f"Block {block['color']} z={block['pos'][2]:.3f} is below the table"
        )

    # IDs should be 0–5
    ids = sorted(b["id"] for b in state)
    assert ids == list(range(6)), f"Expected IDs 0-5, got {ids}"

    # pos should be a plain list (JSON-serializable), not numpy
    assert type(state[0]["pos"]) is list, (
        f"pos should be a list, got {type(state[0]['pos'])}"
    )

    print("\nRendering...")
    for _ in range(RENDER_STEPS):
        env.step(np.zeros(env.action_dim))

    print("get_scene_state() checks passed!")
    env.close()


def test_get_workspace_bounds():
    """Verify workspace bounds are reasonable."""
    print("Testing get_workspace_bounds()...")

    lower, upper = get_workspace_bounds()

    print(f"  Lower corner: {lower}")
    print(f"  Upper corner: {upper}")

    # Both should be numpy arrays of shape (3,)
    assert isinstance(lower, np.ndarray) and lower.shape == (3,)
    assert isinstance(upper, np.ndarray) and upper.shape == (3,)

    # Upper should be greater than lower in x and y
    assert upper[0] > lower[0], "upper x should be > lower x"
    assert upper[1] > lower[1], "upper y should be > lower y"

    # Z should be at table surface height (0.8)
    assert abs(lower[2] - 0.8) < 0.01, f"lower z should be ~0.8, got {lower[2]}"
    assert abs(upper[2] - 0.8) < 0.01, f"upper z should be ~0.8, got {upper[2]}"

    # Bounds should be symmetric around the origin
    assert abs(lower[0] + upper[0]) < 0.01, "x bounds should be symmetric"
    assert abs(lower[1] + upper[1]) < 0.01, "y bounds should be symmetric"

    # Should be smaller than the full table (margin applied)
    assert upper[0] < 0.4, "upper x should be < 0.4 (half table width)"
    assert upper[0] > 0.2, "upper x should be > 0.2 (reasonable workspace)"

    print("get_workspace_bounds() checks passed!")


def test_scene_state_matches_sim():
    """Verify that reported positions match the actual sim state."""
    print("Testing scene state matches sim positions...")
    env = make_env()
    env.reset()

    state = get_scene_state(env)

    for block in state:
        body_id = env.block_body_ids[block["id"]]
        sim_pos = env.sim.data.body_xpos[body_id]
        reported_pos = np.array(block["pos"])
        assert np.allclose(sim_pos, reported_pos, atol=1e-6), (
            f"Block {block['color']}: sim pos {sim_pos} != reported {reported_pos}"
        )

    # Step the sim forward and check positions update
    for _ in range(50):
        env.step(np.zeros(env.action_dim))

    state_after = get_scene_state(env)
    for block in state_after:
        body_id = env.block_body_ids[block["id"]]
        sim_pos = env.sim.data.body_xpos[body_id]
        reported_pos = np.array(block["pos"])
        assert np.allclose(sim_pos, reported_pos, atol=1e-6), (
            f"After stepping — Block {block['color']}: sim pos {sim_pos} != reported {reported_pos}"
        )

    print("Rendering...")
    for _ in range(RENDER_STEPS):
        env.step(np.zeros(env.action_dim))

    print("Scene state / sim consistency checks passed!")
    env.close()


if __name__ == "__main__":
    test_get_workspace_bounds()
    print("\n" + "=" * 60 + "\n")
    test_get_scene_state()
    print("\n" + "=" * 60 + "\n")
    test_scene_state_matches_sim()
    print("\n" + "=" * 60)
    print("ALL PERCEPTION TESTS PASSED")
