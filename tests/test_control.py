"""
Tests for the pick_and_place controller.

Run from project root:
    mjpython -m tests.test_control

Tests:
  1. Absolute placement — move a block to a fixed position.
  2. Stacking — stack one block on top of another.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

pytestmark = pytest.mark.robosuite
pytest.importorskip("robosuite")

from taska.env import BlockManipulationEnv
from taska.perception import get_scene_state, get_workspace_bounds
from taska.control import pick_and_place

RENDER_STEPS = 200
POS_TOLERANCE = 0.04  # metres — how close the block must be to the target
MAX_CONTROL_ATTEMPTS = 3


def make_env():
    return BlockManipulationEnv(has_renderer=False, has_offscreen_renderer=False)


def _safe_target(lower, upper, block):
    return np.array([
        (lower[0] + upper[0]) / 2 + 0.05,
        (lower[1] + upper[1]) / 2 + 0.05,
        lower[2] + block["height"] / 2,
    ])


def _settle(env, steps=RENDER_STEPS):
    """Step with zero action so the scene settles and renders."""
    for _ in range(steps):
        env.step(np.zeros(env.action_dim))


def _run_with_retries(fn, attempts=MAX_CONTROL_ATTEMPTS):
    """Retry randomized robosuite control checks to avoid one-off bad layouts."""
    last_result = None
    for _ in range(attempts):
        result = fn()
        if result[0]:
            return result
        last_result = result
    return last_result


def test_absolute_placement():
    """Pick up block 0 and place it at a fixed position on the table."""
    def _attempt():
        print("=" * 60)
        print("TEST: Absolute placement")
        print("=" * 60)

        env = make_env()
        env.reset()

        state = get_scene_state(env)
        block = state[0]
        print(f"Picking up block {block['id']} ({block['color']}, {block['size']})")
        print(f"  Start pos: {block['pos']}")

        lower, upper = get_workspace_bounds()
        target_pos = _safe_target(lower, upper, block)
        print(f"  Target pos: {target_pos.tolist()}")

        success = pick_and_place(env, block["id"], target_pos)
        print(f"  pick_and_place returned: {success}")

        _settle(env)

        final_state = get_scene_state(env)
        final_block = next(b for b in final_state if b["id"] == block["id"])
        final_pos = np.array(final_block["pos"])
        error = np.linalg.norm(final_pos[:2] - target_pos[:2])
        print(f"  Final pos:  {final_block['pos']}")
        print(f"  XY error:   {error:.4f} m")

        passed = success and error < POS_TOLERANCE
        if passed:
            print("  PASSED")
        else:
            print(f"  FAILED (success={success}, error={error:.4f})")

        _settle(env)
        env.close()
        return passed, error

    passed, error = _run_with_retries(_attempt)
    assert passed, f"absolute placement still failed after {MAX_CONTROL_ATTEMPTS} attempts (xy_error={error:.4f})"


def test_stacking():
    """Stack block 0 on top of block 1."""
    def _attempt():
        print("=" * 60)
        print("TEST: Stacking")
        print("=" * 60)

        env = make_env()
        env.reset()

        state = get_scene_state(env)
        top_block = state[0]
        bottom_block = state[1]
        print(
            f"Stacking block {top_block['id']} ({top_block['color']}) "
            f"on block {bottom_block['id']} ({bottom_block['color']})"
        )
        print(f"  Top start pos:    {top_block['pos']}")
        print(f"  Bottom start pos: {bottom_block['pos']}")

        lower, upper = get_workspace_bounds()
        anchor_target = _safe_target(lower, upper, bottom_block)
        anchor_ok = pick_and_place(env, bottom_block["id"], anchor_target)
        print(f"  Anchoring bottom block returned: {anchor_ok}")
        _settle(env)

        success = pick_and_place(env, top_block["id"], bottom_block["id"])
        print(f"  pick_and_place returned: {success}")

        _settle(env)

        final_state = get_scene_state(env)
        final_top = next(b for b in final_state if b["id"] == top_block["id"])
        final_bottom = next(b for b in final_state if b["id"] == bottom_block["id"])

        top_pos = np.array(final_top["pos"])
        bottom_pos = np.array(final_bottom["pos"])

        xy_error = np.linalg.norm(top_pos[:2] - bottom_pos[:2])
        z_diff = top_pos[2] - bottom_pos[2]
        expected_z_diff = final_bottom["height"] / 2 + final_top["height"] / 2

        print(f"  Final top pos:    {final_top['pos']}")
        print(f"  Final bottom pos: {final_bottom['pos']}")
        print(f"  XY error:         {xy_error:.4f} m")
        print(f"  Z diff:           {z_diff:.4f} m  (expected ~{expected_z_diff:.4f})")

        stacked = (
            success
            and xy_error < POS_TOLERANCE
            and abs(z_diff - expected_z_diff) < POS_TOLERANCE
        )

        if stacked:
            print("  PASSED")
        else:
            print(f"  FAILED (success={success}, xy_err={xy_error:.4f}, z_diff={z_diff:.4f})")

        _settle(env)
        env.close()
        return stacked, max(xy_error, abs(z_diff - expected_z_diff))

    stacked, err = _run_with_retries(_attempt)
    assert stacked, f"stacking still failed after {MAX_CONTROL_ATTEMPTS} attempts (worst_err={err:.4f})"


if __name__ == "__main__":
    r1 = test_absolute_placement()
    print()
    r2 = test_stacking()
    print("\n" + "=" * 60)
    if r1 and r2:
        print("ALL CONTROL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        if not r1:
            print("  - Absolute placement failed")
        if not r2:
            print("  - Stacking failed")
