"""
Integration test for the BlockEnvironment unified API.

Exercises the full Task A interface as Task B would use it:
    reset → get_scene_state → get_workspace_bounds → pick_and_place

Run from project root:
    mjpython -m tests.test_integration
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from task_a import BlockEnvironment

RENDER_STEPS = 200


def _settle(env, steps=RENDER_STEPS):
    for _ in range(steps):
        env.step(np.zeros(env.action_dim))


def test_full_workflow():
    """Run a complete episode: reset, inspect, place, stack."""
    print("=" * 60)
    print("TEST: Full workflow through BlockEnvironment")
    print("=" * 60)

    env = BlockEnvironment(has_renderer=True)

    # --- Reset ---
    state = env.reset()
    assert len(state) == 6
    print("\nScene after reset:")
    for b in state:
        print(
            f"  ID={b['id']}  color={b['color']:>6s}  "
            f"size={b['size']:>5s}  pos=[{b['pos'][0]:+.3f}, {b['pos'][1]:+.3f}, {b['pos'][2]:.3f}]"
        )

    # --- Workspace bounds ---
    lower, upper = env.get_workspace_bounds()
    print(f"\nWorkspace: lower={lower.tolist()}, upper={upper.tolist()}")

    _settle(env)

    # --- Absolute placement: move block 0 toward center ---
    block_0 = state[0]
    target_pos = np.array([0.0, 0.0, lower[2] + block_0["height"] / 2])
    print(f"\n1) Moving block {block_0['id']} ({block_0['color']}) to {target_pos.tolist()}")

    ok = env.pick_and_place(block_0["id"], target_pos)
    print(f"   Success: {ok}")
    _settle(env)

    # --- Stack: block 2 on block 0 ---
    state = env.get_scene_state()
    block_2 = next(b for b in state if b["id"] == 2)
    print(f"\n2) Stacking block {block_2['id']} ({block_2['color']}) on block {block_0['id']} ({block_0['color']})")

    ok2 = env.pick_and_place(block_2["id"], block_0["id"])
    print(f"   Success: {ok2}")
    _settle(env)

    # --- Final state ---
    final = env.get_scene_state()
    print("\nFinal scene state:")
    for b in final:
        print(
            f"  ID={b['id']}  color={b['color']:>6s}  "
            f"pos=[{b['pos'][0]:+.3f}, {b['pos'][1]:+.3f}, {b['pos'][2]:.3f}]"
        )

    _settle(env, steps=400)
    env.close()
    return ok and ok2


def test_error_handling():
    """Verify validation catches bad inputs."""
    print("=" * 60)
    print("TEST: Error handling")
    print("=" * 60)

    env = BlockEnvironment(has_renderer=False, has_offscreen_renderer=False)

    # Calling before reset should raise
    try:
        env.get_scene_state()
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        print("  Caught RuntimeError for get_scene_state before reset — OK")

    env.reset()

    # Invalid source ID
    try:
        env.pick_and_place(99, np.array([0.0, 0.0, 0.82]))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Caught ValueError for bad source_id — OK: {e}")

    # Invalid target ID
    try:
        env.pick_and_place(0, 99)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Caught ValueError for bad target_id — OK: {e}")

    # Stack on self
    try:
        env.pick_and_place(0, 0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Caught ValueError for self-stack — OK: {e}")

    # Bad target shape
    try:
        env.pick_and_place(0, np.array([1.0, 2.0]))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Caught ValueError for bad target shape — OK: {e}")

    print("  All error handling checks passed!")
    env.close()
    return True


if __name__ == "__main__":
    r1 = test_error_handling()
    print()
    r2 = test_full_workflow()
    print("\n" + "=" * 60)
    if r1 and r2:
        print("ALL INTEGRATION TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
