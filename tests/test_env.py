"""
Smoke tests for BlockManipulationEnv with visual rendering.

Run from project root:
    mjpython -m tests.test_env
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from task_a.env import BlockManipulationEnv
from task_a.config import BLOCK_COLORS

RENDER_STEPS = 300


def make_env():
    return BlockManipulationEnv(has_renderer=True)


def test_env_creation_and_reset():
    """Verify env creates, resets, and blocks are on the table."""
    print("Creating BlockManipulationEnv...")
    env = make_env()

    print("Resetting environment...")
    obs = env.reset()

    # Check that we have 6 blocks
    assert len(env.block_meta) == 6, f"Expected 6 blocks, got {len(env.block_meta)}"
    assert len(env.block_body_ids) == 6, f"Expected 6 body IDs, got {len(env.block_body_ids)}"

    print("\nBlock metadata after reset:")
    print("-" * 60)
    for meta in env.block_meta:
        body_id = env.block_body_ids[meta["id"]]
        pos = env.sim.data.body_xpos[body_id]
        print(
            f"  ID={meta['id']}  color={meta['color']:>6s}  "
            f"size={meta['size']:>5s}  pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}]"
        )

    # Verify all blocks are above the table surface (z > 0.8)
    for meta in env.block_meta:
        body_id = env.block_body_ids[meta["id"]]
        pos = env.sim.data.body_xpos[body_id]
        assert pos[2] > 0.79, f"Block {meta['color']} z={pos[2]:.3f} is below the table!"

    # Verify colors match config
    for meta, expected in zip(env.block_meta, BLOCK_COLORS):
        assert meta["color"] == expected["color"], (
            f"Color mismatch: {meta['color']} != {expected['color']}"
        )

    # Verify sizes are valid
    for meta in env.block_meta:
        assert meta["size"] in ("small", "large"), f"Invalid size: {meta['size']}"

    print("\nAll checks passed! Rendering for a few seconds...")
    for _ in range(RENDER_STEPS):
        env.step(np.zeros(env.action_dim))

    env.close()


def test_reset_randomizes():
    """Verify that resetting changes block positions and possibly sizes."""
    env = make_env()

    env.reset()
    positions_1 = []
    sizes_1 = []
    for meta in env.block_meta:
        body_id = env.block_body_ids[meta["id"]]
        positions_1.append(env.sim.data.body_xpos[body_id].copy())
        sizes_1.append(meta["size"])

    print("First reset — rendering...")
    for _ in range(RENDER_STEPS):
        env.step(np.zeros(env.action_dim))

    env.reset()
    positions_2 = []
    sizes_2 = []
    for meta in env.block_meta:
        body_id = env.block_body_ids[meta["id"]]
        positions_2.append(env.sim.data.body_xpos[body_id].copy())
        sizes_2.append(meta["size"])

    print("Second reset — rendering...")
    for _ in range(RENDER_STEPS):
        env.step(np.zeros(env.action_dim))

    # Positions should differ (extremely unlikely to be identical)
    positions_changed = any(
        not np.allclose(p1, p2) for p1, p2 in zip(positions_1, positions_2)
    )
    print(f"Positions changed between resets: {positions_changed}")
    assert positions_changed, "Positions should change between resets"

    print(f"Sizes reset 1: {sizes_1}")
    print(f"Sizes reset 2: {sizes_2}")

    print("Randomization check passed!")
    env.close()


def test_step():
    """Verify env.step works with a zero action."""
    env = make_env()
    env.reset()

    action = np.zeros(env.action_dim)
    obs, reward, done, info = env.step(action)

    print(f"Action dim: {env.action_dim}")
    print(f"Step returned — reward={reward}, done={done}")

    print("Rendering for a few seconds...")
    for _ in range(RENDER_STEPS):
        env.step(np.zeros(env.action_dim))

    print("Step check passed!")
    env.close()


if __name__ == "__main__":
    test_env_creation_and_reset()
    print("\n" + "=" * 60 + "\n")
    test_reset_randomizes()
    print("\n" + "=" * 60 + "\n")
    test_step()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
