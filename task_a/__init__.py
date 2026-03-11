"""
Task A — Unified API for the robosuite block manipulation environment.

This module exposes a single class, ``BlockEnvironment``, that wraps the
robosuite simulator and provides the three functions needed by Task B:

    env = BlockEnvironment()
    env.reset()

    state  = env.get_scene_state()
    bounds = env.get_workspace_bounds()
    ok     = env.pick_and_place(source_id=0, target=np.array([0.1, 0.0, 0.82]))
    ok     = env.pick_and_place(source_id=2, target=0)   # stack block 2 on block 0
"""

import numpy as np

from task_a.env import BlockManipulationEnv
from task_a.perception import (
    get_scene_state as _get_scene_state,
    get_workspace_bounds as _get_workspace_bounds,
)
from task_a.control import pick_and_place as _pick_and_place


class BlockEnvironment:
    """
    Unified interface to the robosuite block manipulation environment.

    Wraps the low-level robosuite env, perception, and control into a clean
    API that Task B's MCP server can call directly.

    Args:
        has_renderer (bool): Open the MuJoCo viewer window. Use True for
            visual debugging, False for headless / batch evaluation.
        **env_kwargs: Additional keyword arguments forwarded to
            ``BlockManipulationEnv``.
    """

    def __init__(self, has_renderer=False, **env_kwargs):
        self._env = BlockManipulationEnv(has_renderer=has_renderer, **env_kwargs)
        self._has_reset = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """
        Reset the environment: randomise block sizes and positions.

        Returns:
            list[dict]: The initial scene state (same format as
            ``get_scene_state()``).
        """
        self._env.reset()
        self._has_reset = True
        return self.get_scene_state()

    def close(self):
        """Release simulator resources."""
        self._env.close()

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def get_scene_state(self):
        """
        Return the complete environment state: one dict per block.

        Each dict contains::

            {
                "id":     int,
                "color":  str,
                "size":   str,          # "small" | "large"
                "pos":    list[float],  # [x, y, z]
                "height": float,        # full z-extent
            }

        Raises:
            RuntimeError: If ``reset()`` has not been called yet.

        Returns:
            list[dict]
        """
        self._assert_reset()
        return _get_scene_state(self._env)

    def get_workspace_bounds(self):
        """
        Return ``(lower_corner, upper_corner)`` of the usable table surface.

        Each corner is an ``np.ndarray`` of shape ``(3,)``.  A small inset
        margin is applied so targets near the edge won't cause blocks to
        fall off.

        Returns:
            tuple[np.ndarray, np.ndarray]
        """
        return _get_workspace_bounds(self._env)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def pick_and_place(self, source_id, target):
        """
        Pick up a block and place it at a position or on another block.

        Args:
            source_id (int): ID of the block to pick up (0–5).
            target (np.ndarray | int):
                * ``np.ndarray [x, y, z]`` — absolute placement position.
                * ``int`` — stack on the block with this ID; the controller
                  computes the correct z-offset automatically.

        Raises:
            RuntimeError: If ``reset()`` has not been called yet.
            ValueError: If *source_id* or *target* ID is out of range, or
                if *source_id* equals the *target* ID.

        Returns:
            bool: True if the pick-and-place succeeded.
        """
        self._assert_reset()
        self._validate_pick_and_place(source_id, target)
        return _pick_and_place(self._env, source_id, target)

    # ------------------------------------------------------------------
    # Helpers for Task B logging
    # ------------------------------------------------------------------

    @property
    def action_dim(self):
        """Number of action dimensions (7 for Panda OSC_POSE)."""
        return self._env.action_dim

    def step(self, action):
        """Low-level step (exposed for testing / manual control)."""
        return self._env.step(action)

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _assert_reset(self):
        if not self._has_reset:
            raise RuntimeError(
                "Environment has not been reset. Call env.reset() first."
            )

    def _validate_pick_and_place(self, source_id, target):
        num_blocks = len(self._env.block_meta)
        valid_ids = set(range(num_blocks))

        if source_id not in valid_ids:
            raise ValueError(
                f"Invalid source_id={source_id}. Must be in {sorted(valid_ids)}."
            )

        if isinstance(target, (int, np.integer)):
            if target not in valid_ids:
                raise ValueError(
                    f"Invalid target block ID={target}. Must be in {sorted(valid_ids)}."
                )
            if target == source_id:
                raise ValueError(
                    f"Cannot stack block {source_id} on itself."
                )
        else:
            target = np.asarray(target, dtype=float)
            if target.shape != (3,):
                raise ValueError(
                    f"target position must be shape (3,), got {target.shape}."
                )
