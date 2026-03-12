"""
Scripted pick-and-place controller for the BlockManipulationEnv.

Implements the motion sequence:
  open gripper → approach → descend → grasp → lift → move → descend → release → retreat

Supports two modes via the `target` argument:
  - Absolute placement: target is np.ndarray [x, y, z]
  - Stacking: target is int (block ID); z-offset computed automatically
"""

import numpy as np

from taska.config import PLACEMENT_Z_OFFSET, SIZE_MAP

# ---------------------------------------------------------------------------
# Gripper action values (robosuite Panda convention)
# ---------------------------------------------------------------------------
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

# ---------------------------------------------------------------------------
# Motion parameters — expect to tune these
# ---------------------------------------------------------------------------
HOVER_HEIGHT = 0.15          # metres above block / target to hover
LIFT_HEIGHT = 0.15           # metres to lift after grasping
STACK_RELEASE_OFFSET = 0.01  # release slightly above computed target when stacking
POS_THRESHOLD = 0.01         # metres — convergence threshold
MAX_STEPS_PER_PHASE = 700    # simulation steps before declaring a phase timed-out
GRIPPER_WAIT_STEPS = 75      # steps to hold the gripper action

# From the default Panda OSC_POSE controller config: output_max for position
POS_ACTION_SCALE = 0.05

# Speed limit — caps the position action magnitude in [-1, 1] range.
# 1.0 = full speed (0.05 m/step → ~1 m/s at 20 Hz)
# 0.25 = quarter speed (~0.25 m/s) — smooth and reliable
MAX_ACTION_MAG = 0.15
SURFACE_Z_THRESHOLD = 0.02
PLACEMENT_XY_TOLERANCE = 0.04
STACK_Z_TOLERANCE = 0.03
POST_ACTION_SETTLE_STEPS = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_ee_pos(env):
    """Return the current end-effector (grip site) position as np.ndarray(3,)."""
    robot = env.robots[0]
    arm = robot.arms[0]
    return np.array(env.sim.data.site_xpos[robot.eef_site_id[arm]])


def _get_block_pos(env, block_id):
    """Return the current 3-D position of a block."""
    body_id = env.block_body_ids[block_id]
    return np.array(env.sim.data.body_xpos[body_id])


def _make_action(delta_pos, gripper):
    """Build a 7-D OSC_POSE action: [pos(3) | ori(3) | gripper(1)]."""
    pos_action = delta_pos / POS_ACTION_SCALE
    # Cap magnitude so the arm moves at a controlled speed
    mag = np.linalg.norm(pos_action)
    if mag > MAX_ACTION_MAG:
        pos_action = pos_action / mag * MAX_ACTION_MAG
    pos_action = np.clip(pos_action, -1.0, 1.0)
    ori_action = np.zeros(3)
    return np.concatenate([pos_action, ori_action, [gripper]])


def _move_to(env, target_pos, gripper,
             threshold=POS_THRESHOLD, max_steps=MAX_STEPS_PER_PHASE):
    """
    Proportional controller that drives the EEF toward *target_pos*.

    Returns True if the EEF converges within *threshold*, False on timeout.
    """
    for _ in range(max_steps):
        ee_pos = _get_ee_pos(env)
        delta = target_pos - ee_pos
        if np.linalg.norm(delta) < threshold:
            return True
        action = _make_action(delta, gripper)
        env.step(action)
    return False


def _actuate_gripper(env, gripper_action, steps=GRIPPER_WAIT_STEPS):
    """Hold a constant gripper command for *steps* simulation steps."""
    for _ in range(steps):
        action = _make_action(np.zeros(3), gripper_action)
        env.step(action)


def _settle(env, steps=POST_ACTION_SETTLE_STEPS):
    """Advance the sim with zero actions so blocks can settle."""
    zero_action = np.zeros(env.action_dim)
    for _ in range(steps):
        env.step(zero_action)


def _normalize_place_pos(env, target, source_half_z):
    """Interpret table-surface z targets as tabletop placements for the source block."""
    place_pos = np.asarray(target, dtype=float).copy()
    table_z = float(env.table_offset[2])
    if place_pos[2] <= table_z + SURFACE_Z_THRESHOLD:
        place_pos[2] = table_z + PLACEMENT_Z_OFFSET + source_half_z
    return place_pos


def _verify_result(env, source_id, place_pos, *, stacking, target_id=None):
    """Return True when the final block pose matches the requested outcome."""
    _settle(env)
    source_pos = _get_block_pos(env, source_id)

    if stacking:
        bottom_pos = _get_block_pos(env, target_id)
        source_half_z = SIZE_MAP[env.block_meta[source_id]["size"]][2]
        target_half_z = SIZE_MAP[env.block_meta[target_id]["size"]][2]
        xy_error = np.linalg.norm(source_pos[:2] - bottom_pos[:2])
        z_error = abs(source_pos[2] - (bottom_pos[2] + source_half_z + target_half_z))
        return xy_error < PLACEMENT_XY_TOLERANCE and z_error < STACK_Z_TOLERANCE

    xy_error = np.linalg.norm(source_pos[:2] - np.asarray(place_pos)[:2])
    return xy_error < PLACEMENT_XY_TOLERANCE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pick_and_place(env, source_id, target):
    """
    Pick up a block and place it at an absolute position or on another block.

    Args:
        env:  BlockManipulationEnv (must already be ``reset()``).
        source_id (int): ID of the block to pick up.
        target (np.ndarray | int):
            * ``np.ndarray [x, y, z]`` — place at this absolute position.
            * ``int`` — stack on the block with this ID (height computed
              automatically from both blocks' sizes).

    Returns:
        bool: True if the pick-and-place succeeded.
    """
    # ------------------------------------------------------------------
    # 0.  Resolve source & target
    # ------------------------------------------------------------------
    source_meta = env.block_meta[source_id]
    source_half_z = SIZE_MAP[source_meta["size"]][2]
    source_pos = _get_block_pos(env, source_id)

    stacking = isinstance(target, (int, np.integer))

    if stacking:
        target_meta = env.block_meta[target]
        target_half_z = SIZE_MAP[target_meta["size"]][2]
        bottom_pos = _get_block_pos(env, target)
        # Compute so that the bottom of source sits on the top of target.
        place_pos = np.array([
            bottom_pos[0],
            bottom_pos[1],
            bottom_pos[2] + target_half_z + source_half_z,
        ])
    else:
        place_pos = _normalize_place_pos(env, target, source_half_z)

    def _failed():
        return _verify_result(
            env,
            source_id,
            place_pos,
            stacking=stacking,
            target_id=int(target) if stacking else None,
        )

    # ------------------------------------------------------------------
    # 1.  Open gripper
    # ------------------------------------------------------------------
    _actuate_gripper(env, GRIPPER_OPEN)

    # ------------------------------------------------------------------
    # 2.  Approach — move above source block
    # ------------------------------------------------------------------
    approach_pos = source_pos.copy()
    approach_pos[2] += HOVER_HEIGHT
    if not _move_to(env, approach_pos, GRIPPER_OPEN):
        return _failed()

    # ------------------------------------------------------------------
    # 3.  Descend — lower to grasp height (block centre)
    # ------------------------------------------------------------------
    grasp_pos = source_pos.copy()
    if not _move_to(env, grasp_pos, GRIPPER_OPEN):
        return _failed()

    # ------------------------------------------------------------------
    # 4.  Grasp — close gripper
    # ------------------------------------------------------------------
    _actuate_gripper(env, GRIPPER_CLOSE)

    # ------------------------------------------------------------------
    # 5.  Verify grasp
    # ------------------------------------------------------------------
    grasped = env._check_grasp(
        gripper=env.robots[0].gripper,
        object_geoms=env.block_objects[source_id],
    )
    if not grasped:
        return _failed()

    # ------------------------------------------------------------------
    # 6.  Lift
    # ------------------------------------------------------------------
    lift_pos = _get_ee_pos(env).copy()
    lift_pos[2] += LIFT_HEIGHT
    if not _move_to(env, lift_pos, GRIPPER_CLOSE):
        return _failed()

    # ------------------------------------------------------------------
    # 7.  Move — translate to above target
    # ------------------------------------------------------------------
    above_target = place_pos.copy()
    above_target[2] += HOVER_HEIGHT
    if not _move_to(env, above_target, GRIPPER_CLOSE):
        return _failed()

    # ------------------------------------------------------------------
    # 8.  Descend — lower to placement height
    # ------------------------------------------------------------------
    release_pos = place_pos.copy()
    if stacking:
        release_pos[2] += STACK_RELEASE_OFFSET
    if not _move_to(env, release_pos, GRIPPER_CLOSE):
        return _failed()

    # ------------------------------------------------------------------
    # 9.  Release — open gripper
    # ------------------------------------------------------------------
    _actuate_gripper(env, GRIPPER_OPEN)

    # ------------------------------------------------------------------
    # 10. Retreat — move up to safe height
    # ------------------------------------------------------------------
    retreat_pos = _get_ee_pos(env).copy()
    retreat_pos[2] += HOVER_HEIGHT
    _move_to(env, retreat_pos, GRIPPER_OPEN)

    return _verify_result(
        env,
        source_id,
        place_pos,
        stacking=stacking,
        target_id=int(target) if stacking else None,
    )
