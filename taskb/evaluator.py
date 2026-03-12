"""Success verification for Task B episodes."""
import numpy as np

POSITION_TOLERANCE = 0.04   # metres for x/y placement
STACK_TOLERANCE = 0.03      # metres for z stacking


def verify_episode(call_trace: list, scene_after: list, use_stub: bool = True) -> tuple[bool, str | None]:
    """
    Determine whether an episode succeeded.

    When use_stub=True, trust pick_and_place return values from the call trace.
    When use_stub=False, verify positions from scene_after.

    Returns (success: bool, failure_reason: str | None).
    """
    if not call_trace:
        return False, "No actions were taken."

    # Check for any runtime errors captured in the trace (sandbox sets success=False)
    # This function receives the call_trace from sandbox; errors bubble up separately.

    if use_stub:
        # Trust stub return values: any False means failure
        for entry in call_trace:
            if entry["fn"] == "pick_and_place" and entry["result"] is False:
                return False, f"pick_and_place returned False for args {entry['args']}"
        return True, None

    # Real environment: verify final positions from scene_after
    scene_map = {b["id"]: b for b in scene_after}

    for entry in call_trace:
        if entry["fn"] != "pick_and_place":
            continue

        args = entry["args"]
        if len(args) < 2:
            continue

        source_id = int(args[0])
        target = args[1]

        src = scene_map.get(source_id)
        if src is None:
            return False, f"Block {source_id} not found in scene_after."

        actual_pos = np.array(src["pos"])

        if isinstance(target, (int, np.integer)):
            # Stacking: check x/y alignment and z height
            tgt = scene_map.get(int(target))
            if tgt is None:
                return False, f"Target block {target} not found in scene_after."
            tgt_pos = np.array(tgt["pos"])
            xy_dist = np.linalg.norm(actual_pos[:2] - tgt_pos[:2])
            expected_z = tgt_pos[2] + tgt["height"] + src["height"] / 2
            z_err = abs(actual_pos[2] - expected_z)
            if xy_dist > POSITION_TOLERANCE:
                return False, f"Block {source_id} x/y off by {xy_dist:.3f}m (tolerance {POSITION_TOLERANCE}m)."
            if z_err > STACK_TOLERANCE:
                return False, f"Block {source_id} z off by {z_err:.3f}m (tolerance {STACK_TOLERANCE}m)."
        else:
            # Placement: check x/y
            target_pos = np.array(target)
            xy_dist = np.linalg.norm(actual_pos[:2] - target_pos[:2])
            if xy_dist > POSITION_TOLERANCE:
                return False, f"Block {source_id} x/y off by {xy_dist:.3f}m (tolerance {POSITION_TOLERANCE}m)."

    return True, None
