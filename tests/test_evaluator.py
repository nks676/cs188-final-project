"""Unit tests for taskb/evaluator.py."""
import pytest

from taskb.evaluator import POSITION_TOLERANCE, STACK_TOLERANCE, verify_episode


def _block(bid, pos, height=0.05):
    return {"id": bid, "color": "red", "size": "large", "pos": list(pos), "height": height}


# ── stub mode (trusts return values) ─────────────────────────────────────────

class TestVerifyEpisodeStubMode:
    def test_empty_trace_fails(self):
        success, reason = verify_episode([], [], use_stub=True)
        assert success is False
        assert reason is not None

    def test_all_true_returns_succeeds(self):
        trace = [
            {"fn": "get_scene_state", "args": [], "result": []},
            {"fn": "pick_and_place", "args": [0, [0.1, 0.2, 0.82]], "result": True},
        ]
        success, reason = verify_episode(trace, [], use_stub=True)
        assert success is True
        assert reason is None

    def test_false_return_fails(self):
        trace = [{"fn": "pick_and_place", "args": [99, [0, 0, 0]], "result": False}]
        success, reason = verify_episode(trace, [], use_stub=True)
        assert success is False
        assert "False" in reason

    def test_non_pnp_calls_ignored(self):
        trace = [
            {"fn": "get_scene_state", "args": [], "result": []},
            {"fn": "say", "args": ["hello"], "result": None},
        ]
        # No pick_and_place calls → trace not empty, but no pnp to fail
        success, reason = verify_episode(trace, [], use_stub=True)
        assert success is True

    def test_multiple_pnp_all_true(self):
        trace = [
            {"fn": "pick_and_place", "args": [0, [0.1, 0.2, 0.82]], "result": True},
            {"fn": "pick_and_place", "args": [1, [0.3, 0.3, 0.82]], "result": True},
        ]
        success, _ = verify_episode(trace, [], use_stub=True)
        assert success is True

    def test_second_pnp_false_fails(self):
        trace = [
            {"fn": "pick_and_place", "args": [0, [0.1, 0.2, 0.82]], "result": True},
            {"fn": "pick_and_place", "args": [1, [0.3, 0.3, 0.82]], "result": False},
        ]
        success, reason = verify_episode(trace, [], use_stub=True)
        assert success is False


# ── real mode (checks positions) ─────────────────────────────────────────────

class TestVerifyEpisodeRealMode:
    def test_placement_within_tolerance(self):
        target = [0.4, 0.3, 0.82]
        # block moved to exactly the target
        scene_after = [_block(0, target)]
        trace = [{"fn": "pick_and_place", "args": [0, target], "result": True}]
        success, reason = verify_episode(trace, scene_after, use_stub=False)
        assert success is True

    def test_placement_outside_tolerance(self):
        target = [0.4, 0.3, 0.82]
        actual = [0.4 + POSITION_TOLERANCE + 0.01, 0.3, 0.82]
        scene_after = [_block(0, actual)]
        trace = [{"fn": "pick_and_place", "args": [0, target], "result": True}]
        success, reason = verify_episode(trace, scene_after, use_stub=False)
        assert success is False
        assert "x/y" in reason

    def test_stacking_xy_within_tolerance(self):
        tgt_block = _block(1, [0.3, 0.3, 0.82], height=0.05)
        src_block = _block(0, [0.3, 0.3, 0.82 + 0.05 + 0.025], height=0.05)
        scene_after = [src_block, tgt_block]
        trace = [{"fn": "pick_and_place", "args": [0, 1], "result": True}]
        success, _ = verify_episode(trace, scene_after, use_stub=False)
        assert success is True

    def test_stacking_z_off_fails(self):
        tgt_block = _block(1, [0.3, 0.3, 0.82], height=0.05)
        # z is wrong — too high
        bad_z = 0.82 + 0.05 + 0.025 + STACK_TOLERANCE + 0.01
        src_block = _block(0, [0.3, 0.3, bad_z], height=0.05)
        scene_after = [src_block, tgt_block]
        trace = [{"fn": "pick_and_place", "args": [0, 1], "result": True}]
        success, reason = verify_episode(trace, scene_after, use_stub=False)
        assert success is False
        assert "z off" in reason

    def test_missing_source_block_fails(self):
        target = [0.4, 0.3, 0.82]
        scene_after = []  # block 99 not in scene
        trace = [{"fn": "pick_and_place", "args": [99, target], "result": True}]
        success, reason = verify_episode(trace, scene_after, use_stub=False)
        assert success is False
        assert "99" in reason

    def test_missing_target_stack_block_fails(self):
        src_block = _block(0, [0.3, 0.3, 0.90])
        scene_after = [src_block]  # target block 5 missing
        trace = [{"fn": "pick_and_place", "args": [0, 5], "result": True}]
        success, reason = verify_episode(trace, scene_after, use_stub=False)
        assert success is False
        assert "5" in reason
