"""Unit tests for taskb/sandbox.py."""
import pytest
import numpy as np

from taskb.sandbox import SafetyError, check_safety, run_code


# ── check_safety: syntax errors ──────────────────────────────────────────────

class TestCheckSafetySyntax:
    def test_valid_code_passes(self):
        check_safety("x = 1 + 2")

    def test_syntax_error_raised(self):
        with pytest.raises(SyntaxError):
            check_safety("def foo(:\n  pass")

    def test_empty_code_passes(self):
        check_safety("")


# ── check_safety: blocked imports ────────────────────────────────────────────

class TestCheckSafetyImports:
    def test_import_blocked(self):
        with pytest.raises(SafetyError, match="Imports are not allowed"):
            check_safety("import os")

    def test_from_import_blocked(self):
        with pytest.raises(SafetyError, match="Imports are not allowed"):
            check_safety("from os import path")

    def test_import_sys_blocked(self):
        with pytest.raises(SafetyError):
            check_safety("import sys")

    def test_import_subprocess_blocked(self):
        with pytest.raises(SafetyError):
            check_safety("import subprocess")


# ── check_safety: blocked calls ──────────────────────────────────────────────

class TestCheckSafetyBlockedCalls:
    def test_eval_blocked(self):
        with pytest.raises(SafetyError, match="Blocked function call"):
            check_safety("eval('1+1')")

    def test_exec_blocked(self):
        with pytest.raises(SafetyError, match="Blocked function call"):
            check_safety("exec('x=1')")

    def test_open_blocked(self):
        with pytest.raises(SafetyError, match="Blocked function call"):
            check_safety("open('/etc/passwd')")

    def test_compile_blocked(self):
        with pytest.raises(SafetyError, match="Blocked function call"):
            check_safety("compile('x=1', '', 'exec')")

    def test_dunder_import_blocked(self):
        with pytest.raises(SafetyError, match="Blocked function call"):
            check_safety("__import__('os')")

    def test_allowed_builtins_pass(self):
        check_safety("x = len([1,2,3])\ny = min(1,2)\nz = max(3,4)")

    def test_allowed_api_calls_pass(self):
        check_safety(
            "scene = get_scene_state()\n"
            "pick_and_place(0, [0.1, 0.2, 0.82])\n"
            "say('hello')"
        )


# ── run_code: basic execution ─────────────────────────────────────────────────

class TestRunCode:
    def _make_env(self):
        calls = []

        def fake_scene():
            return [{"id": 0, "color": "red", "size": "large",
                     "pos": [0.1, 0.2, 0.82], "height": 0.05}]

        def fake_pnp(src_id, target):
            calls.append((src_id, target))
            return True

        env_api = {
            "get_scene_state": fake_scene,
            "pick_and_place": fake_pnp,
            "say": print,
        }
        return env_api, calls

    def test_success_flag(self):
        env_api, _ = self._make_env()
        result = run_code("x = 1 + 1", env_api)
        assert result["success"] is True
        assert result["error"] is None

    def test_call_trace_captured(self):
        env_api, _ = self._make_env()
        code = (
            'scene = get_scene_state()\n'
            'pick_and_place(scene[0]["id"], [0.4, 0.3, 0.82])\n'
        )
        result = run_code(code, env_api)
        assert result["success"] is True
        fns = [e["fn"] for e in result["call_trace"]]
        assert "get_scene_state" in fns
        assert "pick_and_place" in fns

    def test_call_trace_args_recorded(self):
        env_api, _ = self._make_env()
        code = 'pick_and_place(0, [0.1, 0.2, 0.82])\n'
        result = run_code(code, env_api)
        pnp = next(e for e in result["call_trace"] if e["fn"] == "pick_and_place")
        assert pnp["args"][0] == 0
        assert pnp["result"] is True

    def test_runtime_error_caught(self):
        env_api, _ = self._make_env()
        result = run_code("x = 1 / 0", env_api)
        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]

    def test_name_error_caught(self):
        env_api, _ = self._make_env()
        result = run_code("undefined_func()", env_api)
        assert result["success"] is False

    def test_blocked_import_fails(self):
        env_api, _ = self._make_env()
        result = run_code("import os", env_api)
        assert result["success"] is False
        assert result["error"] is not None

    def test_blocked_eval_fails(self):
        env_api, _ = self._make_env()
        result = run_code("eval('1+1')", env_api)
        assert result["success"] is False

    def test_np_available(self):
        env_api, _ = self._make_env()
        result = run_code("arr = np.array([1, 2, 3])", env_api)
        assert result["success"] is True

    def test_empty_call_trace_on_no_calls(self):
        env_api, _ = self._make_env()
        result = run_code("x = 42", env_api)
        assert result["call_trace"] == []

    def test_multiple_pnp_calls_all_traced(self):
        env_api, _ = self._make_env()
        code = (
            'scene = get_scene_state()\n'
            'pick_and_place(0, [0.1, 0.2, 0.82])\n'
            'pick_and_place(0, [0.3, 0.3, 0.82])\n'
        )
        result = run_code(code, env_api)
        pnp_calls = [e for e in result["call_trace"] if e["fn"] == "pick_and_place"]
        assert len(pnp_calls) == 2
