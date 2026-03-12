"""Batch evaluation harness for Task B."""
import copy
import logging
import random

from taskb.evaluator import verify_episode
from taskb.instructions.categories import CATEGORY_LABELS, Category
from taskb.instructions.dataset import INSTRUCTIONS
from taskb.llm import CodeGenerationError, generate_code
from taskb.logger import log_episode
from taskb.sandbox import SafetyError, run_code

logger = logging.getLogger("taskb.eval_runner")


def _build_env_api():
    """Return the env_api dict for the sandbox."""
    try:
        from taska.env import (
            get_scene_state,
            get_workspace_bounds,
            pick_and_place,
            reset_env,
            using_stub_fallback,
        )
        use_stub = using_stub_fallback()
    except ImportError:
        from taskb.stubs import get_scene_state, get_workspace_bounds, pick_and_place
        from taskb.stubs import reset_scene as reset_env
        use_stub = True

    from taskb.spatial import (
        get_corner_pos,
        get_midpoint,
        get_point_offset,
        get_side_pos,
        make_circle_positions,
        make_line_positions,
    )
    from taskb.say import say

    env_api = {
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
    return env_api, use_stub, reset_env


def run_episode(item: dict) -> dict:
    """Run a single instruction through the full pipeline. Returns result dict."""
    instruction = item["instruction"]
    category = item["category"]

    # Snapshot scene before
    env_api, use_stub, reset_env = _build_env_api()
    reset_env()
    get_scene_state = env_api["get_scene_state"]
    scene_before = copy.deepcopy(get_scene_state())

    generated_code = ""
    parse_ok = False
    safety_ok = False
    call_trace = []
    success = False
    failure_reason = None

    try:
        generated_code = generate_code(instruction)
        parse_ok = True
        safety_ok = True  # generate_code performs check_safety internally
    except CodeGenerationError as exc:
        failure_reason = str(exc)
        log_episode(
            instruction=instruction,
            category=category,
            generated_code=generated_code,
            parse_ok=parse_ok,
            safety_ok=safety_ok,
            call_trace=call_trace,
            scene_before=scene_before,
            scene_after=[],
            success=False,
            failure_reason=failure_reason,
        )
        return {"instruction": instruction, "category": category, "success": False, "failure_reason": failure_reason}

    result = run_code(generated_code, env_api)
    call_trace = result["call_trace"]

    scene_after = copy.deepcopy(get_scene_state())

    if not result["success"]:
        failure_reason = result["error"]
    else:
        success, failure_reason = verify_episode(call_trace, scene_after, use_stub=use_stub)

    log_episode(
        instruction=instruction,
        category=category,
        generated_code=generated_code,
        parse_ok=parse_ok,
        safety_ok=safety_ok,
        call_trace=call_trace,
        scene_before=scene_before,
        scene_after=scene_after,
        success=success,
        failure_reason=failure_reason,
    )

    return {"instruction": instruction, "category": category, "success": success, "failure_reason": failure_reason}


def run_eval(instructions: list[dict] | None = None, n_episodes: int | None = None) -> dict:
    """
    Run all instructions (or a random sample of n_episodes) through the pipeline.
    Returns a results dict with per-category and overall success rates.
    """
    if instructions is None:
        instructions = INSTRUCTIONS

    if n_episodes is not None and n_episodes < len(instructions):
        instructions = random.sample(instructions, n_episodes)

    per_category: dict[str, dict] = {}
    for cat in Category:
        per_category[cat.value] = {"n": 0, "success": 0}

    for item in instructions:
        result = run_episode(item)
        cat = result["category"]
        if cat not in per_category:
            per_category[cat] = {"n": 0, "success": 0}
        per_category[cat]["n"] += 1
        if result["success"]:
            per_category[cat]["success"] += 1

    total_n = sum(v["n"] for v in per_category.values())
    total_success = sum(v["success"] for v in per_category.values())

    # Print summary table
    print(f"\n{'Category':<22} | {'N':>4} | {'Success':>7} | {'Rate':>6}")
    print("-" * 22 + "-+-" + "-" * 4 + "-+-" + "-" * 7 + "-+-" + "-" * 6)
    for cat in Category:
        d = per_category[cat.value]
        label = f"{cat.value} ({CATEGORY_LABELS[cat]})"
        rate = d["success"] / d["n"] * 100 if d["n"] > 0 else 0.0
        print(f"{label:<22} | {d['n']:>4} | {d['success']:>7} | {rate:>5.1f}%")
    print("-" * 22 + "-+-" + "-" * 4 + "-+-" + "-" * 7 + "-+-" + "-" * 6)
    overall_rate = total_success / total_n * 100 if total_n > 0 else 0.0
    print(f"{'Total':<22} | {total_n:>4} | {total_success:>7} | {overall_rate:>5.1f}%")
    print()

    return {
        "per_category": per_category,
        "total_n": total_n,
        "total_success": total_success,
        "overall_rate": overall_rate,
    }
