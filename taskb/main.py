"""CLI entry point: natural-language instruction → run episode."""
import argparse
import copy
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _get_api(use_stub: bool = None):
    """Return (env_api dict, use_stub bool, reset callable)."""
    try:
        from taska.env import (
            get_scene_state,
            get_workspace_bounds,
            pick_and_place,
            reset_env,
            using_stub_fallback,
        )
        _use_stub = using_stub_fallback()
    except ImportError:
        from taskb.stubs import get_scene_state, get_workspace_bounds, pick_and_place
        from taskb.stubs import reset_scene as reset_env
        _use_stub = True

    if use_stub is not None:
        if use_stub:
            from taskb.stubs import get_scene_state, get_workspace_bounds, pick_and_place
            from taskb.stubs import reset_scene as reset_env
        else:
            from taska.env import get_scene_state, get_workspace_bounds, pick_and_place, reset_env
        _use_stub = use_stub

    from taskb.spatial import (
        get_corner_pos,
        get_midpoint,
        get_point_offset,
        get_side_pos,
        make_circle_positions,
        make_line_positions,
    )
    from taskb.say import say

    return {
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
    }, _use_stub, reset_env


def run_instruction(instruction: str, category: str = "?") -> bool:
    """Run a single natural-language instruction through the full pipeline. Returns success bool."""
    from taskb.evaluator import verify_episode
    from taskb.llm import CodeGenerationError, generate_code
    from taskb.logger import log_episode
    from taskb.sandbox import run_code

    env_api, use_stub, reset_env = _get_api()
    reset_env()
    get_scene_state = env_api["get_scene_state"]
    scene_before = copy.deepcopy(get_scene_state())

    print(f"\n>>> Instruction: {instruction}")

    # Generate code
    try:
        code = generate_code(instruction)
    except CodeGenerationError as exc:
        print(f"[ERROR] Code generation failed: {exc}")
        log_episode(
            instruction=instruction,
            category=category,
            generated_code="",
            parse_ok=False,
            safety_ok=False,
            call_trace=[],
            scene_before=scene_before,
            scene_after=[],
            success=False,
            failure_reason=str(exc),
        )
        return False

    print("\n--- Generated code ---")
    print(code)
    print("----------------------\n")

    # Execute
    result = run_code(code, env_api)
    scene_after = copy.deepcopy(get_scene_state())

    if not result["success"]:
        print(f"[ERROR] Execution failed: {result['error']}")
        success, failure_reason = False, result["error"]
    else:
        success, failure_reason = verify_episode(result["call_trace"], scene_after, use_stub=use_stub)

    if success:
        print("[SUCCESS]")
    else:
        print(f"[FAILURE] {failure_reason}")

    log_episode(
        instruction=instruction,
        category=category,
        generated_code=code,
        parse_ok=True,
        safety_ok=True,
        call_trace=result["call_trace"],
        scene_before=scene_before,
        scene_after=scene_after,
        success=success,
        failure_reason=failure_reason,
    )
    return success


def main():
    parser = argparse.ArgumentParser(description="Task B: LLM-powered tabletop robot controller")
    subparsers = parser.add_subparsers(dest="command")

    # run sub-command
    run_p = subparsers.add_parser("run", help="Execute a single instruction.")
    run_p.add_argument("instruction", type=str, help="Natural-language instruction.")
    run_p.add_argument("--category", type=str, default="?", help="Category label (optional).")

    # eval sub-command
    eval_p = subparsers.add_parser("eval", help="Run batch evaluation.")
    eval_p.add_argument("--n", type=int, default=None, help="Number of random episodes to sample.")

    args = parser.parse_args()

    if args.command == "run":
        success = run_instruction(args.instruction, category=args.category)
        sys.exit(0 if success else 1)

    elif args.command == "eval":
        from taskb.eval_runner import run_eval
        run_eval(n_episodes=args.n)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
