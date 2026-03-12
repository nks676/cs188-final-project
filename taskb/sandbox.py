"""AST safety checker and exec() runner for LLM-generated robot code."""
import ast
import numpy as np


ALLOWED_CALLS = {
    "get_scene_state", "get_workspace_bounds", "pick_and_place",
    "get_corner_pos", "get_side_pos", "get_midpoint",
    "get_point_offset", "make_line_positions", "make_circle_positions",
    "say", "np", "numpy", "print", "len", "range", "zip",
    "enumerate", "sorted", "min", "max", "sum", "abs", "next",
    "list", "dict", "int", "float", "str", "bool",
}

BLOCKED_CALLS = {
    "exec", "eval", "compile", "__import__", "open",
    "os", "sys", "subprocess", "importlib",
}


class SafetyError(Exception):
    pass


class _SafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node):
        raise SafetyError("Imports are not allowed in generated code.")

    def visit_ImportFrom(self, node):
        raise SafetyError("Imports are not allowed in generated code.")

    def visit_Call(self, node):
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # e.g. np.array — check the root object
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                name = root.id

        if name and name in BLOCKED_CALLS:
            raise SafetyError(f"Blocked function call: {name!r}")

        self.generic_visit(node)


def check_safety(code: str) -> None:
    """
    Parse code and walk AST for safety violations.
    Raises SyntaxError or SafetyError on problems.
    """
    tree = ast.parse(code)
    _SafetyVisitor().visit(tree)


def run_code(code: str, env_api: dict) -> dict:
    """
    Execute code in a restricted sandbox.

    env_api: dict mapping function names to callables.
    Returns: {"success": bool, "error": str | None, "call_trace": list}
    """
    call_trace = []

    def _wrap(fn_name, fn):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            call_trace.append({"fn": fn_name, "args": list(args), "result": result})
            return result
        return wrapper

    wrapped_api = {name: _wrap(name, fn) for name, fn in env_api.items()}
    globals_dict = {"np": np, **wrapped_api}

    try:
        check_safety(code)
        exec(code, globals_dict)  # noqa: S102
        return {"success": True, "error": None, "call_trace": call_trace}
    except (SafetyError, SyntaxError) as exc:
        return {"success": False, "error": str(exc), "call_trace": call_trace}
    except Exception as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}", "call_trace": call_trace}
