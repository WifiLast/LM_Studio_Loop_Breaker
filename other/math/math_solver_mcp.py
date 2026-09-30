from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
from fastmcp import FastMCP
from scipy import optimize

mcp = FastMCP("Equation Solver MCP 🧮")
SERVER_HOST = os.getenv("MATH_MCP_HOST", "10.0.0.10")
SERVER_PORT = int(os.getenv("MATH_MCP_PORT", "2001"))
SERVER_PATH = os.getenv("MATH_MCP_PATH", "/solve")

_SAFE_GLOBALS: Dict[str, object] = {
    "np": np,
    "pi": np.pi,
    "e": np.e,
}

for _name in dir(np):
    if not _name.startswith("_"):
        _SAFE_GLOBALS.setdefault(_name, getattr(np, _name))


def _load_z3_module():
    """Load the external z3-solver package without being shadowed by ./z3."""
    script_dir = Path(__file__).resolve().parent
    removed_paths: list[str] = []

    for candidate in ("", str(script_dir)):
        while candidate in sys.path:
            sys.path.remove(candidate)
            removed_paths.append(candidate)

    sys.modules.pop("z3", None)

    try:
        z3_module = importlib.import_module("z3")
    finally:
        for path_entry in reversed(removed_paths):
            sys.path.insert(0, path_entry)

    if not hasattr(z3_module, "Solver"):
        raise ImportError(
            "Imported module 'z3' does not expose Solver(). "
            "The local ./z3 directory is likely shadowing the z3-solver package."
        )

    return z3_module


_z3 = _load_z3_module()
_Z3_GLOBALS: Dict[str, object] = {
    name: getattr(_z3, name)
    for name in dir(_z3)
    if not name.startswith("_")
}
_Z3_RESERVED_NAMES = set(_Z3_GLOBALS) | {"And", "Or", "Not", "If", "True", "False"}


# --------------------------------------------------------------------------- #
# Numeric root finding (single-variable equations)
# --------------------------------------------------------------------------- #
def _build_callable(expression: str, variable: str) -> Callable[[float], float]:
    """Compile expression into a callable f(x) with a restricted namespace."""
    try:
        compiled = compile(expression, "<expression>", "eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {expression}") from exc

    def func(value: float) -> float:
        local_env = {variable: value}
        return float(eval(compiled, _SAFE_GLOBALS, local_env))

    return func


def _normalize_equation_expression(expression: str) -> str:
    """Normalize `lhs = rhs` into `lhs - (rhs)` so the numeric solver can find a root."""
    normalized = expression.strip()
    if "=" in normalized and "==" not in normalized and "<=" not in normalized and ">=" not in normalized:
        left, right = normalized.split("=", 1)
        return f"({left.strip()}) - ({right.strip()})"
    return normalized


def _has_sign_change(func: Callable[[float], float], start: float, end: float) -> bool:
    """Return True if the function takes both signs across a wide sampled range.

    Distinguishes "the solver failed to converge" from "there is no real root
    because the expression never crosses zero".
    """
    a, b = (start, end) if start <= end else (end, start)
    span = max(abs(b - a), 1.0)
    lo, hi = a - 100.0 * span, b + 100.0 * span
    saw_positive = saw_negative = False
    for point in np.linspace(lo, hi, 1001):
        try:
            value = float(func(float(point)))
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if not np.isfinite(value):
            continue
        if abs(value) < 1e-12:
            return True
        if value > 0:
            saw_positive = True
        else:
            saw_negative = True
        if saw_positive and saw_negative:
            return True
    return False


def _find_real_roots(
    func: Callable[[float], float],
    start: float,
    end: float,
    samples: int = 1001,
) -> List[float]:
    """Find all real roots of `func` in [start, end] by scanning for sign changes.

    Each sign-changing subinterval is refined with Brent's method; exact zeros hit
    while sampling are captured directly. Roots are de-duplicated.
    """
    a, b = (start, end) if start <= end else (end, start)
    if a == b:
        b = a + 1.0
    grid = np.linspace(a, b, samples)
    roots: List[float] = []

    def remember(value: float) -> None:
        if all(abs(value - existing) > 1e-7 for existing in roots):
            roots.append(float(value))

    prev_x = float(grid[0])
    try:
        prev_y = float(func(prev_x))
    except (ValueError, ZeroDivisionError, OverflowError):
        prev_y = float("nan")
    if np.isfinite(prev_y) and abs(prev_y) < 1e-10:
        remember(prev_x)

    for point in grid[1:]:
        cur_x = float(point)
        try:
            cur_y = float(func(cur_x))
        except (ValueError, ZeroDivisionError, OverflowError):
            prev_x, prev_y = cur_x, float("nan")
            continue
        if np.isfinite(cur_y) and abs(cur_y) < 1e-10:
            remember(cur_x)
        elif np.isfinite(prev_y) and np.isfinite(cur_y) and prev_y * cur_y < 0:
            try:
                result = optimize.root_scalar(func, bracket=[prev_x, cur_x], method="brentq")
                if result.converged:
                    remember(result.root)
            except (ValueError, RuntimeError):
                pass
        prev_x, prev_y = cur_x, cur_y

    return sorted(roots)


def _solve_single_variable(
    expression: str,
    bracket_start: float,
    bracket_end: float,
    variable: str,
) -> Dict[str, object]:
    normalized = _normalize_equation_expression(expression)
    func = _build_callable(normalized, variable)

    roots = _find_real_roots(func, bracket_start, bracket_end)
    if roots:
        return {
            "status": "solved",
            "kind": "equation",
            "variable": variable,
            "roots": roots,
            "root": roots[0],
            "count": len(roots),
        }

    if not _has_sign_change(func, bracket_start, bracket_end):
        return {
            "status": "no_real_root",
            "kind": "equation",
            "reason": (
                "The expression does not appear to cross zero, so it has no real root in "
                "the searched range. A strictly positive/negative expression (e.g. a sum of "
                "even powers plus a constant) has no real solution; widening the bracket will "
                "not help. If you meant `expression = c`, write it that way."
            ),
        }
    return {
        "status": "not_found",
        "kind": "equation",
        "reason": "No root located in the bracket. Try widening or shifting the bracket.",
    }


# --------------------------------------------------------------------------- #
# Symbolic solving via Z3 (inequalities and systems)
# --------------------------------------------------------------------------- #
def _extract_symbol_names(expression: str) -> list[str]:
    names = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expression))
    return sorted(name for name in names if name not in _Z3_RESERVED_NAMES)


def _split_top_level(text: str, separators: str = ",;") -> list[str]:
    """Split `text` on top-level separators, ignoring those inside parentheses/brackets."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if depth == 0 and char in separators:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _normalize_constraint(expression: str) -> str:
    """Turn a single `=` (not `==`/`<=`/`>=`) into `==` for Z3."""
    normalized = expression.strip()
    if "=" in normalized and "==" not in normalized and "<=" not in normalized and ">=" not in normalized:
        left, right = normalized.split("=", 1)
        return f"({left.strip()}) == ({right.strip()})"
    return normalized


def _build_z3_variables(
    expressions: Sequence[str],
    vars: Dict[str, str] | None,
) -> Dict[str, object]:
    declared = {k: v.strip().lower() for k, v in (vars or {}).items()}
    variables: Dict[str, object] = {}
    for expression in expressions:
        for name in _extract_symbol_names(expression):
            if name in variables:
                continue
            kind = declared.get(name, "real")
            if kind == "int":
                variables[name] = _z3.Int(name)
            elif kind == "bool":
                variables[name] = _z3.Bool(name)
            else:
                variables[name] = _z3.Real(name)
    return variables


def _solve_constraint_system(
    constraints: List[str],
    vars: Dict[str, str] | None,
) -> Dict[str, object]:
    normalized = [_normalize_constraint(c) for c in constraints]
    variables = _build_z3_variables(normalized, vars)
    eval_env = dict(_Z3_GLOBALS)
    eval_env.update(variables)

    solver = _z3.Solver()
    try:
        for constraint in normalized:
            solver.add(eval(constraint, {"__builtins__": {}}, eval_env))
    except Exception as exc:
        return {
            "status": "invalid",
            "kind": "system",
            "reason": f"Unable to parse constraints: {exc}",
        }

    result = solver.check()
    if result == _z3.sat:
        model = solver.model()
        assignment = {
            name: str(model.eval(var, model_completion=True))
            for name, var in variables.items()
        }
        return {
            "status": "sat",
            "kind": "system",
            "model": assignment,
            "constraints": list(constraints),
        }
    if result == _z3.unsat:
        return {
            "status": "unsat",
            "kind": "system",
            "reason": "The constraints are inconsistent; no assignment satisfies them all.",
            "constraints": list(constraints),
        }
    return {
        "status": "unknown",
        "kind": "system",
        "reason": "Z3 could not decide. Try simplifying the constraints.",
        "constraints": list(constraints),
    }


# --------------------------------------------------------------------------- #
# Matrix / linear-system solving
# --------------------------------------------------------------------------- #
def _coerce_matrix(value: object) -> np.ndarray:
    """Parse a matrix/vector given as a nested list or JSON text into a float array."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty matrix input")
        value = json.loads(text)  # may raise; caller turns it into an 'invalid' status
    arr = np.asarray(value, dtype=float)
    return arr


def _round_list(arr: np.ndarray, ndigits: int = 12) -> object:
    """Return a JSON-friendly, rounded nested list."""
    return np.round(arr, ndigits).tolist()


@mcp.tool
def solve_matrix_equation(
    A: List[List[float]] | str,
    b: List[float] | List[List[float]] | str,
) -> Dict[str, object]:
    """Solve a linear matrix equation A x = b.

    `A` is an m×n coefficient matrix and `b` is the right-hand side (a length-m vector,
    or an m×k matrix for several right-hand sides at once). Both may be passed as nested
    lists or as JSON text, e.g. A='[[2,1],[1,3]]', b='[3,5]'.

    The system is classified with the rank (Rouché–Capelli) test:

    * **unique**   — one exact solution (square full-rank, or consistent overdetermined).
    * **infinite** — consistent but underdetermined; a least-norm particular solution is
      returned along with the number of free variables.
    * **none**     — inconsistent (rank(A) < rank([A|b])); a least-squares best fit is
      still reported.

    The residual ||A x - b|| is always included.

    Examples:
    >>> solve_matrix_equation([[2, 1], [1, 3]], [3, 5])['solution']
    [0.8, 1.4]
    >>> solve_matrix_equation([[1, 1], [1, 1]], [2, 5])['status']
    'none'
    """
    try:
        A_arr = _coerce_matrix(A)
        b_arr = _coerce_matrix(b)
    except Exception as exc:
        return {
            "status": "invalid",
            "kind": "matrix",
            "reason": f"Could not parse A or b: {exc}",
        }

    if A_arr.ndim != 2:
        return {"status": "invalid", "kind": "matrix", "reason": "A must be a 2-D matrix."}
    m, n = A_arr.shape
    if b_arr.ndim == 1:
        if b_arr.shape[0] != m:
            return {
                "status": "invalid",
                "kind": "matrix",
                "reason": f"Length of b ({b_arr.shape[0]}) must match rows of A ({m}).",
            }
    elif b_arr.ndim == 2:
        if b_arr.shape[0] != m:
            return {
                "status": "invalid",
                "kind": "matrix",
                "reason": f"Rows of b ({b_arr.shape[0]}) must match rows of A ({m}).",
            }
    else:
        return {"status": "invalid", "kind": "matrix", "reason": "b must be a vector or matrix."}

    # Classify with the rank test (use the vector case for free-variable counting).
    rank_A = int(np.linalg.matrix_rank(A_arr))
    augmented = np.column_stack([A_arr, b_arr])
    rank_aug = int(np.linalg.matrix_rank(augmented))

    payload: Dict[str, object] = {
        "kind": "matrix",
        "shape": [m, n],
        "rank": rank_A,
    }
    if m == n:
        payload["determinant"] = float(np.linalg.det(A_arr))

    # Solve: exact when square & full rank, otherwise least-squares (min-norm).
    if m == n and rank_A == n:
        x = np.linalg.solve(A_arr, b_arr)
        method = "exact"
    else:
        x, _residuals, _rank, _sv = np.linalg.lstsq(A_arr, b_arr, rcond=None)
        method = "least_squares"

    residual = float(np.linalg.norm(A_arr @ x - b_arr))
    payload["solution"] = _round_list(x)
    payload["residual"] = residual
    payload["method"] = method

    if rank_A < rank_aug:
        payload["status"] = "none"
        payload["reason"] = (
            "The system is inconsistent (rank(A) < rank([A|b])); no exact solution exists. "
            "A least-squares best fit is returned."
        )
    elif rank_A < n:
        payload["status"] = "infinite"
        payload["free_variables"] = n - rank_A
        payload["reason"] = (
            f"The system is underdetermined with {n - rank_A} free variable(s); infinitely "
            "many solutions exist. A least-norm particular solution is returned."
        )
    else:
        payload["status"] = "unique"

    return payload


# --------------------------------------------------------------------------- #
# The single universal tool
# --------------------------------------------------------------------------- #
@mcp.tool
def solve_equation(
    expression: str,
    bracket_start: float = -10.0,
    bracket_end: float = 10.0,
    variable: str = "x",
    vars: Dict[str, str] | str | None = None,
) -> Dict[str, object]:
    """Universal equation solver.

    Two modes, selected automatically:

    * **Single equation in one variable** (no `<`/`>`, no comma) — finds the real roots
      numerically over `[bracket_start, bracket_end]`, returning every root found.
      Supports arithmetic and `np.*` functions, and accepts either a bare expression
      (`x**2 - 9`) or an equation (`x + 6 = 11`, `np.cos(x) = x`).
    * **Inequalities or systems** — pass several constraints in one string, comma or
      semicolon separated, e.g. `'x + y - 1 = 20, x > 1, y > 5'`. These go to Z3, which
      returns a satisfying assignment or reports the system unsatisfiable. `=` is read as
      equality; `vars` optionally pins a name to `Real`, `Int`, or `Bool`.

    Examples:
    >>> solve_equation('x**2 - 9', bracket_start=-5, bracket_end=5)['roots']
    [-3.0, 3.0]
    >>> solve_equation('x + 6 = 11')['root']
    5.0
    >>> solve_equation('x + y - 1 = 20, x > 1, y > 5')['status']
    'sat'
    """
    # The chat template serializes a dict argument as JSON text; accept that too.
    if isinstance(vars, str):
        try:
            parsed = json.loads(vars) if vars.strip() else None
            vars = parsed if isinstance(parsed, dict) else None
        except Exception:
            vars = None

    parts = _split_top_level(expression)
    has_inequality = any(re.search(r"[<>]", part) for part in parts)

    if len(parts) <= 1 and not has_inequality:
        return _solve_single_variable(
            parts[0] if parts else expression,
            bracket_start,
            bracket_end,
            variable,
        )

    return _solve_constraint_system(parts, vars)


def run_server(transport: str | None = None) -> None:
    """Start the FastMCP server using the selected transport."""
    selected_transport = transport or os.getenv("MATH_MCP_TRANSPORT", "streamable-http")
    if selected_transport in {"streamable-http", "http"}:
        mcp.run(transport="http", path=SERVER_PATH, host=SERVER_HOST, port=SERVER_PORT)
    else:
        mcp.run(transport=selected_transport)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Universal equation solver MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http"],
        default=os.getenv("MATH_MCP_TRANSPORT", "streamable-http"),
        help="Transport to use for FastMCP (default: streamable-http)",
    )
    args = parser.parse_args(argv)
    run_server(args.transport)


if __name__ == "__main__":
    main()
