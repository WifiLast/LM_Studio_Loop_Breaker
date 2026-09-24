from __future__ import annotations

import argparse
import ast
import builtins
import importlib
import itertools
import keyword
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from fastmcp import FastMCP
from scipy import integrate, optimize, stats
import z3

USAGE_EXAMPLES: Dict[str, List[Dict[str, str]]] = {
    "evaluate": [
        {
            "call": "evaluate('5 * 2')",
            "expected_result": "{'ok': True, 'value': 10.0}",
            "description": "Compute a plain arithmetic expression directly.",
        },
        {
            "call": "evaluate('2 + 3 * 4')",
            "expected_result": "{'ok': True, 'value': 14.0}",
            "description": "Evaluate operator precedence without root finding.",
        },
    ],
    "check_equation": [
        {
            "call": "check_equation('5 * 2', '10')",
            "expected_result": "{'ok': True, 'holds': True, 'diff': 0.0}",
            "description": "Validate a simple arithmetic equality.",
        },
        {
            "call": "check_equation('5000 mV', '5 V')",
            "expected_result": "{'ok': True, 'holds': True, 'diff': 0.0}",
            "description": "Normalize units before comparing values.",
        },
    ],
    "check_consistency": [
        {
            "call": "check_consistency(['certified = True', 'in_stock = True'], 'And(certified, in_stock)')",
            "expected_result": "{'ok': True, 'status': 'consistent'}",
            "description": "Check whether a claim is consistent with known facts.",
        },
        {
            "call": "check_consistency(['voltage <= 5'], 'voltage > 5')",
            "expected_result": "{'ok': True, 'status': 'contradicts'}",
            "description": "Detect a contradiction against an authoritative limit.",
        },
    ],
    "check_entailment": [
        {
            "call": "check_entailment(['And(certified, in_stock)'], 'shippable')",
            "expected_result": "{'ok': True, 'status': 'not_entailed'}",
            "description": "Determine whether premises entail a claim.",
        }
    ],
    "verify_claims": [
        {
            "call": "verify_claims([{'kind': 'check_equation', 'lhs': '5 * 2', 'rhs': '10'}])",
            "expected_result": "{'ok': True, 'results': [...] }",
            "description": "Batch several checks in one round-trip.",
        }
    ],
    "solve_equation": [
        {
            "call": "solve_equation('np.cos(x) - x', bracket_start=0.0, bracket_end=1.0)",
            "expected_result": "0.7390851332151607",
            "description": "Finds the fixed point of cos(x).",
        },
        {
            "call": "solve_equation('np.sin(x)', bracket_start=-1.0, bracket_end=1.0)",
            "expected_result": "0.0",
            "description": "Root of sin(x) closest to the origin.",
        },
        {
            "call": "solve_equation('x**2 - 9', bracket_start=0.0, bracket_end=5.0)",
            "expected_result": "3.0",
            "description": "Positive root of x^2 - 9 using Brent's method.",
        },
    ],
    "differentiate": [
        {
            "call": "differentiate('x**3 + 2*x', point=2.0)",
            "expected_result": "14.0",
            "description": "First derivative of a polynomial at x = 2.",
        },
        {
            "call": "differentiate('np.sin(x)', point=0.0)",
            "expected_result": "1.0",
            "description": "Derivative of sin(x) at the origin equals cos(0).",
        },
        {
            "call": "differentiate('np.exp(-x**2)', point=0.0, n=2)",
            "expected_result": "-2.0",
            "description": "Second derivative of the Gaussian at x = 0.",
        },
    ],
    "integrate_function": [
        {
            "call": "integrate_function('np.exp(-x**2)', lower=0.0, upper=1.0)",
            "expected_result": "{'value': 0.7468241328124271, 'abserr': 8.291413475940725e-15}",
            "description": "Area under a Gaussian bell on [0, 1].",
        },
        {
            "call": "integrate_function('np.sin(x)', lower=0.0, upper=np.pi)",
            "expected_result": "{'value': 2.0, 'abserr': 2.220446049250313e-14}",
            "description": "Integral of sin(x) over one half-period.",
        },
        {
            "call": "integrate_function('x**2', lower=-1.0, upper=1.0)",
            "expected_result": "{'value': 0.6666666666666667, 'abserr': 7.401486830834377e-15}",
            "description": "Definite integral of x^2 symmetric around the origin.",
        },
    ],
    "distribution_pdf": [
        {
            "call": "distribution_pdf('norm', x=0.0)",
            "expected_result": "0.3989422804014327",
            "description": "Standard normal density at the origin.",
        },
        {
            "call": "distribution_pdf('binom', x=3, shape_args=[10, 0.5])",
            "expected_result": "0.1171875",
            "description": "Binomial PMF for 10 trials with p=0.5 at k=3.",
        },
    ],
    "distribution_cdf": [
        {
            "call": "distribution_cdf('norm', x=1.96)",
            "expected_result": "0.9750021048517795",
            "description": "Lower-tail probability for Z <= 1.96.",
        },
        {
            "call": "distribution_cdf('binom', x=4, shape_args=[10, 0.5])",
            "expected_result": "0.376953125",
            "description": "Cumulative probability P(X <= 4) for a binomial(10, 0.5).",
        },
    ],
    "distribution_quantile": [
        {
            "call": "distribution_quantile('norm', probability=0.975)",
            "expected_result": "1.959963984540054",
            "description": "0.975 quantile (two-sided 95%) of the standard normal.",
        },
        {
            "call": "distribution_quantile('chi2', probability=0.95, shape_args=[4])",
            "expected_result": "9.487729036781154",
            "description": "0.95 quantile of a chi-square distribution with 4 degrees of freedom.",
        },
    ],
    "distribution_probability_between": [
        {
            "call": "distribution_probability_between('norm', lower=-1.0, upper=1.0)",
            "expected_result": "0.6826894921370859",
            "description": "Probability mass within one standard deviation of the mean.",
        },
        {
            "call": "distribution_probability_between('binom', lower=0, upper=2, shape_args=[5, 0.5])",
            "expected_result": "0.5",
            "description": "Probability of <=2 successes in 5 fair Bernoulli trials.",
        },
    ],
    "z3_solve_constraints": [
        {
            "call": "z3_solve_constraints(['x > 2', 'x < 5'])",
            "expected_result": "{'status': 'sat', 'model': {'x': '3'}}",
            "description": "Finds a satisfying assignment for a simple interval.",
        },
        {
            "call": "z3_solve_constraints(['x + y == 10', 'x > 3', 'y > 2'])",
            "expected_result": "{'status': 'sat', 'model': {'x': '4', 'y': '6'}}",
            "description": "Solves a small linear constraint system.",
        },
    ],
    "z3_prove_theorem": [
        {
            "call": "z3_prove_theorem(['x > 2'], 'x > 1')",
            "expected_result": "{'proved': true, 'status': 'entailed'}",
            "description": "Proves a simple arithmetic entailment (flat expressions only - no quantifiers/custom sorts; use z3_run_script for those).",
        }
    ],
    "z3_run_script": [
        {
            "call": "z3_run_script([\"Object = DeclareSort('Object')\", \"Human = Function('Human', Object, BoolSort())\", \"Mortal = Function('Mortal', Object, BoolSort())\", \"socrates = Const('socrates', Object)\", \"x = Const('x', Object)\", \"solver = Solver()\", \"solver.add(ForAll([x], Implies(Human(x), Mortal(x))))\", \"solver.add(Human(socrates))\", \"solver.add(Not(Mortal(socrates)))\"])",
            "expected_result": "{'ok': True, 'status': 'unsat'}",
            "description": "Proves Socrates is mortal: premises + negated conclusion is unsat, so the conclusion holds. Needs quantifiers/custom sorts, so it uses the general script tool rather than z3_prove_theorem. `itertools` is preloaded (no import needed) for enumerating finite domains, e.g. `for perm in itertools.permutations(roles): ...` when checking a property across every case of a logic puzzle.",
        },
    ],
}


mcp = FastMCP("Math MCP 🧮")
SERVER_HOST = os.getenv("MATH_MCP_HOST", "10.0.0.10")
SERVER_PORT = int(os.getenv("MATH_MCP_PORT", "2000"))
SERVER_PATH = os.getenv("MATH_MCP_PATH", "/math")

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

solver_context: Dict[str, object] = {
    "solver": None,
    "variables": {},
    "constraints": [],
}


def reset_solver_context() -> Dict[str, object]:
    """Reset the persistent Z3 solver context."""
    solver_context["solver"] = _z3.Solver()
    solver_context["variables"] = {}
    solver_context["constraints"] = []
    return solver_context


reset_solver_context()


def numerical_derivative(
    func: Callable[[float], float],
    x0: float,
    dx: float = 1e-6,
    n: int = 1,
    order: int = 5,
) -> float:
    """Numerically approximate the n-th derivative using recursive central differences."""
    if n < 1:
        raise ValueError("n must be a positive integer.")
    if order % 2 == 0:
        raise ValueError("order must be an odd integer for numerical differentiation.")
    if dx <= 0:
        raise ValueError("dx must be positive.")

    if n == 1:
        return float((func(x0 + dx) - func(x0 - dx)) / (2.0 * dx))

    def lower(value: float) -> float:
        return numerical_derivative(func, value, dx=dx, n=n - 1, order=order)

    return float((lower(x0 + dx) - lower(x0 - dx)) / (2.0 * dx))


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
    if (
        "=" in normalized
        and "==" not in normalized
        and "!=" not in normalized
        and "<=" not in normalized
        and ">=" not in normalized
    ):
        left, right = normalized.split("=", 1)
        return f"({left.strip()}) - ({right.strip()})"
    return normalized


def _split_top_level(text: str, separators: str = ",;") -> list[str]:
    """Split `text` on top-level separators, ignoring those inside parentheses/brackets.

    Lets a single argument carry a whole system, e.g. ``"x + y - 1 = 20, x > 1, y > 5"``
    without breaking ``f(x, y)`` apart.
    """
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


def _get_distribution(name: str):
    """Return a SciPy distribution object by name with basic validation."""
    try:
        dist = getattr(stats, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown distribution '{name}'. Refer to scipy.stats for valid names.") from exc
    if not hasattr(dist, "cdf"):
        raise ValueError(f"Distribution '{name}' does not expose the expected SciPy API.")
    return dist


def _distribution_args_kwargs(
    dist,
    shape_args: Optional[Sequence[float]],
    loc: float,
    scale: float,
    extra_params: Optional[Dict[str, float]] = None,
):
    args = tuple(shape_args or [])
    kwargs: Dict[str, float] = dict(extra_params or {})
    if loc != 0.0:
        kwargs.setdefault("loc", loc)
    if scale != 1.0 and hasattr(dist, "pdf"):
        kwargs.setdefault("scale", scale)
    return args, kwargs


def _pmf_or_pdf(dist, x: float, args: Sequence[float], kwargs: Dict[str, float]) -> float:
    if hasattr(dist, "pdf"):
        return float(dist.pdf(x, *args, **kwargs))
    if hasattr(dist, "pmf"):
        return float(dist.pmf(x, *args, **kwargs))
    raise ValueError("Distribution does not provide pdf/pmf evaluation.")


def _try_bracket(func: Callable[[float], float], start: float, end: float) -> tuple[float, float] | None:
    """Return a sign-changing bracket if one exists in [start, end]."""
    if start == end:
        end = start + 1.0

    a, b = (start, end) if start < end else (end, start)
    sample_points = np.linspace(a, b, 25)
    previous_x = float(sample_points[0])
    previous_y = float(func(previous_x))
    if abs(previous_y) < 1e-12:
        return previous_x - 1e-6, previous_x + 1e-6

    for point in sample_points[1:]:
        current_x = float(point)
        current_y = float(func(current_x))
        if abs(current_y) < 1e-12:
            return current_x - 1e-6, current_x + 1e-6
        if previous_y * current_y < 0:
            return previous_x, current_x
        previous_x = current_x
        previous_y = current_y
    return None


def _has_sign_change(func: Callable[[float], float], start: float, end: float) -> bool:
    """Return True if the function takes both signs across a wide sampled range.

    Used to distinguish "the solver failed to converge" from "there is no real root
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


def _find_bracket(func: Callable[[float], float], bracket_start: float, bracket_end: float) -> tuple[float, float] | None:
    """Search for a sign-changing bracket near the provided range."""
    direct = _try_bracket(func, bracket_start, bracket_end)
    if direct is not None:
        return direct

    center = (bracket_start + bracket_end) / 2.0
    width = max(abs(bracket_end - bracket_start), 1.0)
    for multiplier in (2.0, 5.0, 10.0, 25.0, 50.0):
        span = width * multiplier
        candidate = _try_bracket(func, center - span, center + span)
        if candidate is not None:
            return candidate

    for candidate_range in [(-10.0, 10.0), (-100.0, 100.0), (-1000.0, 1000.0)]:
        candidate = _try_bracket(func, candidate_range[0], candidate_range[1])
        if candidate is not None:
            return candidate

    return None


def _extract_symbol_names(expression: str) -> list[str]:
    names = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expression))
    return sorted(
        name
        for name in names
        if name not in _Z3_RESERVED_NAMES and not keyword.iskeyword(name)
    )


class _BoolOpToZ3(ast.NodeTransformer):
    """Rewrite Python's `and` / `or` / `not` into calls to Z3's `And` / `Or` / `Not`.

    Z3's `BoolRef` does not raise on Python's `bool()` the way one might expect, so
    Python's own short-circuit `and`/`or` silently evaluate against a z3 symbolic
    expression's truthiness (unrelated to its logical meaning) and discard one operand
    entirely - e.g. `x == 0 or x == 2` does NOT build `Or(x==0, x==2)`; it silently
    asserts only one of the two comparisons, with no error, and no way to tell from the
    result alone that anything went wrong. This is the single most common way a
    language model's z3 constraint quietly does the wrong thing (see math/z3_usage.log
    for a real trace: half a dozen `"x == 0 Or x == 2"` attempts failed on invalid
    syntax, and the lowercase `"or"` retry that finally returned `ok: true` had in fact
    silently dropped one side of the disjunction).

    Rewriting at the AST level - rather than a text substitution of `and`/`or` to
    `&`/`|` - preserves operator precedence correctly: `&`/`|` bind tighter than `==` in
    Python, so `x == 0 | x == 2` parses as `x == (0 | x) == 2`, not the intended
    disjunction. Transforming the parsed tree sidesteps that trap entirely.
    """

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        func_name = "And" if isinstance(node.op, ast.And) else "Or"
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=node.values,
                keywords=[],
            ),
            node,
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="Not", ctx=ast.Load()),
                    args=[node.operand],
                    keywords=[],
                ),
                node,
            )
        return node


def _compile_logic_expression(expression: str):
    """Parse a claim/constraint expression and rewrite and/or/not into Z3's And/Or/Not
    calls (see `_BoolOpToZ3`) before compiling, so natural lowercase boolean language
    ("a == 1 or a == 2") produces correct Z3 logic instead of silently wrong logic."""
    tree = ast.parse(expression, mode="eval")
    tree = _BoolOpToZ3().visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, "<expression>", "eval")


_INFIX_BOOL_KEYWORD_RE = re.compile(r"\b(And|Or|Not)\b(?!\s*\()")


def _boolean_syntax_hint(expression: str, default: str) -> str:
    """A more specific suggestion when a SyntaxError looks like capitalized And/Or/Not
    used as an infix operator (`a Or b`) - the most common z3 constraint mistake beyond
    the silently-wrong lowercase case `_BoolOpToZ3` already fixes. `Or`/`And`/`Not` only
    exist as Z3 functions, so they must be called with parentheses; as bare words they
    are not valid Python syntax at all, which is what the raw SyntaxError doesn't say
    explicitly."""
    if _INFIX_BOOL_KEYWORD_RE.search(expression):
        return (
            "Use lowercase 'and'/'or'/'not' as infix operators (for example "
            "'a == 1 or a == 2'), or call the capitalized Z3 functions with "
            "parentheses: Or(a, b), And(a, b), Not(a). 'a Or b' without parentheses "
            "is not valid syntax either way."
        )
    return default


def _coerce_model_value(value: object) -> str:
    return str(value)


def _build_z3_eval_env(variables: Dict[str, object] | None = None) -> Dict[str, object]:
    env = dict(_Z3_GLOBALS)
    if variables:
        env.update(variables)
    return env


def _ensure_context_variables(expressions: Sequence[str]) -> Dict[str, object]:
    variables = solver_context["variables"]
    assert isinstance(variables, dict)
    for expression in expressions:
        for name in _extract_symbol_names(expression):
            variables.setdefault(name, _z3.Real(name))
    return variables


def _check_solver_result(solver, variables: Dict[str, object], constraints: Sequence[str]) -> Dict[str, object]:
    result = solver.check()
    status = str(result)
    payload: Dict[str, object] = {
        "status": status,
        "constraints": list(constraints),
    }
    if result == _z3.sat:
        model = solver.model()
        assignments: Dict[str, str] = {}
        for var_name, var in variables.items():
            value = model.eval(var, model_completion=True)
            assignments[var_name] = _coerce_model_value(value)
        payload["model"] = assignments
    return payload


def _create_theorem_context() -> Dict[str, object]:
    return {
        "solver": _z3.Solver(),
        "sorts": {"Object": _z3.DeclareSort("Object")},
    }


def _tool_fn(tool: Callable) -> Callable:
    """Underlying plain callable for an `@mcp.tool`-decorated function.

    Some FastMCP versions wrap the decorated function in an object exposing the
    original callable as `.fn`; others (e.g. 3.2.4, installed here) leave the
    decorated name bound to the plain function. Internal tool-to-tool calls use
    this so they work either way instead of assuming `.fn` exists.
    """
    return getattr(tool, "fn", tool)


def _verdict(
    ok: bool,
    *,
    value: object | None = None,
    status: str | None = None,
    reason: str = "",
    suggestion: str = "",
    **extra: object,
) -> Dict[str, object]:
    """Build a uniform tool verdict with repair hints.

    Supported grammar for claim-shaped expressions: `==`, `!=`, `<=`, `>=`, `<`, `>`,
    `+`, `-`, `*`, `/`, lowercase `and`/`or`/`not`, or `And`/`Or`/`Not`/`Implies` as
    function calls.
    """

    payload: Dict[str, object] = {
        "ok": bool(ok),
        "reason": reason,
        "suggestion": suggestion,
    }
    if value is not None:
        payload["value"] = value
    if status is not None:
        payload["status"] = status
    payload.update(extra)
    return payload


_UNIT_FACTORS: Dict[str, float] = {
    "v": 1.0,
    "mv": 1e-3,
    "kv": 1e3,
    "a": 1.0,
    "ma": 1e-3,
    "w": 1.0,
    "mw": 1e-3,
    "kw": 1e3,
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
    "ohm": 1.0,
    "ω": 1.0,
    "kohm": 1e3,
    "mohm": 1e-3,
    "g": 1e-3,
    "kg": 1.0,
    "mg": 1e-6,
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
}


def normalize_units(value: float, unit: str, target: str = "si") -> Dict[str, object]:
    """Normalize a numeric magnitude into SI-style units.

    Use this when comparing datasheet values like `5000 mV` and `5 V` before feeding the
    result into `check_equation` or `check_consistency`.
    """

    normalized_unit = unit.strip().lower()
    factor = _UNIT_FACTORS.get(normalized_unit)
    if factor is None:
        return _verdict(
            False,
            status="unsupported_unit",
            reason=f"Unsupported unit '{unit}'.",
            suggestion="Use an SI-like unit such as V, A, W, Hz, ohm, g, m, cm, or mm.",
        )
    return _verdict(True, value=float(value) * factor, status="normalized", reason="Unit normalized to SI")


def _normalize_unit_literals(expression: str) -> str:
    """Replace simple numeric unit literals with SI-normalized magnitudes."""

    pattern = re.compile(
        r"(?P<value>\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?P<unit>mV|kV|V|mA|A|mW|kW|W|kHz|MHz|GHz|Hz|ohm|Ω|kohm|mohm|kg|g|mg|cm|mm|m)\b",
        re.IGNORECASE,
    )

    def repl(match: re.Match[str]) -> str:
        value = float(match.group("value"))
        unit = match.group("unit")
        result = normalize_units(value, unit)
        if result.get("ok"):
            normalized = float(result["value"])
            return str(normalized)
        return match.group(0)

    return pattern.sub(repl, expression)


def _safe_eval_numeric(expression: str, variables: Dict[str, object] | None = None) -> tuple[bool, object | None, str, str]:
    """Evaluate an arithmetic expression under a restricted namespace."""

    normalized = _normalize_unit_literals(expression)
    try:
        compiled = compile(normalized, "<expression>", "eval")
    except SyntaxError as exc:
        return False, None, "invalid_expression", f"Check the syntax: {exc.msg}."

    env = dict(_SAFE_GLOBALS)
    if variables:
        env.update(variables)

    try:
        value = eval(compiled, {"__builtins__": {}}, env)
    except Exception as exc:
        return False, None, "evaluation_failed", (
            f"Unable to evaluate the expression: {exc}. "
            "If this is symbolic logic, use check_consistency or check_entailment."
        )

    if isinstance(value, (int, float, np.floating)):
        return True, float(value), "value_computed", "Expression evaluated successfully."
    return True, value, "value_computed", "Expression evaluated successfully."


def _z3_type_from_name(type_name: str | None) -> object:
    normalized = (type_name or "Real").strip().lower()
    if normalized == "bool":
        return "Bool"
    if normalized == "int":
        return "Int"
    return "Real"


def _looks_boolean_context(expression: str, name: str) -> bool:
    """Heuristically determine whether a symbol is used as a Boolean."""

    if re.fullmatch(rf"{re.escape(name)}", expression.strip()):
        return True
    logical_keywords = ("And(", "Or(", "Not(", "Implies(")
    if any(keyword in expression for keyword in logical_keywords):
        if re.search(rf"\b{name}\b", expression):
            return True
    if re.search(rf"\b{name}\b\s*(==|!=)\s*(True|False)\b", expression):
        return True
    if re.search(rf"\b(True|False)\b\s*(==|!=)\s*\b{name}\b", expression):
        return True
    return False


def _build_typed_z3_variables(
    expressions: Sequence[str],
    vars: Dict[str, str] | None = None,
) -> Dict[str, object]:
    """Build typed Z3 symbols for the variables found in the expressions.

    `vars` maps variable names to `Real`, `Int`, or `Bool`.
    """

    variables: Dict[str, object] = {}
    declared = vars or {}
    for expression in expressions:
        for name in _extract_symbol_names(expression):
            if name in _Z3_RESERVED_NAMES:
                continue
            if name in variables:
                continue
            type_name = declared.get(name, "Real")
            if type_name == "Real" and _looks_boolean_context(expression, name):
                type_name = "Bool"
            sort = _z3_type_from_name(type_name)
            if sort == "Bool":
                variables[name] = _z3.Bool(name)
            elif sort == "Int":
                variables[name] = _z3.Int(name)
            else:
                variables[name] = _z3.Real(name)
    return variables


def _structured_model(model: object, variables: Dict[str, object]) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    for var_name, var in variables.items():
        try:
            value = model.eval(var, model_completion=True)
        except Exception:
            continue
        assignments[var_name] = _coerce_model_value(value)
    return assignments


def _normalize_claim_expression(expression: str) -> str:
    normalized = expression.strip()
    if (
        "=" in normalized
        and "==" not in normalized
        and "!=" not in normalized
        and "<=" not in normalized
        and ">=" not in normalized
    ):
        left, right = normalized.split("=", 1)
        return f"({left.strip()}) == ({right.strip()})"
    return normalized


@mcp.tool
def evaluate(expression: str) -> Dict[str, object]:
    """Compute a numeric expression directly.

    Supported grammar: arithmetic `+`, `-`, `*`, `/`, parentheses, and simple unit
    literals such as `5 V`, `5000 mV`, or `2 kHz`.
    """

    ok, value, status, reason = _safe_eval_numeric(expression)
    if not ok:
        return _verdict(False, status=status, reason=reason, suggestion="Fix the arithmetic or use a logical checker.")
    return _verdict(True, value=value, status=status, reason=reason)


@mcp.tool
def check_equation(lhs: str, rhs: str, tol: float = 1e-9) -> Dict[str, object]:
    """Check whether two expressions are equal within tolerance.

    Supported grammar: arithmetic `+`, `-`, `*`, `/`, parentheses, and simple unit
    literals such as `5 V` and `5000 mV`.
    """

    lhs_ok, lhs_value, lhs_status, lhs_reason = _safe_eval_numeric(lhs)
    rhs_ok, rhs_value, rhs_status, rhs_reason = _safe_eval_numeric(rhs)
    if not lhs_ok:
        return _verdict(False, status=lhs_status, reason=lhs_reason, suggestion="Check the left-hand side expression.")
    if not rhs_ok:
        return _verdict(False, status=rhs_status, reason=rhs_reason, suggestion="Check the right-hand side expression.")

    diff = abs(float(lhs_value) - float(rhs_value))
    holds = diff <= tol
    if holds:
        return _verdict(
            True,
            status="holds",
            reason="The expressions agree within tolerance.",
            value={
                "lhs_value": float(lhs_value),
                "rhs_value": float(rhs_value),
                "diff": float(diff),
            },
        )
    return _verdict(
        True,
        status="fails",
        reason="The expressions differ beyond tolerance.",
        suggestion="Normalize units or check for a missing factor, sign, or parenthesis.",
        value={
            "lhs_value": float(lhs_value),
            "rhs_value": float(rhs_value),
            "diff": float(diff),
            "holds": False,
        },
    )


def _solve_root_equation(
    expression: str,
    bracket_start: float = -10.0,
    bracket_end: float = 10.0,
    variable: str = "x",
) -> Dict[str, object]:
    normalized_expression = _normalize_equation_expression(_normalize_unit_literals(expression))
    func = _build_callable(normalized_expression, variable)
    try:
        bracket = _find_bracket(func, bracket_start, bracket_end)
        if bracket is not None:
            result = optimize.root_scalar(func, bracket=list(bracket), method="brentq")
        else:
            x0 = bracket_start
            x1 = bracket_end if bracket_end != bracket_start else bracket_start + 1.0
            result = optimize.root_scalar(func, x0=x0, x1=x1, method="secant")
    except Exception as exc:
        return _verdict(
            False,
            status="solve_failed",
            reason=f"Root finding failed: {exc}",
            suggestion="Widen the bracket, simplify the expression, or check for unit mismatch.",
        )

    if not result.converged:
        if bracket is None and not _has_sign_change(func, bracket_start, bracket_end):
            return _verdict(
                False,
                status="no_root",
                reason="The expression does not appear to cross zero in the searched range.",
                suggestion="Widen the bracket or rewrite the claim as `expression = c`.",
            )
        return _verdict(
            False,
            status="did_not_converge",
            reason="The root solver did not converge.",
            suggestion="Try a different bracket or simplify the expression.",
        )
    return _verdict(True, value=float(result.root), status="root_found", reason="Root found successfully.")

@mcp.tool
def check_consistency(
    facts: List[str],
    claim: str,
    vars: Dict[str, str] | None = None,
) -> Dict[str, object]:
    """Check whether a claim is consistent with known facts.

    Supported grammar: `==`, `!=`, `<=`, `>=`, `<`, `>`, `+`, `-`, `*`, `/`, `Implies`,
    and boolean combinators as lowercase infix `and`/`or`/`not` (preferred - e.g.
    `x == 1 or x == 2`) or the capitalized Z3 functions `And(...)`/`Or(...)`/`Not(...)`
    (NOT as infix - `a Or b` is invalid; call it `Or(a, b)`).
    """

    if not facts:
        return _verdict(False, status="invalid_input", reason="facts must not be empty.", suggestion="Provide at least one fact.")
    if not claim.strip():
        return _verdict(False, status="invalid_input", reason="claim must not be empty.", suggestion="Provide a claim to test.")

    normalized_facts = [_normalize_claim_expression(fact) for fact in facts]
    normalized_claim = _normalize_claim_expression(claim)
    variables = _build_typed_z3_variables([*normalized_facts, normalized_claim], vars=vars)
    eval_env = _build_z3_eval_env(variables)
    solver = _z3.Solver()

    try:
        for fact in normalized_facts:
            solver.add(eval(_compile_logic_expression(fact), {"__builtins__": {}}, eval_env))
        solver.add(eval(_compile_logic_expression(normalized_claim), {"__builtins__": {}}, eval_env))
    except Exception as exc:
        return _verdict(
            False,
            status="invalid_expression",
            reason=f"Unable to parse claim or facts: {exc}",
            suggestion=_boolean_syntax_hint(
                " ".join([*normalized_facts, normalized_claim]),
                "Use arithmetic or Z3-style logic with And/Or/Not/Implies.",
            ),
        )

    result = solver.check()
    if result == _z3.sat:
        model = solver.model()
        return _verdict(
            True,
            status="consistent",
            reason="The claim is consistent with the provided facts.",
            value={
                "model": _structured_model(model, variables),
                "facts": list(facts),
                "claim": claim,
            },
        )

    return _verdict(
        True,
        status="contradicts",
        reason="The claim conflicts with the provided facts.",
        suggestion="Revise the claim or re-check the facts and units.",
        value={
            "conflicting_facts": list(facts),
            "claim": claim,
        },
    )


@mcp.tool
def check_entailment(
    premises: List[str],
    claim: str,
    vars: Dict[str, str] | None = None,
) -> Dict[str, object]:
    """Check whether premises entail a claim.

    Supported grammar: `==`, `!=`, `<=`, `>=`, `<`, `>`, `+`, `-`, `*`, `/`, `Implies`,
    and boolean combinators as lowercase infix `and`/`or`/`not` (preferred - e.g.
    `x == 1 or x == 2`) or the capitalized Z3 functions `And(...)`/`Or(...)`/`Not(...)`
    (NOT as infix - `a Or b` is invalid; call it `Or(a, b)`).
    """

    if not premises:
        return _verdict(False, status="invalid_input", reason="premises must not be empty.", suggestion="Provide at least one premise.")
    if not claim.strip():
        return _verdict(False, status="invalid_input", reason="claim must not be empty.", suggestion="Provide a claim to prove.")

    normalized_premises = [_normalize_claim_expression(premise) for premise in premises]
    normalized_claim = _normalize_claim_expression(claim)
    variables = _build_typed_z3_variables([*normalized_premises, normalized_claim], vars=vars)
    eval_env = _build_z3_eval_env(variables)
    solver = _z3.Solver()

    try:
        for premise in normalized_premises:
            solver.add(eval(_compile_logic_expression(premise), {"__builtins__": {}}, eval_env))
        solver.add(
            _z3.Not(
                eval(_compile_logic_expression(normalized_claim), {"__builtins__": {}}, eval_env)
            )
        )
    except Exception as exc:
        return _verdict(
            False,
            status="invalid_expression",
            reason=f"Unable to parse the premises or claim: {exc}",
            suggestion=_boolean_syntax_hint(
                " ".join([*normalized_premises, normalized_claim]),
                "Use arithmetic or Z3-style logic with And/Or/Not/Implies.",
            ),
        )

    result = solver.check()
    if result == _z3.unsat:
        return _verdict(
            True,
            status="entailed",
            reason="The premises entail the claim.",
            value={"premises": list(premises), "claim": claim},
        )

    counterexample = None
    if result == _z3.sat:
        counterexample = _structured_model(solver.model(), variables)
    return _verdict(
        True,
        status="not_entailed",
        reason="The premises do not entail the claim.",
        suggestion="Add missing premises or relax the claim.",
        value={"counterexample": counterexample, "premises": list(premises), "claim": claim},
    )


@mcp.tool
def solve_equation(
    expression: str,
    bracket_start: float = -10.0,
    bracket_end: float = 10.0,
    variable: str = "x",
    vars: Dict[str, str] | None = None,
) -> Dict[str, object]:
    """Solve an equation, an inequality, or a system of them.

    Two modes, chosen automatically:

    * **Single equation in one variable** (no `<`/`>`, no comma) — finds a real root
      numerically over the bracket, exactly as before. Supports `np.*` functions, e.g.
      `np.cos(x) - x` or `x**2 - 9` or `x + 6 = 11`.
    * **Inequalities or multiple constraints** — pass them comma/semicolon separated in a
      single string, e.g. `'x + y - 1 = 20, x > 1, y > 5'`. These are handed to Z3, which
      returns a satisfying assignment (or reports it unsatisfiable). `=` is read as
      equality; `vars` may pin a variable to `Real`, `Int`, or `Bool`.
    """

    parts = _split_top_level(expression)
    has_inequality = any(re.search(r"[<>]", part) for part in parts)

    if len(parts) <= 1 and not has_inequality:
        return _solve_root_equation(
            parts[0] if parts else expression,
            bracket_start=bracket_start,
            bracket_end=bracket_end,
            variable=variable,
        )

    return _tool_fn(z3_solve_constraints)(parts, vars=vars)


@mcp.tool
def verify_claims(claims: List[Dict[str, object]]) -> Dict[str, object]:
    """Verify a batch of claim-shaped requests in one call.

    Each item should include a `kind` key with one of `evaluate`, `check_equation`,
    `check_consistency`, `check_entailment`, or `solve_equation`.
    Supported grammar: `==`, `!=`, `<=`, `>=`, `<`, `>`, `+`, `-`, `*`, `/`, `Implies`,
    and boolean combinators as lowercase infix `and`/`or`/`not` (preferred - e.g.
    `x == 1 or x == 2`) or the capitalized Z3 functions `And(...)`/`Or(...)`/`Not(...)`
    (NOT as infix - `a Or b` is invalid; call it `Or(a, b)`).
    """

    if not claims:
        return _verdict(False, status="invalid_input", reason="claims must not be empty.", suggestion="Provide at least one claim.")

    results: List[Dict[str, object]] = []
    all_ok = True
    for item in claims:
        kind = str(item.get("kind", "")).strip()
        try:
            if kind == "evaluate":
                result = _tool_fn(evaluate)(str(item.get("expression", "")))
            elif kind == "check_equation":
                result = _tool_fn(check_equation)(
                    str(item.get("lhs", "")),
                    str(item.get("rhs", "")),
                    float(item.get("tol", 1e-9)),
                )
            elif kind == "check_consistency":
                result = _tool_fn(check_consistency)(
                    [str(fact) for fact in item.get("facts", [])],
                    str(item.get("claim", "")),
                    item.get("vars"),
                )
            elif kind == "check_entailment":
                result = _tool_fn(check_entailment)(
                    [str(premise) for premise in item.get("premises", [])],
                    str(item.get("claim", "")),
                    item.get("vars"),
                )
            elif kind == "solve_equation":
                result = _tool_fn(solve_equation)(
                    str(item.get("expression", "")),
                    float(item.get("bracket_start", -10.0)),
                    float(item.get("bracket_end", 10.0)),
                    str(item.get("variable", "x")),
                )
            else:
                result = _verdict(
                    False,
                    status="invalid_input",
                    reason=f"Unsupported claim kind '{kind}'.",
                    suggestion="Use evaluate, check_equation, check_consistency, check_entailment, or solve_equation.",
                )
        except Exception as exc:
            result = _verdict(
                False,
                status="batch_item_failed",
                reason=f"Batch item failed: {exc}",
                suggestion="Check the item structure and required fields.",
            )
        if not result.get("ok", False):
            all_ok = False
        results.append(result)

    return _verdict(
        all_ok,
        status="batch_complete" if all_ok else "batch_partial",
        reason="Batch verification finished.",
        suggestion="Inspect any failed items and retry them after repair.",
        value={"results": results},
    )


@mcp.tool
def z3_solve_constraints(constraints: List[str], vars: Dict[str, str] | None = None) -> Dict[str, object]:
    """Solve one or more symbolic constraints with Z3.

    Supported grammar: `==`, `!=`, `<=`, `>=`, `<`, `>`, `+`, `-`, `*`, `/`, `Implies`,
    and boolean combinators as lowercase infix `and`/`or`/`not` (preferred - e.g.
    `x == 1 or x == 2`) or the capitalized Z3 functions `And(...)`/`Or(...)`/`Not(...)`
    (NOT as infix - `a Or b` is invalid; call it `Or(a, b)`).
    """

    if not constraints:
        return _verdict(False, status="invalid_input", reason="constraints must not be empty.", suggestion="Provide at least one constraint.")

    normalized = [_normalize_claim_expression(constraint) for constraint in constraints]
    variables = _build_typed_z3_variables(normalized, vars=vars)
    eval_env = _build_z3_eval_env(variables)
    solver = _z3.Solver()

    try:
        for constraint in normalized:
            solver.add(eval(_compile_logic_expression(constraint), {"__builtins__": {}}, eval_env))
    except Exception as exc:
        return _verdict(
            False,
            status="invalid_expression",
            reason=f"Unable to parse constraints: {exc}",
            suggestion=_boolean_syntax_hint(
                " ".join(normalized),
                "Use arithmetic or Z3-style logic with And/Or/Not/Implies.",
            ),
        )

    result = solver.check()
    if result == _z3.sat:
        model = solver.model()
        return _verdict(
            True,
            status="sat",
            reason="The constraints are satisfiable.",
            value={"model": _structured_model(model, variables), "constraints": list(constraints)},
        )
    if result == _z3.unsat:
        return _verdict(
            True,
            status="unsat",
            reason="The constraints are inconsistent.",
            suggestion="Relax one constraint or verify the units and bounds.",
            value={"constraints": list(constraints)},
        )
    return _verdict(
        False,
        status="unknown",
        reason="Z3 returned an unknown result.",
        suggestion="Simplify the constraints or add more structure.",
    )


@mcp.tool
def z3_add_constraint(constraint: str) -> Dict[str, object]:
    """Add a symbolic constraint to the persistent Z3 solver context."""
    if not constraint.strip():
        raise ValueError("constraint must not be empty.")

    solver = solver_context["solver"]
    assert solver is not None
    variables = _ensure_context_variables([constraint])
    eval_env = _build_z3_eval_env(variables)
    solver.add(eval(_compile_logic_expression(constraint), {"__builtins__": {}}, eval_env))
    constraints = solver_context["constraints"]
    assert isinstance(constraints, list)
    constraints.append(constraint)
    return {
        "message": "Constraint added",
        "constraint": constraint,
        "constraints": list(constraints),
    }


@mcp.tool
def z3_check_satisfiability() -> Dict[str, object]:
    """Check the persistent Z3 solver context and return its satisfiability status."""
    solver = solver_context["solver"]
    variables = solver_context["variables"]
    constraints = solver_context["constraints"]
    assert solver is not None
    assert isinstance(variables, dict)
    assert isinstance(constraints, list)
    return _check_solver_result(solver, variables, constraints)


@mcp.tool
def z3_reset_solver() -> Dict[str, str]:
    """Reset the persistent Z3 solver context."""
    reset_solver_context()
    return {"message": "Solver context reset successfully"}


@mcp.tool
def z3_solver_status() -> Dict[str, object]:
    """Return the current persistent solver status."""
    variables = solver_context["variables"]
    constraints = solver_context["constraints"]
    solver = solver_context["solver"]
    assert isinstance(variables, dict)
    assert isinstance(constraints, list)
    return {
        "status": "active" if solver is not None else "inactive",
        "constraints_count": len(constraints),
        "variables_count": len(variables),
        "constraints": list(constraints),
    }


@mcp.tool
def check_entailment_from_z3(premises: List[str], claim: str, vars: Dict[str, str] | None = None) -> Dict[str, object]:
    """Compatibility wrapper for claim-shaped entailment checks.

    Supported grammar: `==`, `!=`, `<=`, `>=`, `<`, `>`, `+`, `-`, `*`, `/`, `Implies`,
    and boolean combinators as lowercase infix `and`/`or`/`not` (preferred - e.g.
    `x == 1 or x == 2`) or the capitalized Z3 functions `And(...)`/`Or(...)`/`Not(...)`
    (NOT as infix - `a Or b` is invalid; call it `Or(a, b)`).
    """

    return _tool_fn(check_entailment)(premises, claim, vars)


@mcp.tool
def z3_prove_theorem(premises: List[str], conclusion: str, vars: Dict[str, str] | None = None) -> Dict[str, object]:
    """Compatibility wrapper around `check_entailment`.

    Supported grammar: `==`, `!=`, `<=`, `>=`, `<`, `>`, `+`, `-`, `*`, `/`, `Implies`,
    and boolean combinators as lowercase infix `and`/`or`/`not` (preferred - e.g.
    `x == 1 or x == 2`) or the capitalized Z3 functions `And(...)`/`Or(...)`/`Not(...)`
    (NOT as infix - `a Or b` is invalid; call it `Or(a, b)`).
    """

    result = _tool_fn(check_entailment)(premises, conclusion, vars)
    response = {
        "proved": result.get("status") == "entailed",
        "status": result.get("status"),
        "ok": result.get("ok", False),
        "reason": result.get("reason", ""),
        "suggestion": result.get("suggestion", ""),
    }
    if result.get("status") == "not_entailed":
        response["counterexample"] = result.get("value", {}).get("counterexample")
    return response


# Builtins safe enough to let a verification script enumerate finite domains
# (permutations of roles, loops building constraints) without opening any real
# capability (no file/network/import access).
_SCRIPT_SAFE_BUILTINS: Dict[str, object] = {
    name: getattr(builtins, name)
    for name in (
        "range", "len", "list", "dict", "tuple", "set", "frozenset", "enumerate",
        "zip", "map", "filter", "min", "max", "sum", "all", "any", "sorted",
        "reversed", "abs", "bool", "int", "float", "str", "print", "isinstance",
    )
}


@mcp.tool
def z3_run_script(statements: List[str]) -> Dict[str, object]:
    """Run a short Z3 Python script for models the single-expression tools can't
    express: quantifiers, custom sorts/functions, or a finite-domain check built with
    a loop (e.g. enumerate every permutation of roles and assert a property holds for
    each, instead of hand-verifying each case).

    The script is a list of Python statements (assignments, loops, `solver.add(...)`
    calls, ...), executed in order with the full Z3 Python API available by name
    (Solver, Bool, Int, Real, And, Or, Not, Implies, ForAll, Exists, DeclareSort,
    Function, Const, If, Distinct, ...) plus `itertools` and basic builtins
    (range/len/enumerate/zip/...) - no file, network, or import access. The script
    MUST define a variable named `solver` (a `Solver()` or `Optimize()` instance)
    holding the assembled model; its satisfiability and model are returned.

    To PROVE a claim: add the premises, then `solver.add(Not(claim))` - `unsat` means
    the premises entail the claim (no counterexample exists); `sat` gives a
    counterexample via the returned model.

    >>> z3_run_script([
    ...     "Object = DeclareSort('Object')",
    ...     "Human = Function('Human', Object, BoolSort())",
    ...     "Mortal = Function('Mortal', Object, BoolSort())",
    ...     "socrates = Const('socrates', Object)",
    ...     "x = Const('x', Object)",
    ...     "solver = Solver()",
    ...     "solver.add(ForAll([x], Implies(Human(x), Mortal(x))))",
    ...     "solver.add(Human(socrates))",
    ...     "solver.add(Not(Mortal(socrates)))",
    ... ])
    {'status': 'unsat', ...}   # premises + negated conclusion is unsat -> conclusion proved
    """
    if not statements:
        return _verdict(
            False,
            status="invalid_input",
            reason="statements must not be empty.",
            suggestion="Provide statements that build a model and assign it to `solver`.",
        )

    env = _build_z3_eval_env()
    env["itertools"] = itertools
    env["__builtins__"] = _SCRIPT_SAFE_BUILTINS
    try:
        # A single namespace for both globals and locals: if the script `def`s a
        # helper function, that function's __globals__ becomes this dict, so it can
        # still see Solver/If/And/... when called later in the script. Passing two
        # separate dicts (module-exec style) would leave such a function unable to
        # resolve any of these names at call time.
        exec("\n".join(statements), env)  # noqa: S102 - sandboxed: no file/network/import access
    except Exception as exc:
        return _verdict(
            False,
            status="script_failed",
            reason=f"Script raised: {exc}",
            suggestion="Check the statements for syntax errors, undefined names, or a missing `solver = Solver()`.",
        )

    solver = env.get("solver")
    if solver is None or not hasattr(solver, "check"):
        return _verdict(
            False,
            status="invalid_input",
            reason="The script did not define `solver` as a Solver()/Optimize() instance.",
            suggestion="Add `solver = Solver()` and build the model with `solver.add(...)`.",
        )

    try:
        result = solver.check()
    except Exception as exc:
        return _verdict(
            False,
            status="check_failed",
            reason=f"solver.check() raised: {exc}",
            suggestion="Simplify the model or check for malformed constraints.",
        )

    status = str(result)
    value: Dict[str, object] = {"status": status}
    if result == _z3.sat:
        model = solver.model()
        value["model"] = {
            str(decl.name()): _coerce_model_value(model[decl]) for decl in model.decls()
        }
    return _verdict(True, status=status, reason=f"Z3 returned {status}.", value=value)


@mcp.tool
def differentiate(
    expression: str,
    point: float,
    variable: str = "x",
    dx: float = 1e-6,
    order: int = 5,
    n: int = 1,
) -> float:
    """Evaluate the n-th derivative of the expression at the given point.

    Uses central differences for numerical differentiation. Examples:
    >>> differentiate('x**3 + 2*x', point=2.0)
    14.0
    >>> differentiate('np.sin(x)', point=0.0)
    1.0
    """
    if order % 2 == 0:
        raise ValueError("order must be an odd integer for numerical differentiation.")
    func = _build_callable(expression, variable)
    return float(numerical_derivative(func, point, dx=dx, n=n, order=order))


@mcp.tool
def test_root_stability(
    expression: str,
    bracket_start: float = -10.0,
    bracket_end: float = 10.0,
    variable: str = "x",
    relative_perturbation: float = 1e-3,
    tolerance: float = 1e-6,
) -> Dict[str, object]:
    """Assess how numerically stable a root of `expression` is.

    Solves the equation (reusing `solve_equation`), then re-solves with the bracket
    endpoints nudged by +/- `relative_perturbation` of the bracket width. A stable root
    barely moves; a large spread signals an ill-conditioned (flat or clustered) root.
    The slope `f'(root)` is reported because a near-zero slope at the crossing is the
    classic source of root ill-conditioning.

    >>> test_root_stability('x**2 - 9', 0.0, 5.0)['stable']
    True
    """
    baseline_result = _tool_fn(solve_equation)(expression, bracket_start, bracket_end, variable)
    if not baseline_result.get("ok"):
        return baseline_result
    baseline = float(baseline_result["value"])

    try:
        func = _build_callable(_normalize_equation_expression(expression), variable)
        slope = numerical_derivative(func, baseline)
    except Exception as exc:
        return _verdict(
            False,
            status="analysis_failed",
            reason=f"Unable to analyze root stability: {exc}",
            suggestion="Check the expression for unsupported functions or unit mismatches.",
        )

    width = max(abs(bracket_end - bracket_start), 1.0)
    shift = width * relative_perturbation
    roots: List[float] = [baseline]
    for ds, de in ((shift, 0.0), (-shift, 0.0), (0.0, shift), (0.0, -shift)):
        result = _tool_fn(solve_equation)(expression, bracket_start + ds, bracket_end + de, variable)
        if not result.get("ok"):
            continue
        roots.append(float(result["value"]))

    roots_arr = np.asarray(roots, dtype=float)
    spread = float(roots_arr.max() - roots_arr.min())
    return {
        "root": float(baseline),
        "slope_at_root": float(slope),
        "samples": [float(r) for r in roots],
        "std_dev": float(roots_arr.std()),
        "spread": spread,
        "stable": bool(spread <= tolerance and abs(slope) > tolerance),
    }


@mcp.tool
def test_derivative_stability(
    expression: str,
    point: float,
    variable: str = "x",
    n: int = 1,
    order: int = 5,
) -> Dict[str, object]:
    """Assess the numerical stability of `differentiate` by sweeping the step size `dx`.

    Central differences trade truncation error (large `dx`) against rounding noise
    (tiny `dx`). This sweeps `dx` across several magnitudes (reusing
    `numerical_derivative`) and reports the value plateau plus the `dx` whose neighbours
    agree most closely, which is the most trustworthy estimate.

    >>> round(test_derivative_stability('np.sin(x)', 0.0)['value'], 6)
    1.0
    """
    func = _build_callable(expression, variable)
    step_sizes = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    values_by_dx: Dict[str, float] = {}
    for dx in step_sizes:
        try:
            values_by_dx[f"{dx:.0e}"] = float(
                numerical_derivative(func, point, dx=dx, n=n, order=order)
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            continue

    if not values_by_dx:
        return _verdict(
            False,
            status="analysis_failed",
            reason="Could not evaluate the derivative at any step size.",
            suggestion="Check the expression for unsupported functions or simplify it.",
        )

    keys = list(values_by_dx)
    vals = [values_by_dx[k] for k in keys]
    # Pick the dx whose value is closest to its neighbours (the stable plateau).
    best_idx, best_diff = 0, float("inf")
    for i in range(len(vals)):
        neighbours = [vals[j] for j in (i - 1, i + 1) if 0 <= j < len(vals)]
        diff = sum(abs(vals[i] - nb) for nb in neighbours) / len(neighbours)
        if diff < best_diff:
            best_idx, best_diff = i, diff

    return {
        "value": float(vals[best_idx]),
        "best_dx": float(step_sizes[best_idx]),
        "values_by_dx": values_by_dx,
        "max_variation": float(max(vals) - min(vals)),
        "stable": bool(best_diff <= 1e-4 * (abs(vals[best_idx]) + 1.0)),
    }


@mcp.tool
def integrate_function(
    expression: str,
    lower: float,
    upper: float,
    variable: str = "x",
) -> Dict[str, float]:
    """Compute the definite integral of the expression between lower and upper bounds.

    Uses `scipy.integrate.quad`. Examples:
    >>> integrate_function('np.exp(-x**2)', lower=0.0, upper=1.0)
    {'value': 0.7468241328124271, 'abserr': 8.291413475940725e-15}
    >>> integrate_function('np.sin(x)', lower=0.0, upper=np.pi)
    {'value': 2.0, 'abserr': 2.220446049250313e-14}
    """
    func = _build_callable(expression, variable)
    value, error = integrate.quad(func, lower, upper)
    return {"value": float(value), "abserr": float(error)}


@mcp.tool
def distribution_pdf(
    distribution: str,
    x: float,
    shape_args: Optional[List[float]] = None,
    loc: float = 0.0,
    scale: float = 1.0,
    extra_params: Optional[Dict[str, float]] = None,
) -> float:
    """Evaluate the PDF/PMF of a SciPy distribution at x.

    >>> distribution_pdf('norm', x=0.0)
    0.3989422804014327
    >>> distribution_pdf('binom', x=3, shape_args=[10, 0.5])
    0.1171875
    """
    dist = _get_distribution(distribution)
    args, kwargs = _distribution_args_kwargs(dist, shape_args, loc, scale, extra_params)
    return _pmf_or_pdf(dist, x, args, kwargs)


@mcp.tool
def distribution_cdf(
    distribution: str,
    x: float,
    shape_args: Optional[List[float]] = None,
    loc: float = 0.0,
    scale: float = 1.0,
    lower_tail: bool = True,
    extra_params: Optional[Dict[str, float]] = None,
) -> float:
    """Evaluate the CDF (or survival function) of a SciPy distribution at x.

    >>> distribution_cdf('norm', x=1.96)
    0.9750021048517795
    """
    dist = _get_distribution(distribution)
    args, kwargs = _distribution_args_kwargs(dist, shape_args, loc, scale, extra_params)
    if lower_tail:
        return float(dist.cdf(x, *args, **kwargs))
    return float(dist.sf(x, *args, **kwargs))


@mcp.tool
def distribution_quantile(
    distribution: str,
    probability: float,
    shape_args: Optional[List[float]] = None,
    loc: float = 0.0,
    scale: float = 1.0,
    lower_tail: bool = True,
    extra_params: Optional[Dict[str, float]] = None,
) -> float:
    """Return the quantile corresponding to the given cumulative probability.

    >>> distribution_quantile('norm', probability=0.975)
    1.959963984540054
    """
    dist = _get_distribution(distribution)
    args, kwargs = _distribution_args_kwargs(dist, shape_args, loc, scale, extra_params)
    if lower_tail:
        return float(dist.ppf(probability, *args, **kwargs))
    return float(dist.isf(probability, *args, **kwargs))


@mcp.tool
def distribution_probability_between(
    distribution: str,
    lower: float,
    upper: float,
    shape_args: Optional[List[float]] = None,
    loc: float = 0.0,
    scale: float = 1.0,
    extra_params: Optional[Dict[str, float]] = None,
) -> float:
    """Compute P(lower <= X <= upper) for the specified distribution.

    Continuous distributions use CDF differences; discrete ones include the lower PMF.
    >>> distribution_probability_between('norm', lower=-1.0, upper=1.0)
    0.6826894921370859
    """
    if lower > upper:
        raise ValueError("lower must be less than or equal to upper.")
    dist = _get_distribution(distribution)
    args, kwargs = _distribution_args_kwargs(dist, shape_args, loc, scale, extra_params)
    cdf_upper = dist.cdf(upper, *args, **kwargs)
    cdf_lower = dist.cdf(lower, *args, **kwargs)
    probability = cdf_upper - cdf_lower
    if hasattr(dist, "pmf"):
        probability += dist.pmf(lower, *args, **kwargs)
    return float(probability)


@mcp.tool
def usage_examples() -> Dict[str, List[Dict[str, str]]]:
    """Return sample tool invocations with expected numeric results."""
    # Provide serializable copies of the example data for easy display.
    return {
        category: [example.copy() for example in entries]
        for category, entries in USAGE_EXAMPLES.items()
    }


def run_server(transport: str | None = None) -> None:
    """Start the FastMCP server using the selected transport."""
    selected_transport = transport or os.getenv("MATH_MCP_TRANSPORT", "streamable-http")
    if selected_transport in {"streamable-http", "http"}:
        mcp.run(transport="http", path=SERVER_PATH, host=SERVER_HOST, port=SERVER_PORT)
    else:
        mcp.run(transport=selected_transport)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Math MCP server")
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
