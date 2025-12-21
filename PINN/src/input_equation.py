from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, List
import re

import sympy as sp
from sympy.parsing.latex import parse_latex  # requires ANTLR runtime 4.11 [web:155]


@dataclass
class ParsedEquation:
    eq: sp.Eq
    latex: Optional[str] = None


def _latex_dfrac(order: int, func_name: str, var: str) -> str:
    if order <= 0:
        return func_name
    if order == 1:
        return rf"\frac{{d {func_name}}}{{d {var}}}"
    return rf"\frac{{d^{order} {func_name}}}{{d {var}^{order}}}"


def latex_expand_primes(latex: str, func_name: str, var: str) -> str:
    """
    Convert prime notation to explicit derivative notation before parse_latex.

    Matches:
      u'   u ''   u'''  (with optional spaces)
    without relying on \\b word boundaries.
    """
    # Match: func_name + optional spaces + one-or-more apostrophes
    pattern = rf"{re.escape(func_name)}\s*('+)"
    def repl(m):
        order = len(m.group(1))
        return _latex_dfrac(order, func_name, var)
    return re.sub(pattern, repl, latex)




def latex_expand_dots(latex: str, func_name: str, var: str) -> str:
    """
    Convert dot notation (typically time derivatives) to explicit derivatives.

    Handles:
      \\dot{u}, \\ddot{u}, \\dddot{u}, \\ddddot{u}
    and also versions without braces: \\dot u, \\ddot u, ...

    By default maps dots to derivatives w.r.t. `var` (e.g. t).
    """
    # Order matters: replace higher dots first
    dot_cmds = [
        (r"\\ddddot", 4),
        (r"\\dddot", 3),
        (r"\\ddot", 2),
        (r"\\dot", 1),
    ]

    out = latex
    for cmd, order in dot_cmds:
        # With braces: \dot{u}
        out = re.sub(
            rf"{cmd}\s*\{{\s*{re.escape(func_name)}\s*\}}",
            _latex_dfrac(order, func_name, var),
            out,
        )
        # Without braces: \dot u
        out = re.sub(
            rf"{cmd}\s+{re.escape(func_name)}\b",
            _latex_dfrac(order, func_name, var),
            out,
        )
    return out


def _to_eq(expr) -> sp.Eq:
    if isinstance(expr, sp.Equality):
        return expr
    return sp.Eq(expr, 0)


def normalize_depvar(eq: sp.Eq, func_name: str, var_names: List[str]) -> sp.Eq:
    """
    Robust normalization for LaTeX-parsed equations where the dependent variable
    may appear as a plain Symbol (e.g. u) instead of Function(u)(x).

    Converts:
      u                  -> u(x) (or u(x,t))
      Derivative(u, x)   -> Derivative(u(x), x)

    This is needed because SymPy's ODE/PDE classifiers/solvers expect u(x), not bare u.
    """
    vars_ = tuple(sp.Symbol(v) for v in var_names)
    u_applied = sp.Function(func_name)(*vars_)
    u_sym = sp.Symbol(func_name)

    # Replace bare symbol u -> u(x,...) everywhere
    eq2 = eq.subs(u_sym, u_applied)

    # Fix derivatives written as Derivative(u, x) where u is still a Symbol
    rep_d = {}
    for node in sp.preorder_traversal(eq2):
        if isinstance(node, sp.Derivative) and node.expr == u_sym:
            rep_d[node] = sp.Derivative(u_applied, *node.variables)
    if rep_d:
        eq2 = eq2.xreplace(rep_d)

    return eq2

def normalize_latex_derivative_symbols(eq: sp.Eq, func_name: str, var_names: List[str]) -> sp.Eq:
    """
    Some LaTeX parses produce symbols like u_{t}, u_{x}, u_{x*x} instead of Derivative(u(x,t), t).
    Convert those symbols into true SymPy Derivative objects so the PINN residual builder can use them.
    """
    vars_ = tuple(sp.Symbol(v) for v in var_names)
    u_applied = sp.Function(func_name)(*vars_)

    # Map var name -> Symbol instance
    vmap = {v.name: v for v in vars_}

    repl = {}
    for s in eq.atoms(sp.Symbol):
        name = s.name

        # Match u_{...} where ... is like t or x or x*x or x*t etc.
        m = re.fullmatch(rf"{re.escape(func_name)}_\{{(.+)\}}", name)
        if not m:
            continue

        inside = m.group(1).strip()

        # tokens separated by '*' (SymPy prints x*x for second derivative)
        parts = [p.strip() for p in inside.split("*") if p.strip()]
        if not parts:
            continue

        # Only convert if all parts are known independent vars
        if any(p not in vmap for p in parts):
            continue

        deriv_vars = [vmap[p] for p in parts]
        repl[s] = sp.Derivative(u_applied, *deriv_vars)

    if repl:
        eq = eq.xreplace(repl)

    return eq



def preprocess_latex(latex: str, func_name: str, var_names: List[str]) -> str:
    """
    Apply prime/dot expansions before parse_latex.

    Practical heuristic:
    - If 1 variable: both primes and dots map to that variable (usually x or t).
    - If >=2 variables: dots map to 't' if present, else the last variable.
      (primes are ambiguous in multivariable PDEs; by default we do NOT expand primes
       when there are multiple variables.)
    """
    latex2 = latex.strip()

    if len(var_names) == 1:
        v = var_names[0]
        latex2 = latex_expand_dots(latex2, func_name, v)
        latex2 = latex_expand_primes(latex2, func_name, v)
        return latex2

    # multi-variable: interpret dots as time derivatives if possible
    v = "t" if "t" in var_names else var_names[-1]
    latex2 = latex_expand_dots(latex2, func_name, v)
    return latex2


def read_equation_from_txt(
    path: str,
    mode: Literal["sympy", "latex"] = "latex",
    func_name: str = "u",
    var_names: Optional[List[str]] = None,
) -> ParsedEquation:
    text = open(path, "r", encoding="utf-8").read().strip()

    if mode == "sympy":
        local = {"Eq": sp.Eq, "diff": sp.diff, "Derivative": sp.Derivative}
        eq = sp.sympify(text, locals=local)
        if not isinstance(eq, sp.Equality):
            raise ValueError("sympy mode requires Eq(lhs, rhs).")
        return ParsedEquation(eq=eq, latex=None)

    if var_names is None:
        raise ValueError("latex mode requires var_names, e.g. ['x'] or ['x','t'].")

    text2 = preprocess_latex(text, func_name, var_names)

    expr = parse_latex(text2)
    eq = _to_eq(expr)
    eq = normalize_depvar(eq, func_name, var_names)
    eq = normalize_latex_derivative_symbols(eq, func_name, var_names)
    return ParsedEquation(eq=eq, latex=text2)


def read_equation_from_image(
    path: str,
    func_name: str = "u",
    var_names: Optional[List[str]] = None,
) -> ParsedEquation:
    """
    OCR image -> LaTeX -> SymPy. OCR dependencies are imported only here.
    """
    if var_names is None:
        raise ValueError("image mode requires var_names, e.g. ['x'] or ['x','t'].")

    from PIL import Image
    from pix2tex.cli import LatexOCR  # OCR: image -> LaTeX

    model = LatexOCR()
    latex = model(Image.open(path)).strip()

    latex2 = preprocess_latex(latex, func_name, var_names)

    expr = parse_latex(latex2)
    eq = _to_eq(expr)
    eq = normalize_depvar(eq, func_name, var_names)
    eq = normalize_latex_derivative_symbols(eq, func_name, var_names)
    return ParsedEquation(eq=eq, latex=latex2)
