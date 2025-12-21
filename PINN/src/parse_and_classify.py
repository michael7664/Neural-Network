from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from sympy.solvers.ode import classify_ode
from sympy.solvers.pde import classify_pde

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


@dataclass
class Problem:
    eq: sp.Eq
    func: sp.Function
    indep_vars: Tuple[sp.Symbol, ...]   # (x,) or (x,t)
    kind: str                           # "ode" | "pde" | "unknown"
    hints: List[str]


def parse_equation(equation_str: str, func_name: str, var_names: List[str]) -> Problem:
    """
    Parse a SymPy-style equation string, e.g.:
      "Eq(diff(u(x), x) + u(x), 0)"
      "Eq(diff(u(x,t), t) - diff(u(x,t), x, 2), 0)"
    """
    vars_ = tuple(sp.Symbol(v) for v in var_names)
    u = sp.Function(func_name)

    local = {v.name: v for v in vars_}
    local[func_name] = u
    local["Eq"] = sp.Eq
    local["diff"] = sp.diff
    local["Derivative"] = sp.Derivative

    eq = parse_expr(equation_str, local_dict=local, transformations=TRANSFORMS)
    if not isinstance(eq, sp.Equality):
        raise ValueError("Equation must be an Eq(lhs, rhs).")

    u_applied = u(*vars_)

    if len(vars_) == 1:
        hints = list(classify_ode(eq, u_applied))
        kind = "ode" if hints else "unknown"
    else:
        hints = list(classify_pde(eq, u_applied))
        kind = "pde" if hints else "unknown"

    return Problem(eq=eq, func=u, indep_vars=vars_, kind=kind, hints=hints)
