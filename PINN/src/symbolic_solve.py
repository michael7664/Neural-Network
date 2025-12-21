from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

import sympy as sp
from sympy.solvers.ode import dsolve
from sympy.solvers.pde import pdsolve

from .parse_and_classify import Problem


@dataclass
class SymbolicResult:
    ok: bool
    solution: Optional[Any]
    error: Optional[str]


def symbolic_solve(problem: Problem, hint: str = "default") -> SymbolicResult:
    try:
        u_applied = problem.func(*problem.indep_vars)

        if problem.kind == "ode":
            sol = dsolve(problem.eq, u_applied, hint=hint)
            return SymbolicResult(ok=True, solution=sol, error=None)

        if problem.kind == "pde":
            sol = pdsolve(problem.eq, u_applied, hint=hint)
            return SymbolicResult(ok=True, solution=sol, error=None)

        return SymbolicResult(ok=False, solution=None, error="Unknown equation type.")
    except Exception as e:
        return SymbolicResult(ok=False, solution=None, error=str(e))
