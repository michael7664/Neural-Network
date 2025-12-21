from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import sympy as sp
from sympy.solvers.ode import classify_ode
from sympy.solvers.pde import classify_pde

from .parse_and_classify import parse_equation, Problem
from .symbolic_solve import symbolic_solve
from .pinn_solve import build_residual_from_sympy, train_pinn


@dataclass
class SolveRequest:
    equation_str: str
    func_name: str
    var_names: List[str]
    prefer: str = "auto"   # "auto" | "symbolic" | "pinn"


def problem_from_eq(eq: sp.Eq, func_name: str, var_names: List[str]) -> Problem:
    vars_ = tuple(sp.Symbol(v) for v in var_names)
    u = sp.Function(func_name)
    u_applied = u(*vars_)

    if len(vars_) == 1:
        hints = list(classify_ode(eq, u_applied))
        kind = "ode" if hints else "unknown"
    else:
        hints = list(classify_pde(eq, u_applied))
        kind = "pde" if hints else "unknown"

    return Problem(eq=eq, func=u, indep_vars=vars_, kind=kind, hints=hints)


def solve_eq(eq: sp.Eq,
             func_name: str,
             var_names: List[str],
             prefer: str = "auto",
             pinn_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prob = problem_from_eq(eq, func_name, var_names)

    # 1) Symbolic path (fast, exact when it works)
    if prefer in ("auto", "symbolic") and prob.kind in ("ode", "pde"):
        sym = symbolic_solve(prob, hint="default")
        if sym.ok:
            return {
                "method": "symbolic",
                "kind": prob.kind,
                "hints": prob.hints,
                "solution": sym.solution,
            }

    # 2) PINN path (returns a trained model, not a closed-form expression)
    if prefer in ("auto", "pinn"):
        if pinn_payload is None:
            return {
                "method": "none",
                "kind": prob.kind,
                "hints": prob.hints,
                "error": "PINN requires pinn_payload with domain + conditions (+ optional const_subs).",
            }

        # residual R = lhs - rhs == 0
        residual_expr = sp.simplify(prob.eq.lhs - prob.eq.rhs)

        # substitute constants (e.g., nu, alpha) if provided
        const_subs = pinn_payload.get("const_subs", {})
        if const_subs:
            residual_expr = residual_expr.subs(const_subs)

        residual_fn = build_residual_from_sympy(residual_expr, prob.indep_vars, prob.func)

        model = train_pinn(
        residual_fn,
        domain=pinn_payload["domain"],
        conditions=pinn_payload["conditions"],
        var_names=var_names,
        steps=pinn_payload.get("steps", 20000),
        lr=pinn_payload.get("lr", 1e-3),
        n_f=pinn_payload.get("n_f", 20000),
        device=pinn_payload.get("device", "cpu"),
        cond_batch=pinn_payload.get("cond_batch", 256),
        print_every=pinn_payload.get("print_every", 500),

        # new: loss weighting + LBFGS polish
        w_f=pinn_payload.get("w_f", 1.0),
        w_c=pinn_payload.get("w_c", 1.0),
        lbfgs_steps=pinn_payload.get("lbfgs_steps", 0),
        lbfgs_lr=pinn_payload.get("lbfgs_lr", 1.0),
        )


        return {
            "method": "pinn",
            "kind": prob.kind,
            "hints": prob.hints,
            "model": model,
            "residual_fn": residual_fn,         # for generic residual validation
            "residual_expr": residual_expr,     # optional: nice for debugging/logging
        }

    return {"method": "none", "kind": prob.kind, "hints": prob.hints, "error": "No method selected/succeeded."}


def solve(req: SolveRequest, pinn_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # legacy: parse from a SymPy-style string
    prob = parse_equation(req.equation_str, req.func_name, req.var_names)
    return solve_eq(prob.eq, req.func_name, req.var_names, prefer=req.prefer, pinn_payload=pinn_payload)
