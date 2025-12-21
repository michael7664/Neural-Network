from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable, Optional

import numpy as np
import sympy as sp
import torch
import torch.nn as nn


@dataclass
class Domain:
    bounds: Dict[str, Tuple[float, float]]  # {"x":(-1,1), "t":(0,1)}


@dataclass
class Conditions:
    # list of (sampler_fn, target_fn)
    ic: List[Tuple[Callable[[int], np.ndarray], Callable[[np.ndarray], np.ndarray]]]
    bc: List[Tuple[Callable[[int], np.ndarray], Callable[[np.ndarray], np.ndarray]]]


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=128, depth=6):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


_TORCH_FUNCS = {
    sp.sin: torch.sin,
    sp.cos: torch.cos,
    sp.exp: torch.exp,
    sp.tanh: torch.tanh,
}


def build_residual_from_sympy(residual_expr: sp.Expr,
                             vars_: tuple[sp.Symbol, ...],
                             u_func: sp.Function):
    """
    Torch-safe evaluator for SymPy residual expressions.
    Supports:
      - numbers, symbols (x,t,...)
      - u(vars)
      - Derivative(u(vars), ...)
      - +, *, Pow
      - basic funcs in _TORCH_FUNCS
    """
    u_applied = u_func(*vars_)

    derivs = set()
    for node in sp.preorder_traversal(residual_expr):
        if isinstance(node, sp.Derivative) and node.expr == u_applied:
            derivs.add(node)
    deriv_list = sorted(list(derivs), key=str)

    var_index = {v.name: i for i, v in enumerate(vars_)}

    def _grad(u, X, idx):
        g = torch.autograd.grad(
            outputs=u,
            inputs=X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,  # multiple derivatives from same forward pass
        )[0]
        return g[:, idx:idx + 1]

    def _eval_sympy(expr, env):
        if expr.is_Number:
            ref = next(iter(env.values()))
            return torch.zeros_like(ref) + float(expr)

        if isinstance(expr, sp.Symbol):
            if expr not in env:
                raise KeyError(f"Unknown symbol in residual: {expr}")
            return env[expr]

        if expr == u_applied:
            return env[u_applied]

        if isinstance(expr, sp.Derivative) and expr.expr == u_applied:
            return env[expr]

        if isinstance(expr, sp.Add):
            out = _eval_sympy(expr.args[0], env)
            for a in expr.args[1:]:
                out = out + _eval_sympy(a, env)
            return out

        if isinstance(expr, sp.Mul):
            out = _eval_sympy(expr.args[0], env)
            for a in expr.args[1:]:
                out = out * _eval_sympy(a, env)
            return out

        if isinstance(expr, sp.Pow):
            base = _eval_sympy(expr.args[0], env)
            exp = expr.args[1]
            if exp.is_Number:
                return base ** float(exp)
            return base ** _eval_sympy(exp, env)

        if isinstance(expr, sp.Function):
            f = type(expr)
            if f in _TORCH_FUNCS:
                arg0 = _eval_sympy(expr.args[0], env)
                return _TORCH_FUNCS[f](arg0)

        raise TypeError(f"Unsupported SymPy node: {expr} (type {type(expr)})")

    def residual(model, X: torch.Tensor):
        X = X.requires_grad_(True)
        u = model(X)  # [N,1]

        deriv_vals = {}
        for d in deriv_list:
            val = u
            for v in d.variables:
                val = _grad(val, X, var_index[v.name])
            deriv_vals[d] = val

        env = {vars_[i]: X[:, i:i + 1] for i in range(len(vars_))}
        env[u_applied] = u
        env.update(deriv_vals)

        return _eval_sympy(residual_expr, env)

    return residual


def sample_interior(domain: Domain, n: int, var_names: List[str]) -> np.ndarray:
    pts = []
    for v in var_names:
        lo, hi = domain.bounds[v]
        pts.append(lo + (hi - lo) * np.random.rand(n, 1))
    return np.hstack(pts).astype(np.float32)


def train_pinn(residual_fn,
               domain: Domain,
               conditions: Conditions,
               var_names: List[str],
               steps=20000,
               lr=1e-3,
               n_f=20000,
               device="cpu",
               cond_batch=256,
               print_every=500,
               w_f: float = 1.0,
               w_c: float = 1.0,
               lbfgs_steps: int = 0,
               lbfgs_lr: float = 1.0):
    model = MLP(in_dim=len(var_names)).to(device)

    def loss_terms():
        # physics points
        Xf = torch.tensor(sample_interior(domain, n_f, var_names), device=device)
        rf = residual_fn(model, Xf)
        loss_f = torch.mean(rf ** 2)

        # condition points
        loss_c = torch.tensor(0.0, device=device)
        for sampler, target in conditions.ic + conditions.bc:
            Xc_np = sampler(cond_batch).astype(np.float32)
            yc_np = target(Xc_np).astype(np.float32)
            Xc = torch.tensor(Xc_np, device=device)
            yc = torch.tensor(yc_np, device=device)
            uc = model(Xc)
            loss_c = loss_c + torch.mean((uc - yc) ** 2)

        loss = w_f * loss_f + w_c * loss_c
        return loss, loss_f, loss_c

    # Stage 1: Adam
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        loss, loss_f, _loss_c = loss_terms()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % print_every == 0 or step == 1:
            print(f"step={step} loss={float(loss.detach().cpu()):.3e} physics={float(loss_f.detach().cpu()):.3e}")

    # Stage 2: LBFGS polish (optional)
    if lbfgs_steps and lbfgs_steps > 0:
        opt2 = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_lr,
            max_iter=lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            loss, _loss_f, _loss_c = loss_terms()
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            return loss

        final_loss = opt2.step(closure)
        print(f"LBFGS done. final_loss={float(final_loss.detach().cpu()):.3e}")

    return model
