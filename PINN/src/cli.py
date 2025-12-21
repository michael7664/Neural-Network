import numpy as np
import sympy as sp
import torch

from .router import SolveRequest, solve
from .pinn_solve import Domain, Conditions


def main():
    # Example 1: ODE symbolic
    req1 = SolveRequest(
        equation_str="Eq(diff(u(x), x) + u(x), 0)",
        func_name="u",
        var_names=["x"],
        prefer="auto",
    )
    out1 = solve(req1)
    print(out1["method"], out1.get("solution"))

    # Example 2: PDE Burgers PINN
    nu = 0.01
    req2 = SolveRequest(
        equation_str="Eq(diff(u(x,t), t) + u(x,t)*diff(u(x,t), x) - nu*diff(u(x,t), x, 2), 0)",
        func_name="u",
        var_names=["x", "t"],
        prefer="pinn",
    )

    domain = Domain(bounds={"x": (-1.0, 1.0), "t": (0.0, 1.0)})

    def ic_sampler(n):
        x = -1 + 2*np.random.rand(n, 1)
        t = np.zeros((n, 1))
        return np.hstack([x, t])

    def ic_target(X):
        x = X[:, 0:1]
        return -np.sin(np.pi * x)

    def bc_sampler_left(n):
        x = -np.ones((n, 1))
        t = np.random.rand(n, 1)
        return np.hstack([x, t])

    def bc_sampler_right(n):
        x = np.ones((n, 1))
        t = np.random.rand(n, 1)
        return np.hstack([x, t])

    def bc_target(X):
        return np.zeros((X.shape[0], 1))

    conditions = Conditions(
        ic=[(ic_sampler, ic_target)],
        bc=[(bc_sampler_left, bc_target), (bc_sampler_right, bc_target)],
    )

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    pinn_payload = {
        "domain": domain,
        "conditions": conditions,
        "steps": 20000,
        "lr": 1e-3,
        "n_f": 20000,
        "device": dev,
        "const_subs": {sp.Symbol("nu"): nu},
        "cond_batch": 256,
        "print_every": 500,
    }

    out2 = solve(req2, pinn_payload=pinn_payload)
    print(out2["method"], "hints:", out2.get("hints", [])[:3])


if __name__ == "__main__":
    main()
