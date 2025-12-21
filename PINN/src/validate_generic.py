import numpy as np
import torch


def sample_uniform(domain_bounds: dict, var_names: list[str], n: int) -> np.ndarray:
    cols = []
    for v in var_names:
        lo, hi = domain_bounds[v]
        cols.append(lo + (hi - lo) * np.random.rand(n, 1))
    return np.hstack(cols).astype(np.float32)


def residual_stats(residual_fn, model, domain_bounds: dict, var_names: list[str],
                   n=20000, device="cpu"):
    """
    residual_fn: callable(model, X)->torch tensor [N,1]
    """
    xt = sample_uniform(domain_bounds, var_names, n)
    X = torch.tensor(xt, device=device)

    r = residual_fn(model, X)  # needs grads inside residual_fn
    r_np = r.detach().cpu().numpy().ravel()

    return {
        "residual_mean": float(np.mean(r_np)),
        "residual_mae": float(np.mean(np.abs(r_np))),
        "residual_rmse": float(np.sqrt(np.mean(r_np**2))),
        "residual_maxabs": float(np.max(np.abs(r_np))),
    }
