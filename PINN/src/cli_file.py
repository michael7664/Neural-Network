import argparse
import sympy as sp
import torch
import numpy as np
import matplotlib.pyplot as plt

from .input_equation import read_equation_from_txt, read_equation_from_image
from .router import solve_eq
from .pinn_solve import Domain, Conditions
from .config_loader import load_config


def sample_uniform(domain_bounds: dict, var_names: list[str], n: int) -> np.ndarray:
    cols = []
    for v in var_names:
        lo, hi = domain_bounds[v]
        cols.append(lo + (hi - lo) * np.random.rand(n, 1))
    return np.hstack(cols).astype(np.float32)


def residual_stats(residual_fn, model, domain_bounds: dict, var_names: list[str], n=20000):
    device = next(model.parameters()).device
    xt = sample_uniform(domain_bounds, var_names, n)
    X = torch.tensor(xt, device=device)
    r = residual_fn(model, X)  # autodiff happens inside residual_fn
    r_np = r.detach().cpu().numpy().ravel()
    return {
        "residual_mean": float(np.mean(r_np)),
        "residual_mae": float(np.mean(np.abs(r_np))),
        "residual_rmse": float(np.sqrt(np.mean(r_np ** 2))),
        "residual_maxabs": float(np.max(np.abs(r_np))),
    }


def save_solution_grid(out, domain_bounds: dict, var_names: list[str], grid: dict):
    """
    Saves a grid evaluation to .npy for any dimension.
    If vars are exactly x,t also saves a heatmap PNG.
    """
    model = out["model"]
    model.eval()

    axes = []
    for v in var_names:
        lo, hi = domain_bounds[v]
        n = int(grid.get(v, 101))
        axes.append(np.linspace(lo, hi, n, dtype=np.float64))

    meshes = np.meshgrid(*axes, indexing="xy")
    flat = np.stack([m.reshape(-1) for m in meshes], axis=1).astype(np.float32)

    device = next(model.parameters()).device
    with torch.no_grad():  # good practice for inference [web:505]
        u = model(torch.tensor(flat, device=device)).cpu().numpy().reshape(*[len(a) for a in axes])

    npy_path = grid.get("npy", "u_pinn.npy")
    np.save(npy_path, u)
    print(f"Saved {npy_path}")

    # Optional plot for x,t
    if var_names == ["x", "t"] and grid.get("png", None):
        x = axes[0]
        t = axes[1]
        U = u.T  # meshgrid indexing="xy"
        plt.figure(figsize=(8, 3))
        plt.imshow(U, aspect="auto", origin="lower", extent=[x.min(), x.max(), t.min(), t.max()])
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        plt.title("PINN solution u(x,t)")
        plt.tight_layout()
        plt.savefig(grid["png"], dpi=150)
        plt.close()
        print(f"Saved {grid['png']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .txt/.tex or image (.png/.jpg)")
    ap.add_argument("--input-type", required=True, choices=["txt-sympy", "txt-latex", "image"])
    ap.add_argument("--func", default="u")
    ap.add_argument("--vars", default=None, help="Comma-separated vars (optional if --config is used)")
    ap.add_argument("--prefer", default="auto", choices=["auto", "symbolic", "pinn"])
    ap.add_argument("--config", default=None, help="YAML config path (domain, IC/BC, constants, outputs)")
    args = ap.parse_args()

    cfg = None
    if args.config:
        cfg = load_config(args.config)
        var_names = cfg.var_names
    else:
        if not args.vars:
            raise SystemExit("Either --vars or --config must be provided.")
        var_names = [v.strip() for v in args.vars.split(",") if v.strip()]

    if args.input_type == "txt-sympy":
        parsed = read_equation_from_txt(args.input, mode="sympy", func_name=args.func, var_names=var_names)
    elif args.input_type == "txt-latex":
        parsed = read_equation_from_txt(args.input, mode="latex", func_name=args.func, var_names=var_names)
    else:
        parsed = read_equation_from_image(args.input, func_name=args.func, var_names=var_names)

    print("Parsed equation:", parsed.eq)
    if parsed.latex:
        print("LaTeX:", parsed.latex)
    print("Normalized equation:", parsed.eq)

    pinn_payload = None
    domain_bounds = None

    if args.prefer == "pinn" or (args.prefer == "auto" and len(var_names) >= 2):
        if cfg is None:
            raise SystemExit("PINN requires --config to define domain + IC/BC + constants.")
        domain_bounds = cfg.bounds

        domain = Domain(bounds=domain_bounds)
        conditions = Conditions(ic=cfg.ic, bc=cfg.bc)

        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

        pinn_payload = {
            "domain": domain,
            "conditions": conditions,

            # core training params
            "steps": int(cfg.pinn.get("steps", 5000)),
            "lr": float(cfg.pinn.get("lr", 1e-3)),
            "n_f": int(cfg.pinn.get("n_f", 5000)),
            "device": cfg.pinn.get("device", dev),
            "cond_batch": int(cfg.pinn.get("cond_batch", 256)),
            "print_every": int(cfg.pinn.get("print_every", 100)),

            # constants
            "const_subs": cfg.consts,

            # new: weights + optional LBFGS polish
            "w_f": float(cfg.pinn.get("w_f", 1.0)),
            "w_c": float(cfg.pinn.get("w_c", 1.0)),
            "lbfgs_steps": int(cfg.pinn.get("lbfgs_steps", 0)),
            "lbfgs_lr": float(cfg.pinn.get("lbfgs_lr", 1.0)),
        }

    out = solve_eq(parsed.eq, args.func, var_names, prefer=args.prefer, pinn_payload=pinn_payload)

    print("Method:", out["method"])
    print("Kind:", out.get("kind"))
    print("Hints:", out.get("hints", [])[:5])

    if out["method"] == "symbolic":
        print("Solution:", out["solution"])
        return

    if out["method"] == "pinn":
        grid_cfg = (cfg.output.get("grid", {}) if cfg else {})
        save_solution_grid(out, domain_bounds, var_names, grid_cfg)

        n_res = int((cfg.output.get("residual_points", 20000) if cfg else 20000))
        stats = residual_stats(out["residual_fn"], out["model"], domain_bounds, var_names, n=n_res)
        print("Residual stats:", stats)
        return

    print("No solution produced:", out.get("error"))


if __name__ == "__main__":
    main()
