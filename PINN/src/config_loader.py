from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import sympy as sp
import yaml


@dataclass
class LoadedConfig:
    var_names: List[str]
    bounds: Dict[str, Tuple[float, float]]
    consts: Dict[sp.Symbol, float]
    ic: List[Tuple[Callable[[int], np.ndarray], Callable[[np.ndarray], np.ndarray]]]
    bc: List[Tuple[Callable[[int], np.ndarray], Callable[[np.ndarray], np.ndarray]]]
    output: Dict[str, Any]
    pinn: Dict[str, Any]


def _sampler_from_spec(spec: Dict[str, Any], bounds: Dict[str, Tuple[float, float]], var_names: List[str]):
    """
    Supported samplers:
      - {"type":"initial", "var":"t", "value":0.0}  -> t fixed, others uniform
      - {"type":"boundary", "var":"x", "value":-1.0} -> x fixed, others uniform
      - {"type":"uniform"} -> all vars uniform in bounds
    """
    stype = spec["type"]

    if stype == "uniform":
        def sampler(n: int):
            cols = []
            for v in var_names:
                lo, hi = bounds[v]
                cols.append(lo + (hi - lo) * np.random.rand(n, 1))
            return np.hstack(cols)
        return sampler

    if stype in ("initial", "boundary"):
        fixed_var = spec["var"]
        fixed_value = float(spec["value"])
        if fixed_var not in var_names:
            raise ValueError(f"Sampler fixed var '{fixed_var}' not in var_names={var_names}")

        def sampler(n: int):
            cols = []
            for v in var_names:
                if v == fixed_var:
                    cols.append(np.full((n, 1), fixed_value, dtype=np.float64))
                else:
                    lo, hi = bounds[v]
                    cols.append(lo + (hi - lo) * np.random.rand(n, 1))
            return np.hstack(cols)
        return sampler

    raise ValueError(f"Unknown sampler type: {stype}")


def _target_from_spec(spec: Dict[str, Any], var_names: List[str]):
    """
    Supported targets:
      - {"type":"zeros"}
      - {"type":"constant", "value": 0.5}
      - {"type":"expr", "expr":"-sin(pi*x)"}  # sympy expression in vars
    """
    ttype = spec["type"]

    if ttype == "zeros":
        def target(X: np.ndarray):
            return np.zeros((X.shape[0], 1), dtype=np.float64)
        return target

    if ttype == "constant":
        c = float(spec["value"])
        def target(X: np.ndarray):
            return np.full((X.shape[0], 1), c, dtype=np.float64)
        return target

    if ttype == "expr":
        expr_str = spec["expr"]
        syms = {v: sp.Symbol(v) for v in var_names}
        expr = sp.sympify(expr_str, locals={**syms, "pi": sp.pi})
        f = sp.lambdify([syms[v] for v in var_names], expr, "numpy")  # vectorized

        def target(X: np.ndarray):
            cols = [X[:, i] for i in range(X.shape[1])]
            y = f(*cols)
            y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
            return y
        return target

    raise ValueError(f"Unknown target type: {ttype}")


def load_config(path: str) -> LoadedConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)  # safe loader [web:406]

    var_names = [v.strip() for v in cfg["vars"].split(",") if v.strip()]
    bounds = {k: (float(v[0]), float(v[1])) for k, v in cfg["domain"]["bounds"].items()}

    consts_cfg = cfg.get("constants", {}) or {}
    consts = {sp.Symbol(k): float(val) for k, val in consts_cfg.items()}

    ic_specs = cfg.get("conditions", {}).get("ic", []) or []
    bc_specs = cfg.get("conditions", {}).get("bc", []) or []

    ic = []
    for item in ic_specs:
        sampler = _sampler_from_spec(item["sampler"], bounds, var_names)
        target = _target_from_spec(item["target"], var_names)
        ic.append((sampler, target))

    bc = []
    for item in bc_specs:
        sampler = _sampler_from_spec(item["sampler"], bounds, var_names)
        target = _target_from_spec(item["target"], var_names)
        bc.append((sampler, target))

    output = cfg.get("output", {}) or {}
    pinn = cfg.get("pinn", {}) or {}

    return LoadedConfig(
        var_names=var_names,
        bounds=bounds,
        consts=consts,
        ic=ic,
        bc=bc,
        output=output,
        pinn=pinn,
    )
