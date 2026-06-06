from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class FitResult:
    family: str
    params: Dict[str, float]
    y_fit: np.ndarray
    mse: float
    score: float
    equation: str
    label: str


@dataclass
class SharedSymbolicResult:
    best_structure: str
    best_formula: str
    best_objective: float
    structure_summary: pd.DataFrame
    best_site_parameters: pd.DataFrame
    all_site_parameters: pd.DataFrame


# ---------------------------------------------------------------------
# Individual site-level candidate equations.
# These are kept for compatibility with your existing pipeline.
# ---------------------------------------------------------------------
def exp_kernel(t: np.ndarray, a: float, gamma: float) -> np.ndarray:
    return a * np.exp(-gamma * t)


def stretched_exp_kernel(t: np.ndarray, a: float, gamma: float, beta: float) -> np.ndarray:
    return a * np.exp(-((gamma * t) ** beta))


def power_law_kernel(t: np.ndarray, a: float, lam: float, c: float) -> np.ndarray:
    return a * (t + c) ** (-lam)


def tempered_power_law_kernel(t: np.ndarray, a: float, lam: float, gamma: float, c: float) -> np.ndarray:
    return a * (t + c) ** (-lam) * np.exp(-gamma * t)


def gamma_kernel(t: np.ndarray, a: float, k: float, theta: float) -> np.ndarray:
    tp = np.maximum(t, 1e-12)
    return a * (tp ** (k - 1.0)) * np.exp(-tp / theta)


# ---------------------------------------------------------------------
# Shared-structure symbolic-regression grammar.
#
# This is Approach B in practical form:
# - The code searches symbolic structures.
# - Each symbolic structure is fitted to every site separately.
# - The structure is shared across all sites.
# - Constants/parameters are site-specific.
# - The selected structure minimizes global multi-site objective.
#
# This is not PySR black-box symbolic regression. It is a controlled
# symbolic-structure search, which is easier to defend scientifically.
# ---------------------------------------------------------------------
def sym_exp(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(-b * t)


def sym_power(t: np.ndarray, a: float, lam: float, c: float) -> np.ndarray:
    return a * (t + c) ** (-lam)


def sym_stretched_exp(t: np.ndarray, a: float, b: float, beta: float) -> np.ndarray:
    return a * np.exp(-((b * t) ** beta))


def sym_tempered_power(t: np.ndarray, a: float, lam: float, b: float, c: float) -> np.ndarray:
    return a * (t + c) ** (-lam) * np.exp(-b * t)


def sym_gamma(t: np.ndarray, a: float, k: float, theta: float) -> np.ndarray:
    tp = np.maximum(t, 1e-12)
    return a * (tp ** (k - 1.0)) * np.exp(-tp / theta)


def sym_two_exp(t: np.ndarray, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    return a1 * np.exp(-b1 * t) + a2 * np.exp(-b2 * t)


def sym_exp_plus_power(t: np.ndarray, a1: float, b1: float, a2: float, lam: float, c: float) -> np.ndarray:
    return a1 * np.exp(-b1 * t) + a2 * (t + c) ** (-lam)


def sym_tempered_plus_exp(t: np.ndarray, a1: float, lam: float, b1: float, c: float, a2: float, b2: float) -> np.ndarray:
    return a1 * (t + c) ** (-lam) * np.exp(-b1 * t) + a2 * np.exp(-b2 * t)


def sym_softplus_exp_exp(t: np.ndarray, a: float, c: float, b: float, q: float) -> np.ndarray:
    # Similar in spirit to the equation shown in the hypothesis draft:
    # log(c + exp(-b exp(q t))). We multiply by a for amplitude.
    return a * np.log(c + np.exp(-b * np.exp(q * t)))


SYMBOLIC_STRUCTURES: Dict[str, Dict] = {
    "exp": {
        "func": sym_exp,
        "params": ["a", "b"],
        "p0": lambda a0, tmax: [a0, 1.0 / tmax],
        "bounds": lambda a0, tmax: ([1e-12, 1e-8], [10 * a0 + 1.0, 100.0]),
        "complexity": 2,
        "formula": "K(t) = a exp(-b t)",
    },
    "power": {
        "func": sym_power,
        "params": ["a", "lam", "c"],
        "p0": lambda a0, tmax: [a0, 0.8, 0.05 * tmax + 1e-6],
        "bounds": lambda a0, tmax: ([1e-12, 0.01, 1e-8], [10 * a0 + 1.0, 10.0, 10 * tmax + 10]),
        "complexity": 3,
        "formula": "K(t) = a (t+c)^(-lambda)",
    },
    "stretched_exp": {
        "func": sym_stretched_exp,
        "params": ["a", "b", "beta"],
        "p0": lambda a0, tmax: [a0, 1.0 / tmax, 0.8],
        "bounds": lambda a0, tmax: ([1e-12, 1e-8, 0.05], [10 * a0 + 1.0, 100.0, 5.0]),
        "complexity": 3,
        "formula": "K(t) = a exp(-(b t)^beta)",
    },
    "tempered_power": {
        "func": sym_tempered_power,
        "params": ["a", "lam", "b", "c"],
        "p0": lambda a0, tmax: [a0, 0.8, 1.0 / tmax, 0.05 * tmax + 1e-6],
        "bounds": lambda a0, tmax: ([1e-12, 0.01, 1e-8, 1e-8], [10 * a0 + 1.0, 10.0, 100.0, 10 * tmax + 10]),
        "complexity": 4,
        "formula": "K(t) = a (t+c)^(-lambda) exp(-b t)",
    },
    "gamma_type": {
        "func": sym_gamma,
        "params": ["a", "k", "theta"],
        "p0": lambda a0, tmax: [a0, 1.5, 0.25 * tmax + 1e-6],
        "bounds": lambda a0, tmax: ([1e-12, 0.05, 1e-8], [10 * a0 + 1.0, 20.0, 10 * tmax + 10]),
        "complexity": 3,
        "formula": "K(t) = a t^(k-1) exp(-t/theta)",
    },
    "two_exp": {
        "func": sym_two_exp,
        "params": ["a1", "b1", "a2", "b2"],
        "p0": lambda a0, tmax: [0.7 * a0, 1.0 / tmax, 0.3 * a0, 5.0 / tmax],
        "bounds": lambda a0, tmax: ([1e-12, 1e-8, 1e-12, 1e-8], [10 * a0 + 1.0, 100.0, 10 * a0 + 1.0, 100.0]),
        "complexity": 4,
        "formula": "K(t) = a1 exp(-b1 t) + a2 exp(-b2 t)",
    },
    "exp_plus_power": {
        "func": sym_exp_plus_power,
        "params": ["a1", "b1", "a2", "lam", "c"],
        "p0": lambda a0, tmax: [0.5 * a0, 1.0 / tmax, 0.5 * a0, 0.8, 0.05 * tmax + 1e-6],
        "bounds": lambda a0, tmax: ([1e-12, 1e-8, 1e-12, 0.01, 1e-8], [10 * a0 + 1.0, 100.0, 10 * a0 + 1.0, 10.0, 10 * tmax + 10]),
        "complexity": 5,
        "formula": "K(t) = a1 exp(-b1 t) + a2 (t+c)^(-lambda)",
    },
    "tempered_plus_exp": {
        "func": sym_tempered_plus_exp,
        "params": ["a1", "lam", "b1", "c", "a2", "b2"],
        "p0": lambda a0, tmax: [0.7 * a0, 0.8, 1.0 / tmax, 0.05 * tmax + 1e-6, 0.3 * a0, 5.0 / tmax],
        "bounds": lambda a0, tmax: ([1e-12, 0.01, 1e-8, 1e-8, 1e-12, 1e-8], [10 * a0 + 1.0, 10.0, 100.0, 10 * tmax + 10, 10 * a0 + 1.0, 100.0]),
        "complexity": 6,
        "formula": "K(t) = a1 (t+c)^(-lambda) exp(-b1 t) + a2 exp(-b2 t)",
    },
    "softplus_exp_exp": {
        "func": sym_softplus_exp_exp,
        "params": ["a", "c", "b", "q"],
        "p0": lambda a0, tmax: [a0, 1.01, 0.1, 1.0 / tmax],
        "bounds": lambda a0, tmax: ([1e-12, 1.000001, 1e-8, 1e-8], [10 * a0 + 1.0, 10.0, 100.0, 100.0]),
        "complexity": 4,
        "formula": "K(t) = a log(c + exp(-b exp(q t)))",
    },
}


def _prepare_t_y(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    if len(t) < 4:
        raise ValueError("Need at least 4 valid points for symbolic fitting.")

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    t = t - np.min(t)
    if np.max(t) <= 0:
        t = np.linspace(0.0, 1.0, len(t))

    y = np.maximum(y, 1e-12)
    return t, y


def _mse(y: np.ndarray, yfit: np.ndarray) -> float:
    return float(np.mean((y - yfit) ** 2))


def _sse(y: np.ndarray, yfit: np.ndarray) -> float:
    return float(np.sum((y - yfit) ** 2))


def _complexity_penalty(n_params: int) -> float:
    return 0.01 * n_params


def build_equation_string(label: str, family: str, p: Dict[str, float]) -> str:
    if family == "exponential" or family == "exp":
        return f"{label}(t) = {p.get('a', 0):.4g} * exp(-{p.get('gamma', p.get('b', 0)):.4g} * t)"
    if family == "stretched_exponential" or family == "stretched_exp":
        return f"{label}(t) = {p.get('a', 0):.4g} * exp(-({p.get('gamma', p.get('b', 0)):.4g} * t)^({p.get('beta', 0):.4g}))"
    if family == "power_law" or family == "power":
        return f"{label}(t) = {p.get('a', 0):.4g} * (t + {p.get('c', 0):.4g})^(-{p.get('lam', 0):.4g})"
    if family == "tempered_power_law" or family == "tempered_power":
        return f"{label}(t) = {p.get('a', 0):.4g} * (t + {p.get('c', 0):.4g})^(-{p.get('lam', 0):.4g}) * exp(-{p.get('gamma', p.get('b', 0)):.4g} * t)"
    if family == "gamma_type":
        return f"{label}(t) = {p.get('a', 0):.4g} * t^({p.get('k', 0):.4g} - 1) * exp(-t/{p.get('theta', 0):.4g})"
    return f"{label}(t) = {family}"


def _fit_candidate(
    name: str,
    func: Callable,
    t: np.ndarray,
    y: np.ndarray,
    p0: List[float],
    bounds: Tuple[List[float], List[float]],
    param_names: List[str],
    label: str,
) -> FitResult | None:
    try:
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds, maxfev=80000)
        y_fit = func(t, *popt)
        mse = _mse(y, y_fit)
        score = mse + _complexity_penalty(len(popt))
        params = {k: float(v) for k, v in zip(param_names, popt)}
        eq = build_equation_string(label, name, params)
        return FitResult(name, params, y_fit, mse, score, eq, label)
    except Exception:
        return None


def fit_signal_equation(t: np.ndarray, y: np.ndarray, label: str = "K") -> FitResult:
    # Compatibility function for individual per-site fitting.
    t, y = _prepare_t_y(t, y)
    a0 = float(np.max(y))
    tmax = float(np.max(t)) if np.max(t) > 0 else 1.0

    simple_map = {
        "exponential": {
            "func": exp_kernel, "params": ["a", "gamma"],
            "p0": [a0, 1.0 / tmax],
            "bounds": ([1e-12, 1e-8], [10 * a0 + 1.0, 100.0]),
        },
        "stretched_exponential": {
            "func": stretched_exp_kernel, "params": ["a", "gamma", "beta"],
            "p0": [a0, 1.0 / tmax, 0.8],
            "bounds": ([1e-12, 1e-8, 0.05], [10 * a0 + 1.0, 100.0, 5.0]),
        },
        "power_law": {
            "func": power_law_kernel, "params": ["a", "lam", "c"],
            "p0": [a0, 0.8, 0.05 * tmax + 1e-6],
            "bounds": ([1e-12, 0.01, 1e-8], [10 * a0 + 1.0, 10.0, 10 * tmax + 10]),
        },
        "tempered_power_law": {
            "func": tempered_power_law_kernel, "params": ["a", "lam", "gamma", "c"],
            "p0": [a0, 0.8, 1.0 / tmax, 0.05 * tmax + 1e-6],
            "bounds": ([1e-12, 0.01, 1e-8, 1e-8], [10 * a0 + 1.0, 10.0, 100.0, 10 * tmax + 10]),
        },
        "gamma_type": {
            "func": gamma_kernel, "params": ["a", "k", "theta"],
            "p0": [a0, 1.5, 0.25 * tmax + 1e-6],
            "bounds": ([1e-12, 0.05, 1e-8], [10 * a0 + 1.0, 20.0, 10 * tmax + 10]),
        },
    }

    candidates = []
    for name, spec in simple_map.items():
        r = _fit_candidate(name, spec["func"], t, y, spec["p0"], spec["bounds"], spec["params"], label)
        if r is not None:
            candidates.append(r)

    if not candidates:
        raise RuntimeError(f"Equation fitting failed for label '{label}'.")

    candidates.sort(key=lambda z: z.score)
    return candidates[0]


def fit_all_signal_equations(t: np.ndarray, signals: Dict[str, np.ndarray]) -> Dict[str, FitResult]:
    label_map = {
        "velocity_kernel": "g",
        "retention_kernel": "h",
        "direct_kernel": "r",
        "velocity_signal": "g",
        "retention_signal": "h",
        "direct_kernel_signal": "r",
    }
    return {name: fit_signal_equation(t, y, label=label_map.get(name, "K")) for name, y in signals.items()}


def _fit_symbolic_structure_to_one_site(
    t: np.ndarray,
    y: np.ndarray,
    structure_name: str,
) -> Tuple[Dict[str, float], np.ndarray, float, float]:
    t, y = _prepare_t_y(t, y)
    a0 = float(np.max(y))
    tmax = float(np.max(t)) if np.max(t) > 0 else 1.0

    spec = SYMBOLIC_STRUCTURES[structure_name]
    p0 = spec["p0"](a0, tmax)
    bounds = spec["bounds"](a0, tmax)

    popt, _ = curve_fit(spec["func"], t, y, p0=p0, bounds=bounds, maxfev=100000)
    y_fit = spec["func"](t, *popt)
    params = {name: float(value) for name, value in zip(spec["params"], popt)}
    return params, y_fit, _sse(y, y_fit), _mse(y, y_fit)


def discover_shared_symbolic_structure(
    site_kernels: Dict[str, Dict[str, np.ndarray]],
    kernel_name: str = "retention_kernel",
    structures: List[str] | None = None,
    complexity_weight: float = 0.01,
) -> SharedSymbolicResult:
    """
    Approach B: symbolic regression with shared structure.

    It implements:

        Khat_i(t) ≈ Phi(t; theta_i, S)

    where:
        S       = shared symbolic structure discovered by global search
        theta_i = site-specific parameter vector

    For each candidate symbolic structure S:
        1. Fit S to each site separately.
        2. Sum the loss across all sites.
        3. Add a complexity penalty.
        4. Select the structure with the smallest global objective.

    This is different from individual fitting because one structure is selected
    globally for all sites.
    """
    if structures is None:
        structures = list(SYMBOLIC_STRUCTURES.keys())

    summary_rows = []
    param_rows = []

    for structure in structures:
        if structure not in SYMBOLIC_STRUCTURES:
            raise ValueError(f"Unknown symbolic structure: {structure}")

        total_sse = 0.0
        total_points = 0
        n_success = 0
        n_failed = 0

        for site_id, data in site_kernels.items():
            try:
                t = np.asarray(data["time"], dtype=float)
                y = np.asarray(data[kernel_name], dtype=float)
                params, y_fit, sse, mse = _fit_symbolic_structure_to_one_site(t, y, structure)
                t_clean, y_clean = _prepare_t_y(t, y)

                total_sse += sse
                total_points += len(y_clean)
                n_success += 1

                row = {
                    "site": site_id,
                    "kernel_name": kernel_name,
                    "structure": structure,
                    "formula": SYMBOLIC_STRUCTURES[structure]["formula"],
                    "sse": sse,
                    "mse": mse,
                    "fit_status": "success",
                }
                row.update(params)
                param_rows.append(row)

            except Exception as exc:
                n_failed += 1
                param_rows.append({
                    "site": site_id,
                    "kernel_name": kernel_name,
                    "structure": structure,
                    "formula": SYMBOLIC_STRUCTURES[structure]["formula"],
                    "sse": np.nan,
                    "mse": np.nan,
                    "fit_status": f"failed: {exc}",
                })

        complexity = SYMBOLIC_STRUCTURES[structure]["complexity"]
        failure_penalty = 1e12 * n_failed
        objective = total_sse + complexity_weight * complexity + failure_penalty
        global_mse = total_sse / total_points if total_points > 0 else np.inf

        summary_rows.append({
            "kernel_name": kernel_name,
            "structure": structure,
            "formula": SYMBOLIC_STRUCTURES[structure]["formula"],
            "total_sse": total_sse,
            "global_mse": global_mse,
            "complexity": complexity,
            "complexity_weight": complexity_weight,
            "failure_penalty": failure_penalty,
            "objective": objective,
            "n_success": n_success,
            "n_failed": n_failed,
        })

    structure_summary = pd.DataFrame(summary_rows).sort_values("objective").reset_index(drop=True)
    all_site_parameters = pd.DataFrame(param_rows)

    best_structure = str(structure_summary.iloc[0]["structure"])
    best_formula = str(structure_summary.iloc[0]["formula"])
    best_objective = float(structure_summary.iloc[0]["objective"])

    best_site_parameters = (
        all_site_parameters[
            (all_site_parameters["structure"] == best_structure)
            & (all_site_parameters["fit_status"] == "success")
        ]
        .copy()
        .reset_index(drop=True)
    )

    return SharedSymbolicResult(
        best_structure=best_structure,
        best_formula=best_formula,
        best_objective=best_objective,
        structure_summary=structure_summary,
        best_site_parameters=best_site_parameters,
        all_site_parameters=all_site_parameters,
    )
