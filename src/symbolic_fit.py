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
# Candidate equations for individual site-level fitting.
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


def lognormal_kernel(t: np.ndarray, a: float, mu: float, sigma: float, shift: float) -> np.ndarray:
    tp = np.maximum(t + shift, 1e-12)
    return a * np.exp(-((np.log(tp) - mu) ** 2) / (2.0 * sigma**2)) / tp


def two_exp_kernel(t: np.ndarray, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    return a1 * np.exp(-b1 * t) + a2 * np.exp(-b2 * t)


# ---------------------------------------------------------------------
# Shared-structure symbolic-regression grammar.
# ---------------------------------------------------------------------
def sym_exp(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return exp_kernel(t, a, b)


def sym_power(t: np.ndarray, a: float, lam: float, c: float) -> np.ndarray:
    return power_law_kernel(t, a, lam, c)


def sym_stretched_exp(t: np.ndarray, a: float, b: float, beta: float) -> np.ndarray:
    return stretched_exp_kernel(t, a, b, beta)


def sym_tempered_power(t: np.ndarray, a: float, lam: float, b: float, c: float) -> np.ndarray:
    return tempered_power_law_kernel(t, a, lam, b, c)


def sym_gamma(t: np.ndarray, a: float, k: float, theta: float) -> np.ndarray:
    return gamma_kernel(t, a, k, theta)


def sym_lognormal(t: np.ndarray, a: float, mu: float, sigma: float, shift: float) -> np.ndarray:
    return lognormal_kernel(t, a, mu, sigma, shift)


def sym_two_exp(t: np.ndarray, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    return two_exp_kernel(t, a1, b1, a2, b2)


def sym_exp_plus_power(t: np.ndarray, a1: float, b1: float, a2: float, lam: float, c: float) -> np.ndarray:
    return a1 * np.exp(-b1 * t) + a2 * (t + c) ** (-lam)


def sym_tempered_plus_exp(t: np.ndarray, a1: float, lam: float, b1: float, c: float, a2: float, b2: float) -> np.ndarray:
    return a1 * (t + c) ** (-lam) * np.exp(-b1 * t) + a2 * np.exp(-b2 * t)


def sym_softplus_exp_exp(t: np.ndarray, a: float, c: float, b: float, q: float) -> np.ndarray:
    return a * np.log(c + np.exp(-b * np.exp(q * t)))


def _safe_tmax(tmax: float) -> float:
    return float(tmax) if np.isfinite(tmax) and tmax > 0 else 1.0


SYMBOLIC_STRUCTURES: Dict[str, Dict] = {
    "exp": {
        "func": sym_exp,
        "params": ["a", "b"],
        "p0": lambda a0, tmax: [a0, 1.0 / _safe_tmax(tmax)],
        "bounds": lambda a0, tmax: ([1e-12, 1e-8], [10 * a0 + 1.0, 100.0]),
        "complexity": 2,
        "formula": "K(t) = a exp(-b t)",
    },
    "power": {
        "func": sym_power,
        "params": ["a", "lam", "c"],
        "p0": lambda a0, tmax: [a0, 0.8, 0.05 * _safe_tmax(tmax) + 1e-6],
        "bounds": lambda a0, tmax: ([1e-12, 0.01, 1e-8], [10 * a0 + 1.0, 10.0, 10 * _safe_tmax(tmax) + 10]),
        "complexity": 3,
        "formula": "K(t) = a (t+c)^(-lambda)",
    },
    "stretched_exp": {
        "func": sym_stretched_exp,
        "params": ["a", "b", "beta"],
        "p0": lambda a0, tmax: [a0, 1.0 / _safe_tmax(tmax), 0.8],
        "bounds": lambda a0, tmax: ([1e-12, 1e-8, 0.05], [10 * a0 + 1.0, 100.0, 5.0]),
        "complexity": 3,
        "formula": "K(t) = a exp(-(b t)^beta)",
    },
    "tempered_power": {
        "func": sym_tempered_power,
        "params": ["a", "lam", "b", "c"],
        "p0": lambda a0, tmax: [a0, 0.8, 1.0 / _safe_tmax(tmax), 0.05 * _safe_tmax(tmax) + 1e-6],
        "bounds": lambda a0, tmax: ([1e-12, 0.01, 1e-8, 1e-8], [10 * a0 + 1.0, 10.0, 100.0, 10 * _safe_tmax(tmax) + 10]),
        "complexity": 4,
        "formula": "K(t) = a (t+c)^(-lambda) exp(-b t)",
    },
    "gamma_type": {
        "func": sym_gamma,
        "params": ["a", "k", "theta"],
        "p0": lambda a0, tmax: [a0, 2.0, 0.25 * _safe_tmax(tmax) + 1e-6],
        "bounds": lambda a0, tmax: ([1e-12, 0.05, 1e-8], [10 * a0 + 1.0, 25.0, 10 * _safe_tmax(tmax) + 10]),
        "complexity": 3,
        "formula": "K(t) = a t^(k-1) exp(-t/theta)",
    },
    "lognormal": {
        "func": sym_lognormal,
        "params": ["a", "mu", "sigma", "shift"],
        "p0": lambda a0, tmax: [a0, np.log(0.25 * _safe_tmax(tmax) + 1e-3), 0.8, 0.02 * _safe_tmax(tmax) + 1e-6],
        "bounds": lambda a0, tmax: ([1e-12, -20.0, 0.05, 1e-8], [10 * a0 + 1.0, 20.0, 5.0, 10 * _safe_tmax(tmax) + 10]),
        "complexity": 4,
        "formula": "K(t) = a exp(-(log(t+s)-mu)^2/(2 sigma^2))/(t+s)",
    },
    "two_exp": {
        "func": sym_two_exp,
        "params": ["a1", "b1", "a2", "b2"],
        "p0": lambda a0, tmax: [0.7 * a0, 1.0 / _safe_tmax(tmax), 0.3 * a0, 5.0 / _safe_tmax(tmax)],
        "bounds": lambda a0, tmax: ([1e-12, 1e-8, 1e-12, 1e-8], [10 * a0 + 1.0, 100.0, 10 * a0 + 1.0, 100.0]),
        "complexity": 4,
        "formula": "K(t) = a1 exp(-b1 t) + a2 exp(-b2 t)",
    },
    "exp_plus_power": {
        "func": sym_exp_plus_power,
        "params": ["a1", "b1", "a2", "lam", "c"],
        "p0": lambda a0, tmax: [0.5 * a0, 1.0 / _safe_tmax(tmax), 0.5 * a0, 0.8, 0.05 * _safe_tmax(tmax) + 1e-6],
        "bounds": lambda a0, tmax: ([1e-12, 1e-8, 1e-12, 0.01, 1e-8], [10 * a0 + 1.0, 100.0, 10 * a0 + 1.0, 10.0, 10 * _safe_tmax(tmax) + 10]),
        "complexity": 5,
        "formula": "K(t) = a1 exp(-b1 t) + a2 (t+c)^(-lambda)",
    },
    "tempered_plus_exp": {
        "func": sym_tempered_plus_exp,
        "params": ["a1", "lam", "b1", "c", "a2", "b2"],
        "p0": lambda a0, tmax: [0.7 * a0, 0.8, 1.0 / _safe_tmax(tmax), 0.05 * _safe_tmax(tmax) + 1e-6, 0.3 * a0, 5.0 / _safe_tmax(tmax)],
        "bounds": lambda a0, tmax: ([1e-12, 0.01, 1e-8, 1e-8, 1e-12, 1e-8], [10 * a0 + 1.0, 10.0, 100.0, 10 * _safe_tmax(tmax) + 10, 10 * a0 + 1.0, 100.0]),
        "complexity": 6,
        "formula": "K(t) = a1 (t+c)^(-lambda) exp(-b1 t) + a2 exp(-b2 t)",
    },
    "softplus_exp_exp": {
        "func": sym_softplus_exp_exp,
        "params": ["a", "c", "b", "q"],
        "p0": lambda a0, tmax: [a0, 1.01, 0.1, 1.0 / _safe_tmax(tmax)],
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

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 1e-12)
    return t, y


def _mse(y: np.ndarray, yfit: np.ndarray) -> float:
    return float(np.mean((y - yfit) ** 2))


def _sse(y: np.ndarray, yfit: np.ndarray) -> float:
    return float(np.sum((y - yfit) ** 2))


def _aic_like_score(y: np.ndarray, yfit: np.ndarray, n_params: int) -> float:
    """
    Scale-aware model score.

    The old fixed penalty mse + 0.01*n_params was too large relative to kernel
    MSE and often selected over-simple exponentials. This score uses a standard
    AIC-like term so better gamma/lognormal fits can be selected when justified.
    """
    n = max(len(y), 1)
    sse = max(_sse(y, yfit), 1e-18)
    return float(n * np.log(sse / n) + 2.0 * n_params)


def build_equation_string(label: str, family: str, p: Dict[str, float]) -> str:
    if family in {"exponential", "exp"}:
        return f"{label}(t) = {p.get('a', 0):.4g} * exp(-{p.get('gamma', p.get('b', 0)):.4g} * t)"
    if family in {"stretched_exponential", "stretched_exp"}:
        return f"{label}(t) = {p.get('a', 0):.4g} * exp(-({p.get('gamma', p.get('b', 0)):.4g} * t)^({p.get('beta', 0):.4g}))"
    if family in {"power_law", "power"}:
        return f"{label}(t) = {p.get('a', 0):.4g} * (t + {p.get('c', 0):.4g})^(-{p.get('lam', 0):.4g})"
    if family in {"tempered_power_law", "tempered_power"}:
        return f"{label}(t) = {p.get('a', 0):.4g} * (t + {p.get('c', 0):.4g})^(-{p.get('lam', 0):.4g}) * exp(-{p.get('gamma', p.get('b', 0)):.4g} * t)"
    if family == "gamma_type":
        return f"{label}(t) = {p.get('a', 0):.4g} * t^({p.get('k', 0):.4g} - 1) * exp(-t/{p.get('theta', 0):.4g})"
    if family == "lognormal":
        return f"{label}(t) = {p.get('a', 0):.4g} * exp(-(log(t + {p.get('shift', 0):.4g}) - {p.get('mu', 0):.4g})^2 / (2 * {p.get('sigma', 0):.4g}^2)) / (t + {p.get('shift', 0):.4g})"
    if family == "two_exp":
        return f"{label}(t) = {p.get('a1', 0):.4g} * exp(-{p.get('b1', 0):.4g} * t) + {p.get('a2', 0):.4g} * exp(-{p.get('b2', 0):.4g} * t)"
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
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds, maxfev=120000)
        y_fit = np.nan_to_num(func(t, *popt), nan=0.0, posinf=0.0, neginf=0.0)
        y_fit = np.maximum(y_fit, 0.0)
        mse = _mse(y, y_fit)
        score = _aic_like_score(y, y_fit, len(popt))
        params = {k: float(v) for k, v in zip(param_names, popt)}
        eq = build_equation_string(label, name, params)
        return FitResult(name, params, y_fit, mse, score, eq, label)
    except Exception:
        return None


def fit_signal_equation(t: np.ndarray, y: np.ndarray, label: str = "K") -> FitResult:
    """Fit the best interpretable equation to one recovered signal/kernel."""
    t, y = _prepare_t_y(t, y)
    a0 = float(np.max(y))
    tmax = _safe_tmax(float(np.max(t)))

    simple_map = {
        "exponential": {
            "func": exp_kernel,
            "params": ["a", "gamma"],
            "p0": [a0, 1.0 / tmax],
            "bounds": ([1e-12, 1e-8], [10 * a0 + 1.0, 100.0]),
        },
        "stretched_exponential": {
            "func": stretched_exp_kernel,
            "params": ["a", "gamma", "beta"],
            "p0": [a0, 1.0 / tmax, 0.8],
            "bounds": ([1e-12, 1e-8, 0.05], [10 * a0 + 1.0, 100.0, 5.0]),
        },
        "power_law": {
            "func": power_law_kernel,
            "params": ["a", "lam", "c"],
            "p0": [a0, 0.8, 0.05 * tmax + 1e-6],
            "bounds": ([1e-12, 0.01, 1e-8], [10 * a0 + 1.0, 10.0, 10 * tmax + 10]),
        },
        "tempered_power_law": {
            "func": tempered_power_law_kernel,
            "params": ["a", "lam", "gamma", "c"],
            "p0": [a0, 0.8, 1.0 / tmax, 0.05 * tmax + 1e-6],
            "bounds": ([1e-12, 0.01, 1e-8, 1e-8], [10 * a0 + 1.0, 10.0, 100.0, 10 * tmax + 10]),
        },
        "gamma_type": {
            "func": gamma_kernel,
            "params": ["a", "k", "theta"],
            "p0": [a0, 2.0, 0.25 * tmax + 1e-6],
            "bounds": ([1e-12, 0.05, 1e-8], [10 * a0 + 1.0, 25.0, 10 * tmax + 10]),
        },
        "lognormal": {
            "func": lognormal_kernel,
            "params": ["a", "mu", "sigma", "shift"],
            "p0": [a0, np.log(0.25 * tmax + 1e-3), 0.8, 0.02 * tmax + 1e-6],
            "bounds": ([1e-12, -20.0, 0.05, 1e-8], [10 * a0 + 1.0, 20.0, 5.0, 10 * tmax + 10]),
        },
        "two_exp": {
            "func": two_exp_kernel,
            "params": ["a1", "b1", "a2", "b2"],
            "p0": [0.7 * a0, 1.0 / tmax, 0.3 * a0, 5.0 / tmax],
            "bounds": ([1e-12, 1e-8, 1e-12, 1e-8], [10 * a0 + 1.0, 100.0, 10 * a0 + 1.0, 100.0]),
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
    tmax = _safe_tmax(float(np.max(t)))

    spec = SYMBOLIC_STRUCTURES[structure_name]
    p0 = spec["p0"](a0, tmax)
    bounds = spec["bounds"](a0, tmax)

    popt, _ = curve_fit(spec["func"], t, y, p0=p0, bounds=bounds, maxfev=120000)
    y_fit = np.maximum(np.nan_to_num(spec["func"](t, *popt), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    params = {name: float(value) for name, value in zip(spec["params"], popt)}
    return params, y_fit, _sse(y, y_fit), _mse(y, y_fit)


def discover_shared_symbolic_structure(
    site_kernels: Dict[str, Dict[str, np.ndarray]],
    kernel_name: str = "retention_kernel",
    structures: List[str] | None = None,
    complexity_weight: float = 2.0,
) -> SharedSymbolicResult:
    """
    Shared-structure search across sites.

    Each candidate structure is fitted to every site separately. The selected
    structure minimizes global SSE plus a complexity penalty. Parameters remain
    site-specific, but the mathematical structure is shared.
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
                    "family": structure,
                    "formula": SYMBOLIC_STRUCTURES[structure]["formula"],
                    "equation": SYMBOLIC_STRUCTURES[structure]["formula"],
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
                    "family": structure,
                    "formula": SYMBOLIC_STRUCTURES[structure]["formula"],
                    "equation": SYMBOLIC_STRUCTURES[structure]["formula"],
                    "sse": np.nan,
                    "mse": np.nan,
                    "fit_status": f"failed: {exc}",
                })

        complexity = SYMBOLIC_STRUCTURES[structure]["complexity"]
        failure_penalty = 1e12 * n_failed
        global_mse = total_sse / total_points if total_points > 0 else np.inf
        objective = total_sse + complexity_weight * complexity + failure_penalty

        summary_rows.append({
            "kernel_name": kernel_name,
            "structure": structure,
            "family": structure,
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
