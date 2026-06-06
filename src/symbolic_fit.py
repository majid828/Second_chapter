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
class SharedFamilyResult:
    best_family: str
    best_formula: str
    best_objective: float
    family_summary: pd.DataFrame
    best_site_parameters: pd.DataFrame
    all_site_parameters: pd.DataFrame


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


CANDIDATE_FAMILIES: Dict[str, Dict] = {
    "exponential": {"func": exp_kernel, "param_names": ["a", "gamma"], "complexity": 2, "formula": "K(t) = a exp(-gamma t)"},
    "stretched_exponential": {"func": stretched_exp_kernel, "param_names": ["a", "gamma", "beta"], "complexity": 3, "formula": "K(t) = a exp(-(gamma t)^beta)"},
    "power_law": {"func": power_law_kernel, "param_names": ["a", "lam", "c"], "complexity": 3, "formula": "K(t) = a (t + c)^(-lam)"},
    "tempered_power_law": {"func": tempered_power_law_kernel, "param_names": ["a", "lam", "gamma", "c"], "complexity": 4, "formula": "K(t) = a (t + c)^(-lam) exp(-gamma t)"},
    "gamma_type": {"func": gamma_kernel, "param_names": ["a", "k", "theta"], "complexity": 3, "formula": "K(t) = a t^(k-1) exp(-t/theta)"},
}


def _mse(y: np.ndarray, yfit: np.ndarray) -> float:
    return float(np.mean((y - yfit) ** 2))


def _sse(y: np.ndarray, yfit: np.ndarray) -> float:
    return float(np.sum((y - yfit) ** 2))


def _complexity_penalty(n_params: int) -> float:
    return 0.01 * n_params


def _prepare_t_y(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) < 3:
        raise ValueError("Need at least 3 valid data points for kernel fitting.")
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    t = t - np.min(t)
    if np.allclose(t, 0.0):
        t = np.linspace(0.0, 1.0, len(t))
    y = np.maximum(y, 1e-12)
    return t, y


def _initial_guess_and_bounds(family: str, t: np.ndarray, y: np.ndarray) -> Tuple[List[float], Tuple[List[float], List[float]]]:
    a0 = float(np.max(y))
    tmax = float(np.max(t)) if np.max(t) > 0 else 1.0
    if family == "exponential":
        return [a0, 1.0 / tmax], ([1e-12, 1e-8], [10.0 * a0 + 1.0, 100.0])
    if family == "stretched_exponential":
        return [a0, 1.0 / tmax, 0.8], ([1e-12, 1e-8, 0.05], [10.0 * a0 + 1.0, 100.0, 5.0])
    if family == "power_law":
        return [a0, 1.0, 0.1 * tmax + 1e-6], ([1e-12, 0.01, 1e-8], [10.0 * a0 + 1.0, 10.0, 10.0 * tmax + 10.0])
    if family == "tempered_power_law":
        return [a0, 1.0, 1.0 / tmax, 0.1 * tmax + 1e-6], ([1e-12, 0.01, 1e-8, 1e-8], [10.0 * a0 + 1.0, 10.0, 100.0, 10.0 * tmax + 10.0])
    if family == "gamma_type":
        return [a0, 1.5, 0.25 * tmax + 1e-6], ([1e-12, 0.05, 1e-8], [10.0 * a0 + 1.0, 20.0, 10.0 * tmax + 10.0])
    raise ValueError(f"Unknown family: {family}")


def build_equation_string(label: str, family: str, p: Dict[str, float]) -> str:
    if family == "exponential":
        return f"{label}(t) = {p['a']:.4g} * exp(-{p['gamma']:.4g} * t)"
    if family == "stretched_exponential":
        return f"{label}(t) = {p['a']:.4g} * exp(-({p['gamma']:.4g} * t)^({p['beta']:.4g}))"
    if family == "power_law":
        return f"{label}(t) = {p['a']:.4g} * (t + {p['c']:.4g})^(-{p['lam']:.4g})"
    if family == "tempered_power_law":
        return f"{label}(t) = {p['a']:.4g} * (t + {p['c']:.4g})^(-{p['lam']:.4g}) * exp(-{p['gamma']:.4g} * t)"
    if family == "gamma_type":
        return f"{label}(t) = {p['a']:.4g} * t^({p['k']:.4g} - 1) * exp(-t/{p['theta']:.4g})"
    return f"{label}(t) = unknown"


def _fit_candidate(name: str, func: Callable, t: np.ndarray, y: np.ndarray, p0: List[float], bounds: Tuple[List[float], List[float]], param_names: List[str], label: str) -> FitResult | None:
    try:
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds, maxfev=50000)
        y_fit = func(t, *popt)
        mse = _mse(y, y_fit)
        score = mse + _complexity_penalty(len(popt))
        params = {k: float(v) for k, v in zip(param_names, popt)}
        eq = build_equation_string(label, name, params)
        return FitResult(family=name, params=params, y_fit=y_fit, mse=mse, score=score, equation=eq, label=label)
    except Exception:
        return None


def fit_fixed_family(t: np.ndarray, y: np.ndarray, family: str, label: str = "K") -> FitResult:
    """Fit one specified family to one site kernel."""
    if family not in CANDIDATE_FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    t, y = _prepare_t_y(t, y)
    spec = CANDIDATE_FAMILIES[family]
    p0, bounds = _initial_guess_and_bounds(family, t, y)
    result = _fit_candidate(family, spec["func"], t, y, p0=p0, bounds=bounds, param_names=spec["param_names"], label=label)
    if result is None:
        raise RuntimeError(f"Fitting failed for family '{family}'.")
    return result


def fit_signal_equation(t: np.ndarray, y: np.ndarray, label: str = "K") -> FitResult:
    """Choose the best family for one signal at one site. Diagnostic only."""
    t, y = _prepare_t_y(t, y)
    candidates: List[FitResult] = []
    for family, spec in CANDIDATE_FAMILIES.items():
        try:
            p0, bounds = _initial_guess_and_bounds(family, t, y)
            result = _fit_candidate(family, spec["func"], t, y, p0=p0, bounds=bounds, param_names=spec["param_names"], label=label)
            if result is not None:
                candidates.append(result)
        except Exception:
            continue
    if not candidates:
        raise RuntimeError(f"Equation fitting failed for label '{label}'.")
    candidates.sort(key=lambda z: z.score)
    return candidates[0]


def fit_all_signal_equations(t: np.ndarray, signals: Dict[str, np.ndarray]) -> Dict[str, FitResult]:
    label_map = {"velocity_kernel": "g", "retention_kernel": "h", "direct_kernel": "r", "velocity_signal": "g", "retention_signal": "h", "direct_kernel_signal": "r"}
    return {name: fit_signal_equation(t, y, label=label_map.get(name, "K")) for name, y in signals.items()}


def fit_shared_family_across_sites(site_kernels: Dict[str, Dict[str, np.ndarray]], kernel_name: str = "retention_kernel", family_names: List[str] | None = None, complexity_weight: float = 0.01) -> SharedFamilyResult:
    """
    TRUE multi-site shared-family discovery.

    Implements:
        L_m = sum_i sum_j [Khat_i(t_j) - K_m(t_j; theta_i)]^2
        J_m = L_m + alpha * Complexity(m)
        m* = argmin_m J_m
    """
    if family_names is None:
        family_names = list(CANDIDATE_FAMILIES.keys())

    family_rows: List[Dict] = []
    all_param_rows: List[Dict] = []

    for family in family_names:
        if family not in CANDIDATE_FAMILIES:
            raise ValueError(f"Unknown family in family_names: {family}")
        total_sse = 0.0
        total_points = 0
        n_success = 0
        n_failed = 0

        for site_id, data in site_kernels.items():
            if "time" not in data or kernel_name not in data:
                raise KeyError(f"site_kernels['{site_id}'] must contain 'time' and '{kernel_name}'.")
            t = np.asarray(data["time"], dtype=float)
            y = np.asarray(data[kernel_name], dtype=float)
            try:
                t_clean, y_clean = _prepare_t_y(t, y)
                fit = fit_fixed_family(t_clean, y_clean, family=family, label="K")
                sse = _sse(y_clean, fit.y_fit)
                n = int(len(fit.y_fit))
                total_sse += sse
                total_points += n
                n_success += 1
                row = {"site": site_id, "kernel_name": kernel_name, "family": family, "sse": sse, "mse": fit.mse, "equation": fit.equation, "fit_status": "success"}
                row.update(fit.params)
                all_param_rows.append(row)
            except Exception as exc:
                n_failed += 1
                all_param_rows.append({"site": site_id, "kernel_name": kernel_name, "family": family, "sse": np.nan, "mse": np.nan, "equation": "", "fit_status": f"failed: {exc}"})

        complexity = CANDIDATE_FAMILIES[family]["complexity"]
        failure_penalty = 1e12 * n_failed
        objective = total_sse + complexity_weight * complexity + failure_penalty
        global_mse = total_sse / total_points if total_points > 0 else np.inf
        family_rows.append({"kernel_name": kernel_name, "family": family, "formula": CANDIDATE_FAMILIES[family]["formula"], "total_sse": total_sse, "global_mse": global_mse, "complexity": complexity, "complexity_weight": complexity_weight, "failure_penalty": failure_penalty, "objective": objective, "n_success": n_success, "n_failed": n_failed})

    family_summary = pd.DataFrame(family_rows).sort_values("objective").reset_index(drop=True)
    all_site_parameters = pd.DataFrame(all_param_rows)
    if family_summary.empty:
        raise RuntimeError("No shared-family results were produced.")
    best_family = str(family_summary.iloc[0]["family"])
    best_objective = float(family_summary.iloc[0]["objective"])
    best_formula = str(family_summary.iloc[0]["formula"])
    best_site_parameters = all_site_parameters[(all_site_parameters["family"] == best_family) & (all_site_parameters["fit_status"] == "success")].copy().reset_index(drop=True)
    return SharedFamilyResult(best_family=best_family, best_formula=best_formula, best_objective=best_objective, family_summary=family_summary, best_site_parameters=best_site_parameters, all_site_parameters=all_site_parameters)
