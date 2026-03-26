from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
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


def exp_kernel(t: np.ndarray, a: float, gamma: float) -> np.ndarray:
    return a * np.exp(-gamma * t)


def stretched_exp_kernel(t: np.ndarray, a: float, gamma: float, beta: float) -> np.ndarray:
    return a * np.exp(-(gamma * t) ** beta)


def power_law_kernel(t: np.ndarray, a: float, lam: float, c: float) -> np.ndarray:
    return a * (t + c) ** (-lam)


def tempered_power_law_kernel(t: np.ndarray, a: float, lam: float, gamma: float, c: float) -> np.ndarray:
    return a * (t + c) ** (-lam) * np.exp(-gamma * t)


def gamma_kernel(t: np.ndarray, a: float, k: float, theta: float) -> np.ndarray:
    tp = np.maximum(t, 1e-12)
    return a * (tp ** (k - 1.0)) * np.exp(-tp / theta)


def _mse(y: np.ndarray, yfit: np.ndarray) -> float:
    return float(np.mean((y - yfit) ** 2))


def _complexity_penalty(n_params: int) -> float:
    return 0.01 * n_params


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
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds, maxfev=50000)
        y_fit = func(t, *popt)
        mse = _mse(y, y_fit)
        score = mse + _complexity_penalty(len(popt))
        params = {k: float(v) for k, v in zip(param_names, popt)}
        eq = build_equation_string(label, name, params)
        return FitResult(
            family=name,
            params=params,
            y_fit=y_fit,
            mse=mse,
            score=score,
            equation=eq,
            label=label,
        )
    except Exception:
        return None


def fit_signal_equation(t: np.ndarray, y: np.ndarray, label: str = "K") -> FitResult:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    t = t - np.min(t)
    if np.allclose(t, 0.0):
        t = np.linspace(0, 1, len(t))

    y = np.maximum(y, 1e-12)
    a0 = float(np.max(y))

    candidates = [
        _fit_candidate(
            "exponential", exp_kernel, t, y,
            p0=[a0, 0.1],
            bounds=([1e-12, 1e-6], [10 * a0 + 1.0, 10.0]),
            param_names=["a", "gamma"],
            label=label,
        ),
        _fit_candidate(
            "stretched_exponential", stretched_exp_kernel, t, y,
            p0=[a0, 0.1, 0.8],
            bounds=([1e-12, 1e-6, 0.1], [10 * a0 + 1.0, 10.0, 3.0]),
            param_names=["a", "gamma", "beta"],
            label=label,
        ),
        _fit_candidate(
            "power_law", power_law_kernel, t, y,
            p0=[a0, 1.1, 0.1],
            bounds=([1e-12, 0.1, 1e-6], [10 * a0 + 1.0, 5.0, 10.0]),
            param_names=["a", "lam", "c"],
            label=label,
        ),
        _fit_candidate(
            "tempered_power_law", tempered_power_law_kernel, t, y,
            p0=[a0, 1.0, 0.05, 0.1],
            bounds=([1e-12, 0.1, 1e-6, 1e-6], [10 * a0 + 1.0, 5.0, 5.0, 10.0]),
            param_names=["a", "lam", "gamma", "c"],
            label=label,
        ),
        _fit_candidate(
            "gamma_type", gamma_kernel, t, y,
            p0=[a0, 1.5, 1.0],
            bounds=([1e-12, 0.1, 1e-6], [10 * a0 + 1.0, 10.0, 50.0]),
            param_names=["a", "k", "theta"],
            label=label,
        ),
    ]

    valid = [c for c in candidates if c is not None]
    if not valid:
        raise RuntimeError(f"Equation fitting failed for label '{label}'.")
    valid.sort(key=lambda z: z.score)
    return valid[0]


def fit_all_signal_equations(t: np.ndarray, signals: Dict[str, np.ndarray]) -> Dict[str, FitResult]:
    label_map = {
        "velocity_kernel": "g",
        "retention_kernel": "h",
        "direct_kernel": "r",
        "velocity_signal": "g",
        "retention_signal": "h",
        "direct_kernel_signal": "r",
    }
    results = {}
    for name, y in signals.items():
        label = label_map.get(name, "K")
        results[name] = fit_signal_equation(t, y, label=label)
    return results
