from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.optimize import lsq_linear

from .preprocess import SmoothedSeries


@dataclass
class EffectiveSignals:
    time: np.ndarray
    btc_smooth: np.ndarray
    velocity_signal: np.ndarray
    retention_signal: np.ndarray
    direct_kernel_signal: np.ndarray


@dataclass
class SnapshotMoments:
    snapshot_times: np.ndarray
    centroids: np.ndarray
    spreads: np.ndarray
    masses: np.ndarray


def compute_snapshot_moments(snapshot_profiles: Dict[float, SmoothedSeries]) -> SnapshotMoments:
    times, centroids, spreads, masses = [], [], [], []
    for t, prof in sorted(snapshot_profiles.items()):
        x = np.asarray(prof.x_uniform, dtype=float)
        c = np.maximum(np.asarray(prof.y_smooth, dtype=float), 0.0)
        mass = np.trapezoid(c, x)
        if mass <= 0:
            centroid, spread = np.nan, np.nan
        else:
            centroid = np.trapezoid(x * c, x) / mass
            variance = np.trapezoid(((x - centroid) ** 2) * c, x) / mass
            spread = np.sqrt(max(variance, 0.0))
        times.append(float(t))
        centroids.append(float(centroid) if np.isfinite(centroid) else np.nan)
        spreads.append(float(spread) if np.isfinite(spread) else np.nan)
        masses.append(float(mass))
    return SnapshotMoments(np.asarray(times, dtype=float), np.asarray(centroids, dtype=float), np.asarray(spreads, dtype=float), np.asarray(masses, dtype=float))


def _normalize_pdf(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0.0)
    area = np.trapezoid(y, x)
    if area <= 0:
        return np.zeros_like(y)
    return y / area


def _estimate_advective_kernel(t: np.ndarray, c: np.ndarray, early_fraction: float = 0.4) -> np.ndarray:
    c_norm = _normalize_pdf(c, t)
    n = len(c_norm)
    if n == 0:
        return c_norm
    cutoff = max(3, int(np.ceil(early_fraction * n)))
    cutoff = min(cutoff, n)
    g = np.zeros_like(c_norm)
    g[:cutoff] = c_norm[:cutoff]
    if len(g) >= 3:
        g2 = g.copy()
        g2[1:-1] = 0.25 * g[:-2] + 0.5 * g[1:-1] + 0.25 * g[2:]
        g = g2
    return _normalize_pdf(g, t)


def _build_convolution_matrix(g: np.ndarray, dt: float) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    n = len(g)
    G = np.zeros((n, n), dtype=float)
    for row in range(n):
        for col in range(row + 1):
            G[row, col] = g[row - col] * dt
    return G


def _deconvolve_retention(t: np.ndarray, f: np.ndarray, g: np.ndarray, alpha: float = 1e-3, nonnegative: bool = True, smooth_passes: int = 2) -> np.ndarray:
    """Solve min_h ||G h - f||^2 + alpha ||h||^2 using matrix regularized deconvolution."""
    t = np.asarray(t, dtype=float)
    f = _normalize_pdf(f, t)
    g = _normalize_pdf(g, t)
    if len(t) < 2:
        return _normalize_pdf(np.maximum(f, 0.0), t)
    dt = float(np.mean(np.diff(t)))
    G = _build_convolution_matrix(g, dt)
    n = len(f)
    A_aug = np.vstack([G, np.sqrt(alpha) * np.eye(n)])
    b_aug = np.concatenate([f, np.zeros(n)])
    if nonnegative:
        result = lsq_linear(A_aug, b_aug, bounds=(0.0, np.inf), max_iter=5000)
        h = result.x
    else:
        h, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    h = np.maximum(h, 0.0)
    for _ in range(max(0, smooth_passes)):
        if len(h) >= 3:
            h2 = h.copy()
            h2[1:-1] = 0.25 * h[:-2] + 0.5 * h[1:-1] + 0.25 * h[2:]
            h = np.maximum(h2, 0.0)
    return _normalize_pdf(h, t)


def build_effective_signals(btc_smoothed: SmoothedSeries, snapshot_moments: SnapshotMoments) -> EffectiveSignals:
    t = np.asarray(btc_smoothed.x_uniform, dtype=float)
    f = np.maximum(np.asarray(btc_smoothed.y_smooth, dtype=float), 0.0)
    f_norm = _normalize_pdf(f, t)
    g = _estimate_advective_kernel(t, f_norm, early_fraction=0.4)

    st = np.asarray(snapshot_moments.snapshot_times, dtype=float)
    cent = np.asarray(snapshot_moments.centroids, dtype=float)
    valid = np.isfinite(st) & np.isfinite(cent)
    if valid.sum() >= 2:
        vel_snap = np.gradient(cent[valid], st[valid])
        vel_scale = np.nanmedian(np.maximum(vel_snap, 1e-8))
        if np.isfinite(vel_scale) and vel_scale > 0:
            tau = np.maximum(t / (np.max(t) + 1e-12), 0.0)
            weight = np.exp(-tau / max(vel_scale, 1e-6))
            g = _normalize_pdf(g * weight, t)

    h = _deconvolve_retention(t, f_norm, g, alpha=1e-3, nonnegative=True, smooth_passes=2)
    tau = t / (np.max(t) + 1e-12)
    h = _normalize_pdf(h * (0.5 + tau), t)
    direct = np.maximum(f_norm - g, 0.0)
    direct = _normalize_pdf(direct, t) if np.any(direct > 0) else np.zeros_like(direct)
    return EffectiveSignals(time=t, btc_smooth=f_norm, velocity_signal=g, retention_signal=h, direct_kernel_signal=direct)
