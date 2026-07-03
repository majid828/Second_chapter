from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.ndimage import gaussian_filter1d
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
    """Compute plume centroid, spread, and mass for each smoothed snapshot."""
    times, centroids, spreads, masses = [], [], [], []

    for t, prof in sorted(snapshot_profiles.items()):
        x = np.asarray(prof.x_uniform, dtype=float)
        c = np.maximum(np.asarray(prof.y_smooth, dtype=float), 0.0)
        mass = np.trapezoid(c, x)

        if mass <= 0 or not np.isfinite(mass):
            centroid, spread = np.nan, np.nan
        else:
            centroid = np.trapezoid(x * c, x) / mass
            variance = np.trapezoid(((x - centroid) ** 2) * c, x) / mass
            spread = np.sqrt(max(float(variance), 0.0))

        times.append(float(t))
        centroids.append(float(centroid) if np.isfinite(centroid) else np.nan)
        spreads.append(float(spread) if np.isfinite(spread) else np.nan)
        masses.append(float(mass) if np.isfinite(mass) else 0.0)

    return SnapshotMoments(
        snapshot_times=np.asarray(times, dtype=float),
        centroids=np.asarray(centroids, dtype=float),
        spreads=np.asarray(spreads, dtype=float),
        masses=np.asarray(masses, dtype=float),
    )


def _clean_nonnegative(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(y, 0.0)


def _normalize_pdf(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return nonnegative y normalized to unit area on grid x."""
    y = _clean_nonnegative(y)
    x = np.asarray(x, dtype=float)
    if len(y) != len(x) or len(y) == 0:
        return np.zeros_like(y, dtype=float)
    area = np.trapezoid(y, x)
    if area <= 0 or not np.isfinite(area):
        return np.zeros_like(y, dtype=float)
    return y / area


def _smooth(y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    y = _clean_nonnegative(y)
    if len(y) >= 5 and sigma > 0:
        y = gaussian_filter1d(y, sigma=sigma, mode="nearest")
    return _clean_nonnegative(y)


def _gamma_shape_from_btc(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Construct a smooth advective candidate with a rise-and-decay shape.

    This replaces the old early-fraction truncation that forced g(t) to become
    exactly zero after a cutoff time. A gamma-type curve is more appropriate for
    an advective travel-time kernel because it can rise to a peak and then decay
    smoothly.
    """
    t = np.asarray(t, dtype=float)
    f = _normalize_pdf(f, t)
    if len(t) < 3 or np.trapezoid(f, t) <= 0:
        return f

    # Shift time to start at zero, but keep the original grid for normalization.
    tau = t - np.min(t)
    tau = np.maximum(tau, 0.0)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0

    area = np.trapezoid(f, t)
    mean = np.trapezoid(tau * f, t) / max(area, 1e-12)
    var = np.trapezoid(((tau - mean) ** 2) * f, t) / max(area, 1e-12)

    if not np.isfinite(mean) or mean <= 0 or not np.isfinite(var) or var <= 0:
        peak_idx = int(np.argmax(f))
        mean = max(tau[peak_idx], dt)
        var = max((0.4 * mean) ** 2, dt**2)

    theta = max(var / max(mean, 1e-12), dt)
    k = max(mean / theta, 1.05)

    gamma = (np.maximum(tau, dt * 1e-3) ** (k - 1.0)) * np.exp(-np.maximum(tau, 0.0) / theta)
    gamma = _smooth(gamma, sigma=1.0)
    return _normalize_pdf(gamma, t)


def _estimate_advective_kernel(
    t: np.ndarray,
    c: np.ndarray,
    early_fraction: float = 0.4,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Estimate an effective advective kernel g(t).

    Important correction:
    The previous code copied only the first early_fraction of the BTC and set the
    rest of g(t) to zero. That created the artificial sudden cutoff around t≈12.
    This version keeps full support and blends:
      1. a gamma-type travel-time kernel, and
      2. an early-weighted version of the BTC.
    """
    t = np.asarray(t, dtype=float)
    f = _normalize_pdf(c, t)
    if len(f) == 0 or np.trapezoid(f, t) <= 0:
        return f

    tau = (t - np.min(t)) / (np.max(t) - np.min(t) + 1e-12)
    # Early weighting encourages advective dominance near early arrivals without
    # forcing a hard cutoff.
    early_fraction = float(np.clip(early_fraction, 0.15, 0.85))
    early_weight = np.exp(-tau / max(early_fraction, 1e-6))
    early_btc = _normalize_pdf(f * early_weight, t)

    gamma_candidate = _gamma_shape_from_btc(t, f)

    g = 0.70 * gamma_candidate + 0.30 * early_btc
    g = _smooth(g, sigma=smooth_sigma)
    return _normalize_pdf(g, t)


def _build_convolution_matrix(g: np.ndarray, dt: float) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    n = len(g)
    G = np.zeros((n, n), dtype=float)
    for row in range(n):
        for col in range(row + 1):
            G[row, col] = g[row - col] * dt
    return G


def _second_difference_matrix(n: int) -> np.ndarray:
    if n < 3:
        return np.zeros((0, n), dtype=float)
    D = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def _first_bin_penalty_matrix(n: int, n_bins: int = 2) -> np.ndarray:
    """Penalize excessive impulse-like mass in the first few h(t) bins."""
    E = np.zeros((max(0, min(n_bins, n)), n), dtype=float)
    for i in range(E.shape[0]):
        E[i, i] = 1.0
    return E


def _deconvolve_retention(
    t: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    alpha: float = 1e-2,
    smooth_alpha: float = 5e-2,
    early_penalty: float = 2e-2,
    nonnegative: bool = True,
    smooth_passes: int = 4,
) -> np.ndarray:
    """
    Estimate retention/memory kernel h(t) by regularized nonnegative deconvolution.

    Solves approximately:
        min_h ||G h - f||^2
              + alpha ||h||^2
              + smooth_alpha ||D2 h||^2
              + early_penalty ||E h||^2
        subject to h >= 0.

    The D2 term reduces oscillatory deconvolution artifacts. The early penalty
    reduces the unrealistic t=0 spike seen in the previous result.
    """
    t = np.asarray(t, dtype=float)
    f = _normalize_pdf(f, t)
    g = _normalize_pdf(g, t)

    if len(t) < 3 or np.trapezoid(f, t) <= 0 or np.trapezoid(g, t) <= 0:
        return _normalize_pdf(f, t)

    dt = float(np.mean(np.diff(t)))
    n = len(f)
    G = _build_convolution_matrix(g, dt)
    D2 = _second_difference_matrix(n)
    E = _first_bin_penalty_matrix(n, n_bins=2)

    blocks = [G, np.sqrt(alpha) * np.eye(n)]
    rhs = [f, np.zeros(n)]

    if D2.size > 0 and smooth_alpha > 0:
        blocks.append(np.sqrt(smooth_alpha) * D2)
        rhs.append(np.zeros(D2.shape[0]))

    if E.size > 0 and early_penalty > 0:
        blocks.append(np.sqrt(early_penalty) * E)
        rhs.append(np.zeros(E.shape[0]))

    A_aug = np.vstack(blocks)
    b_aug = np.concatenate(rhs)

    if nonnegative:
        result = lsq_linear(A_aug, b_aug, bounds=(0.0, np.inf), max_iter=10000, lsmr_tol="auto")
        h = result.x
    else:
        h, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)

    h = _clean_nonnegative(h)
    for _ in range(max(0, int(smooth_passes))):
        h = _smooth(h, sigma=1.0)

    return _normalize_pdf(h, t)


def _reconstruct_btc(t: np.ndarray, g: np.ndarray, h: np.ndarray) -> np.ndarray:
    if len(t) < 2:
        return _clean_nonnegative(g * h)
    dt = float(np.mean(np.diff(t)))
    recon = np.convolve(g, h, mode="full")[: len(t)] * dt
    return _clean_nonnegative(recon)


def _apply_snapshot_guidance(g: np.ndarray, t: np.ndarray, snapshot_moments: SnapshotMoments) -> np.ndarray:
    """
    Apply only mild snapshot guidance.

    The old code used exp(-tau / velocity_scale), mixing normalized time with a
    dimensional velocity estimate. That can distort g(t). This version uses the
    snapshot-derived centroid trend only to apply a weak, dimensionless timing
    adjustment.
    """
    st = np.asarray(snapshot_moments.snapshot_times, dtype=float)
    cent = np.asarray(snapshot_moments.centroids, dtype=float)
    valid = np.isfinite(st) & np.isfinite(cent)

    if valid.sum() < 3:
        return _normalize_pdf(g, t)

    stv = st[valid]
    cv = cent[valid]
    if np.any(np.diff(stv) <= 0):
        order = np.argsort(stv)
        stv = stv[order]
        cv = cv[order]

    vel = np.gradient(cv, stv)
    vel = vel[np.isfinite(vel)]
    if len(vel) == 0:
        return _normalize_pdf(g, t)

    positive_fraction = np.mean(vel > 0)
    if positive_fraction < 0.5:
        return _normalize_pdf(g, t)

    tau = (t - np.min(t)) / (np.max(t) - np.min(t) + 1e-12)
    # Weakly favor earlier advective contribution, but preserve full support.
    weight = 0.85 + 0.15 * np.exp(-tau / 0.5)
    return _normalize_pdf(g * weight, t)


def build_effective_signals(btc_smoothed: SmoothedSeries, snapshot_moments: SnapshotMoments) -> EffectiveSignals:
    """
    Build effective BTC, advective, retention, and residual/direct signals.

    Key corrections compared with the previous version:
    - g(t) is no longer forced to zero after an early cutoff.
    - h(t) is recovered with stronger smoothness regularization.
    - direct/residual signal is computed from BTC minus reconstructed BTC, not BTC minus g(t).
    - direct/residual is not forced to unit area; it remains an amplitude residual.
    """
    t = np.asarray(btc_smoothed.x_uniform, dtype=float)
    f_raw = _clean_nonnegative(np.asarray(btc_smoothed.y_smooth, dtype=float))
    f_norm = _normalize_pdf(f_raw, t)

    g = _estimate_advective_kernel(t, f_norm, early_fraction=0.4, smooth_sigma=1.0)
    g = _apply_snapshot_guidance(g, t, snapshot_moments)

    h = _deconvolve_retention(
        t,
        f_norm,
        g,
        alpha=1e-2,
        smooth_alpha=5e-2,
        early_penalty=2e-2,
        nonnegative=True,
        smooth_passes=4,
    )

    reconstructed = _reconstruct_btc(t, g, h)
    if np.trapezoid(reconstructed, t) > 0 and np.trapezoid(f_norm, t) > 0:
        reconstructed = reconstructed * (np.trapezoid(f_norm, t) / np.trapezoid(reconstructed, t))

    direct = _clean_nonnegative(f_norm - reconstructed)
    if len(direct) >= 5:
        direct = _smooth(direct, sigma=1.0)

    return EffectiveSignals(
        time=t,
        btc_smooth=f_norm,
        velocity_signal=g,
        retention_signal=h,
        direct_kernel_signal=direct,
    )
