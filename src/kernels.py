from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

from .inference import EffectiveSignals


@dataclass
class KernelSet:
    time: np.ndarray
    velocity_kernel: np.ndarray   # advective kernel g(t)
    retention_kernel: np.ndarray  # retention/memory kernel h(t)
    direct_kernel: np.ndarray     # residual/direct signal, not necessarily normalized
    reconstructed_btc: np.ndarray
    reconstruction_mse: float


def _clean_nonnegative(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    k = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(k, 0.0)


def _normalize_area(k: np.ndarray, t: np.ndarray) -> np.ndarray:
    k = _clean_nonnegative(k)
    area = np.trapezoid(k, t)
    if area > 0 and np.isfinite(area):
        return k / area
    return np.zeros_like(k, dtype=float)


def _smooth_positive_kernel(k: np.ndarray, sigma: float = 1.0, min_positive_floor: float = 0.0) -> np.ndarray:
    """
    Smooth a nonnegative kernel without creating artificial positive tails.

    Important correction:
    The previous code forced every point to be at least regularization_eps. That
    can create an artificial nonzero baseline and distort normalization. This
    version uses zero as the natural floor unless explicitly changed.
    """
    k = _clean_nonnegative(k)
    if len(k) >= 5 and sigma > 0:
        k = gaussian_filter1d(k, sigma=sigma, mode="nearest")
    if min_positive_floor > 0:
        k = np.maximum(k, min_positive_floor)
    return _clean_nonnegative(k)


def _reconstruct_btc(time: np.ndarray, g: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Reconstruct BTC from convolution f(t) ≈ g(t) * h(t)."""
    if len(time) < 2:
        return _clean_nonnegative(g * h)

    dt = float(np.mean(np.diff(time)))
    recon = fftconvolve(g, h, mode="full")[: len(time)] * dt
    return _clean_nonnegative(recon)


def _mse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean((y - yhat) ** 2))


def recover_kernels(
    signals: EffectiveSignals,
    regularization_eps: float = 0.0,
    normalize_kernels: bool = True,
    min_positive_floor: float = 0.0,
) -> KernelSet:
    """
    Recover final kernel set from effective signals.

    Corrections:
    - g(t) and h(t) are smoothed and optionally area-normalized.
    - direct/residual signal is NOT area-normalized because it is an amplitude
      residual, not a probability kernel.
    - reconstruction is scaled to the observed normalized BTC area.
    """
    t = np.asarray(signals.time, dtype=float)
    btc_obs = _clean_nonnegative(np.asarray(signals.btc_smooth, dtype=float))

    g = _smooth_positive_kernel(
        signals.velocity_signal,
        sigma=1.0,
        min_positive_floor=max(float(min_positive_floor), float(regularization_eps), 0.0),
    )
    h = _smooth_positive_kernel(
        signals.retention_signal,
        sigma=1.0,
        min_positive_floor=max(float(min_positive_floor), float(regularization_eps), 0.0),
    )
    d = _smooth_positive_kernel(
        signals.direct_kernel_signal,
        sigma=1.0,
        min_positive_floor=0.0,
    )

    if normalize_kernels:
        g = _normalize_area(g, t)
        h = _normalize_area(h, t)
        # Do not normalize d. It is a residual amplitude signal.

    reconstructed_btc = _reconstruct_btc(t, g, h)

    obs_area = np.trapezoid(btc_obs, t)
    rec_area = np.trapezoid(reconstructed_btc, t)
    if obs_area > 0 and rec_area > 0 and np.isfinite(obs_area) and np.isfinite(rec_area):
        reconstructed_btc = reconstructed_btc * (obs_area / rec_area)

    reconstruction_mse = _mse(btc_obs, reconstructed_btc)

    return KernelSet(
        time=t,
        velocity_kernel=g,
        retention_kernel=h,
        direct_kernel=d,
        reconstructed_btc=reconstructed_btc,
        reconstruction_mse=reconstruction_mse,
    )
