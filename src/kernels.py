from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .inference import EffectiveSignals


@dataclass
class KernelSet:
    time: np.ndarray
    velocity_kernel: np.ndarray   # advective kernel g(t)
    retention_kernel: np.ndarray  # retention kernel h(t)
    direct_kernel: np.ndarray     # residual/direct signal
    reconstructed_btc: np.ndarray
    reconstruction_mse: float


def _normalize_kernel(k: np.ndarray, t: np.ndarray, normalize: bool, floor: float) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    k = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
    k = np.maximum(k, floor)

    if normalize:
        area = np.trapezoid(k, t)
        if area > 0:
            k = k / area

    return k


def _smooth_positive_kernel(k: np.ndarray, regularization_eps: float, floor: float) -> np.ndarray:
    """
    Gentle regularization only.
    Do NOT convert kernels through derivatives anymore.
    """
    k = np.asarray(k, dtype=float)
    k = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
    k = np.maximum(k, floor)

    # very light smoothing by neighbor averaging
    if len(k) >= 3:
        k_reg = k.copy()
        k_reg[1:-1] = 0.25 * k[:-2] + 0.5 * k[1:-1] + 0.25 * k[2:]
        k = k_reg

    k = np.maximum(k, regularization_eps)
    return k


def _reconstruct_btc(time: np.ndarray, g: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Reconstruct BTC from convolution:
        f(t) ≈ g(t) * h(t)
    """
    if len(time) < 2:
        return np.maximum(g * h, 0.0)

    dt = float(np.mean(np.diff(time)))
    recon = fftconvolve(g, h, mode="full")[: len(time)] * dt
    return np.maximum(recon, 0.0)


def _mse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean((y - yhat) ** 2))


def recover_kernels(
    signals: EffectiveSignals,
    regularization_eps: float = 1e-8,
    normalize_kernels: bool = True,
    min_positive_floor: float = 1e-10,
) -> KernelSet:
    """
    New logic:
    - velocity_signal is already interpreted as advective kernel g(t)
    - retention_signal is already interpreted as retention kernel h(t)
    - direct_kernel_signal is the residual/direct component

    So we DO NOT create new kernels from gradients anymore.
    We only regularize + normalize + reconstruct BTC.
    """
    t = np.asarray(signals.time, dtype=float)

    # -----------------------------
    # 1. Use inferred signals directly
    # -----------------------------
    g = _smooth_positive_kernel(
        signals.velocity_signal,
        regularization_eps=regularization_eps,
        floor=min_positive_floor,
    )

    h = _smooth_positive_kernel(
        signals.retention_signal,
        regularization_eps=regularization_eps,
        floor=min_positive_floor,
    )

    d = _smooth_positive_kernel(
        signals.direct_kernel_signal,
        regularization_eps=regularization_eps,
        floor=min_positive_floor,
    )

    # -----------------------------
    # 2. Normalize if requested
    # -----------------------------
    g = _normalize_kernel(g, t, normalize_kernels, min_positive_floor)
    h = _normalize_kernel(h, t, normalize_kernels, min_positive_floor)
    d = _normalize_kernel(d, t, normalize_kernels, min_positive_floor)

    # -----------------------------
    # 3. Reconstruct BTC from g*h
    # -----------------------------
    reconstructed_btc = _reconstruct_btc(t, g, h)

    # normalize reconstruction to BTC scale if possible
    btc_obs = np.asarray(signals.btc_smooth, dtype=float)
    if np.trapezoid(reconstructed_btc, t) > 0 and np.trapezoid(btc_obs, t) > 0:
        reconstructed_btc = reconstructed_btc * (
            np.trapezoid(btc_obs, t) / np.trapezoid(reconstructed_btc, t)
        )

    reconstruction_mse = _mse(btc_obs, reconstructed_btc)

    return KernelSet(
        time=t,
        velocity_kernel=g,
        retention_kernel=h,
        direct_kernel=d,
        reconstructed_btc=reconstructed_btc,
        reconstruction_mse=reconstruction_mse,
    )
