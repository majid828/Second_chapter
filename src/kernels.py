from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inference import EffectiveSignals


@dataclass
class KernelSet:
    time: np.ndarray
    velocity_kernel: np.ndarray
    retention_kernel: np.ndarray
    direct_kernel: np.ndarray


def _normalize_kernel(k: np.ndarray, t: np.ndarray, normalize: bool, floor: float) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    k = np.maximum(k, floor)
    if normalize:
        area = np.trapezoid(k, t)
        if area > 0:
            k = k / area
    return k


def recover_kernels(
    signals: EffectiveSignals,
    regularization_eps: float = 1e-8,
    normalize_kernels: bool = True,
    min_positive_floor: float = 1e-10,
) -> KernelSet:
    t = signals.time
    dt = np.gradient(t)

    # Velocity-derived kernel:
    # emphasize temporal variability / persistence in the velocity-like signal
    v = signals.velocity_signal
    v_dev = v - np.mean(v)
    velocity_kernel = np.abs(np.gradient(v_dev, t)) + regularization_eps

    # Retention-derived kernel:
    # retention already behaves like a memory intensity proxy; smooth derivative is used
    r = signals.retention_signal
    retention_kernel = np.maximum(r + 0.5 * np.abs(np.gradient(r, t)), regularization_eps)

    # Direct BTC-derived kernel:
    d = signals.direct_kernel_signal
    direct_kernel = np.maximum(d + 0.25 * np.abs(np.gradient(d, t)), regularization_eps)

    velocity_kernel = _normalize_kernel(velocity_kernel, t, normalize_kernels, min_positive_floor)
    retention_kernel = _normalize_kernel(retention_kernel, t, normalize_kernels, min_positive_floor)
    direct_kernel = _normalize_kernel(direct_kernel, t, normalize_kernels, min_positive_floor)

    return KernelSet(
        time=t,
        velocity_kernel=velocity_kernel,
        retention_kernel=retention_kernel,
        direct_kernel=direct_kernel,
    )
