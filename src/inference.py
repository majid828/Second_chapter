from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

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
    times = []
    centroids = []
    spreads = []
    masses = []
    for t in sorted(snapshot_profiles):
        prof = snapshot_profiles[t]
        x = prof.x_uniform
        c = np.maximum(prof.y_smooth, 0.0)
        mass = np.trapezoid(c, x)
        if mass <= 0:
            centroid = np.nan
            spread = np.nan
        else:
            centroid = np.trapezoid(x * c, x) / mass
            variance = np.trapezoid(((x - centroid) ** 2) * c, x) / mass
            spread = np.sqrt(max(variance, 0.0))
        times.append(t)
        centroids.append(centroid)
        spreads.append(spread)
        masses.append(mass)
    return SnapshotMoments(
        snapshot_times=np.asarray(times, dtype=float),
        centroids=np.asarray(centroids, dtype=float),
        spreads=np.asarray(spreads, dtype=float),
        masses=np.asarray(masses, dtype=float),
    )


def build_effective_signals(
    btc_smoothed: SmoothedSeries,
    snapshot_moments: SnapshotMoments,
) -> EffectiveSignals:
    t = btc_smoothed.x_uniform
    c = np.maximum(btc_smoothed.y_smooth, 0.0)

    # Velocity-like signal:
    # Use the time-derivative of centroid progression from plume snapshots as a mean-transport proxy,
    # then interpolate it onto the BTC time grid.
    st = snapshot_moments.snapshot_times
    cent = snapshot_moments.centroids
    valid = np.isfinite(st) & np.isfinite(cent)
    if valid.sum() >= 3:
        v_snap = np.gradient(cent[valid], st[valid])
        velocity_signal = np.interp(t, st[valid], v_snap, left=v_snap[0], right=v_snap[-1])
    else:
        velocity_signal = np.gradient(c, t)

    # Retention-like signal:
    # Use late-time persistence / tailing tendency relative to peak and decay slope.
    c_norm = c / (np.max(c) + 1e-12)
    decay_rate = -np.gradient(np.log(c_norm + 1e-12), t)
    retention_signal = np.maximum(decay_rate, 0.0)

    # Direct kernel signal from BTC curvature/tailing information.
    first = np.gradient(c, t)
    second = np.gradient(first, t)
    direct_kernel_signal = np.maximum(-second, 0.0)

    velocity_signal = np.nan_to_num(velocity_signal, nan=0.0, posinf=0.0, neginf=0.0)
    retention_signal = np.nan_to_num(retention_signal, nan=0.0, posinf=0.0, neginf=0.0)
    direct_kernel_signal = np.nan_to_num(direct_kernel_signal, nan=0.0, posinf=0.0, neginf=0.0)

    return EffectiveSignals(
        time=t,
        btc_smooth=c,
        velocity_signal=velocity_signal,
        retention_signal=retention_signal,
        direct_kernel_signal=direct_kernel_signal,
    )
