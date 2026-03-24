from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor


@dataclass
class SmoothedSeries:
    x_raw: np.ndarray
    y_raw: np.ndarray
    x_uniform: np.ndarray
    y_denoised: np.ndarray
    y_smooth: np.ndarray


def _safe_positive(y: np.ndarray, clip_negative_to_zero: bool) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if clip_negative_to_zero:
        y = np.maximum(y, 0.0)
    return y


def _rf_denoise_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    ML-based denoising using a RandomForest regressor trained on time -> concentration.
    This is robust for noisy discrete curves and works without requiring deep learning.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    if len(y) < 8:
        return y.copy()
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=max(1, len(y) // 25),
    )
    model.fit(x, y)
    return model.predict(x)


def smooth_btc(
    btc_df: pd.DataFrame,
    uniform_points: int = 400,
    sg_window: int = 21,
    sg_polyorder: int = 3,
    gaussian_sigma: float = 2.0,
    clip_negative_to_zero: bool = True,
) -> SmoothedSeries:
    x_raw = btc_df["time"].to_numpy(dtype=float)
    y_raw = btc_df["concentration"].to_numpy(dtype=float)

    x_uniform = np.linspace(x_raw.min(), x_raw.max(), uniform_points)
    y_denoised_raw = _rf_denoise_1d(x_raw, y_raw)

    spline = UnivariateSpline(x_raw, y_denoised_raw, s=max(1e-8, 0.01 * len(x_raw) * np.var(y_denoised_raw)))
    y_interp = spline(x_uniform)

    window = min(sg_window if sg_window % 2 == 1 else sg_window + 1, len(x_uniform) - (1 - len(x_uniform) % 2))
    window = max(window, sg_polyorder + 3 + ((sg_polyorder + 3) % 2 == 0))
    if window >= len(x_uniform):
        window = len(x_uniform) - 1 if len(x_uniform) % 2 == 0 else len(x_uniform)
    if window < 5:
        y_sg = y_interp.copy()
    else:
        y_sg = savgol_filter(y_interp, window_length=window, polyorder=min(sg_polyorder, window - 1))

    y_smooth = gaussian_filter1d(y_sg, sigma=gaussian_sigma)
    y_smooth = _safe_positive(y_smooth, clip_negative_to_zero)
    y_denoised_raw = _safe_positive(y_denoised_raw, clip_negative_to_zero)

    return SmoothedSeries(
        x_raw=x_raw,
        y_raw=y_raw,
        x_uniform=x_uniform,
        y_denoised=y_denoised_raw,
        y_smooth=y_smooth,
    )


def smooth_snapshot_profiles(
    snapshot_df: pd.DataFrame,
    uniform_points: int = 250,
    sg_window: int = 15,
    sg_polyorder: int = 3,
    gaussian_sigma: float = 1.5,
    clip_negative_to_zero: bool = True,
) -> Dict[float, SmoothedSeries]:
    out: Dict[float, SmoothedSeries] = {}
    grouped = snapshot_df.groupby("time")

    for t, g in grouped:
        x_raw = g["distance"].to_numpy(dtype=float)
        y_raw = g["concentration"].to_numpy(dtype=float)
        order = np.argsort(x_raw)
        x_raw = x_raw[order]
        y_raw = y_raw[order]

        x_uniform = np.linspace(x_raw.min(), x_raw.max(), uniform_points)
        y_denoised_raw = _rf_denoise_1d(x_raw, y_raw)

        spline = UnivariateSpline(x_raw, y_denoised_raw, s=max(1e-8, 0.01 * len(x_raw) * np.var(y_denoised_raw)))
        y_interp = spline(x_uniform)

        window = min(sg_window if sg_window % 2 == 1 else sg_window + 1, len(x_uniform) - (1 - len(x_uniform) % 2))
        if window < 5:
            y_sg = y_interp.copy()
        else:
            y_sg = savgol_filter(y_interp, window_length=window, polyorder=min(sg_polyorder, window - 1))

        y_smooth = gaussian_filter1d(y_sg, sigma=gaussian_sigma)
        y_smooth = _safe_positive(y_smooth, clip_negative_to_zero)
        y_denoised_raw = _safe_positive(y_denoised_raw, clip_negative_to_zero)

        out[float(t)] = SmoothedSeries(
            x_raw=x_raw,
            y_raw=y_raw,
            x_uniform=x_uniform,
            y_denoised=y_denoised_raw,
            y_smooth=y_smooth,
        )
    return out
