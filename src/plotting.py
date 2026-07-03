from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from .inference import EffectiveSignals, SnapshotMoments
from .kernels import KernelSet
from .preprocess import SmoothedSeries
from .symbolic_fit import FitResult


def _savefig(path: str | Path, dpi: int = 180) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _clean(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _area_normalize(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.maximum(_clean(y), 0.0)
    area = np.trapezoid(y, t)
    if area > 0 and np.isfinite(area):
        return y / area
    return y


def _max_normalize(y: np.ndarray) -> np.ndarray:
    y = np.maximum(_clean(y), 0.0)
    m = np.max(y) if len(y) else 0.0
    if m > 0 and np.isfinite(m):
        return y / m
    return y


def _reconstruct_btc_from_kernels(time: np.ndarray, g: np.ndarray, h: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    dt = float(np.mean(np.diff(time))) if len(time) > 1 else 1.0
    recon = fftconvolve(g, h, mode="full")[: len(time)] * dt
    recon = np.maximum(_clean(recon), 0.0)

    if target is not None:
        target = np.maximum(_clean(target), 0.0)
        rec_area = np.trapezoid(recon, time)
        target_area = np.trapezoid(target, time)
        if rec_area > 0 and target_area > 0 and np.isfinite(rec_area) and np.isfinite(target_area):
            recon = recon * (target_area / rec_area)

    return recon


def _metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    y = _clean(y)
    yhat = _clean(yhat)
    err = y - yhat
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def _metric_text(m: Dict[str, float]) -> str:
    return "\n".join([
        f"MSE = {m['MSE']:.3e}",
        f"RMSE = {m['RMSE']:.3e}",
        f"MAE = {m['MAE']:.3e}",
        f"R² = {m['R2']:.3f}" if np.isfinite(m["R2"]) else "R² = NA",
    ])


def plot_btc_preprocessing(site: str, sm: SmoothedSeries, outdir: Path, dpi: int = 180) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(sm.x_raw, sm.y_raw, marker="o", linestyle="", label="Raw BTC")
    plt.plot(sm.x_raw, sm.y_denoised, label="Denoised BTC")
    plt.plot(sm.x_uniform, sm.y_smooth, label="Smoothed BTC", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(f"{site}: BTC preprocessing")
    plt.legend()
    _savefig(outdir / f"{site}_btc_preprocessing.png", dpi=dpi)


def plot_snapshot_preprocessing(site: str, profiles: Dict[float, SmoothedSeries], outdir: Path, dpi: int = 180) -> None:
    plt.figure(figsize=(10, 6))
    for t, prof in sorted(profiles.items()):
        plt.plot(prof.x_raw, prof.y_raw, marker="o", linestyle="", alpha=0.35)
        plt.plot(prof.x_uniform, prof.y_smooth, label=f"t={t:g}", linewidth=2)
    plt.xlabel("Distance")
    plt.ylabel("Concentration")
    plt.title(f"{site}: snapshot smoothing")
    plt.legend(ncol=2, fontsize=8)
    _savefig(outdir / f"{site}_snapshot_preprocessing.png", dpi=dpi)


def plot_snapshot_moments(site: str, moments: SnapshotMoments, outdir: Path, dpi: int = 180) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(moments.snapshot_times, moments.centroids, marker="o", label="Centroid")
    plt.plot(moments.snapshot_times, moments.spreads, marker="s", label="Spread")
    plt.plot(moments.snapshot_times, moments.masses, marker="^", label="Mass")
    plt.xlabel("Snapshot time")
    plt.ylabel("Value")
    plt.title(f"{site}: snapshot moments")
    plt.legend()
    _savefig(outdir / f"{site}_snapshot_moments.png", dpi=dpi)


def plot_effective_signals(site: str, sig: EffectiveSignals, outdir: Path, dpi: int = 180) -> None:
    """
    Plot effective signals. The main supervisor-safe comparison is normalized
    here so that h(t) cannot dominate the full figure only because of scale.
    """
    reconstructed = _reconstruct_btc_from_kernels(sig.time, sig.velocity_signal, sig.retention_signal, sig.btc_smooth)

    plt.figure(figsize=(10, 6))
    plt.plot(sig.time, _max_normalize(sig.btc_smooth), label="Smoothed BTC, max-normalized", linewidth=2)
    plt.plot(sig.time, _max_normalize(reconstructed), label="Reconstructed BTC = g*h, max-normalized", linestyle="--", linewidth=2)
    plt.plot(sig.time, _max_normalize(sig.velocity_signal), label="Advective kernel g(t), max-normalized")
    plt.plot(sig.time, _max_normalize(sig.retention_signal), label="Retention kernel h(t), max-normalized")
    plt.plot(sig.time, _max_normalize(sig.direct_kernel_signal), label="Residual/direct signal, max-normalized", alpha=0.9)
    plt.xlabel("Time")
    plt.ylabel("Normalized amplitude")
    plt.title(f"{site}: normalized BTC decomposition components")
    plt.legend()
    _savefig(outdir / f"{site}_effective_signals_normalized.png", dpi=dpi)


def plot_kernels(site: str, kernels: KernelSet, outdir: Path, dpi: int = 180) -> None:
    """Plot kernel comparison using max-normalization for visual comparison."""
    reconstructed = _reconstruct_btc_from_kernels(
        kernels.time,
        kernels.velocity_kernel,
        kernels.retention_kernel,
        kernels.reconstructed_btc,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(kernels.time, _max_normalize(kernels.velocity_kernel), label="Advective kernel g(t)")
    plt.plot(kernels.time, _max_normalize(kernels.retention_kernel), label="Retention kernel h(t)")
    plt.plot(kernels.time, _max_normalize(kernels.direct_kernel), label="Residual/direct signal")
    plt.plot(kernels.time, _max_normalize(reconstructed), label="Reconstructed BTC = g*h", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Max-normalized value")
    plt.title(f"{site}: normalized kernel comparison")
    plt.legend()
    _savefig(outdir / f"{site}_kernel_comparison_normalized.png", dpi=dpi)


def plot_single_kernel(site: str, time: np.ndarray, kernel: np.ndarray, name: str, outdir: Path, dpi: int = 180) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(time, kernel, label=name, linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Kernel value")
    plt.title(f"{site}: {name}")
    plt.legend()

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )

    _savefig(outdir / f"{site}_{safe_name}.png", dpi=dpi)


def plot_btc_reconstruction(
    site: str,
    time: np.ndarray,
    btc_smooth: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    outdir: Path,
    dpi: int = 180,
) -> None:
    """Dedicated plot for BTC reconstruction from g(t) and h(t), with metrics."""
    reconstructed = _reconstruct_btc_from_kernels(time, g, h, target=btc_smooth)
    m = _metrics(btc_smooth, reconstructed)

    plt.figure(figsize=(10, 5))
    plt.plot(time, btc_smooth, label="Smoothed BTC", linewidth=2)
    plt.plot(time, reconstructed, label="Reconstructed BTC = g*h", linestyle="--", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Concentration / normalized concentration")
    plt.title(f"{site}: observed vs reconstructed BTC")
    plt.text(
        0.98,
        0.95,
        _metric_text(m),
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=9,
    )
    plt.legend(loc="best")
    _savefig(outdir / f"{site}_btc_reconstruction.png", dpi=dpi)


def plot_equation_fit(site: str, time: np.ndarray, kernel: np.ndarray, fit: FitResult, tag: str, outdir: Path, dpi: int = 180) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(time, kernel, label=f"Recovered {tag}", linewidth=2)
    plt.plot(time, fit.y_fit, label=f"Best-fit equation: {fit.family}", linestyle="--", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Kernel value")
    plt.title(f"{site}: {tag} equation fit\n{fit.equation}\nMSE = {fit.mse:.3e}")
    plt.legend()

    safe_tag = (
        tag.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )

    _savefig(outdir / f"{site}_{safe_tag}_equation_fit.png", dpi=dpi)


def plot_cross_site_family_parameters(parameter_table, outdir: Path, dpi: int = 180) -> None:
    if parameter_table.empty:
        return

    numeric_cols = [c for c in parameter_table.columns if c not in {"site", "kernel_name", "family", "structure", "formula", "equation", "fit_status"}]
    for col in numeric_cols:
        vals = parameter_table[col].astype(float)
        plt.figure(figsize=(8, 4.5))
        plt.plot(parameter_table["site"], vals, marker="o")
        plt.xlabel("Site")
        plt.ylabel(col)
        plt.title(f"Cross-site fitted parameter: {col}")
        _savefig(outdir / f"cross_site_param_{col}.png", dpi=dpi)
