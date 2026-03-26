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


def _savefig(path: str | Path, dpi: int = 160) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _reconstruct_btc_from_kernels(time: np.ndarray, g: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Reconstruct BTC from the convolution of the advective kernel g(t)
    and the retention kernel h(t).
    """
    dt = float(np.mean(np.diff(time))) if len(time) > 1 else 1.0
    recon = fftconvolve(g, h, mode="full")[: len(time)] * dt
    return np.maximum(recon, 0.0)


def plot_btc_preprocessing(site: str, sm: SmoothedSeries, outdir: Path, dpi: int = 160) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(sm.x_raw, sm.y_raw, marker="o", linestyle="", label="Raw BTC")
    plt.plot(sm.x_raw, sm.y_denoised, label="Denoised BTC")
    plt.plot(sm.x_uniform, sm.y_smooth, label="Smoothed BTC")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(f"{site}: BTC preprocessing")
    plt.legend()
    _savefig(outdir / f"{site}_btc_preprocessing.png", dpi=dpi)


def plot_snapshot_preprocessing(site: str, profiles: Dict[float, SmoothedSeries], outdir: Path, dpi: int = 160) -> None:
    plt.figure(figsize=(10, 6))
    for t, prof in sorted(profiles.items()):
        plt.plot(prof.x_raw, prof.y_raw, marker="o", linestyle="", alpha=0.45)
        plt.plot(prof.x_uniform, prof.y_smooth, label=f"t={t:g}")
    plt.xlabel("Distance")
    plt.ylabel("Concentration")
    plt.title(f"{site}: snapshot smoothing")
    plt.legend(ncol=2, fontsize=8)
    _savefig(outdir / f"{site}_snapshot_preprocessing.png", dpi=dpi)


def plot_snapshot_moments(site: str, moments: SnapshotMoments, outdir: Path, dpi: int = 160) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(moments.snapshot_times, moments.centroids, marker="o", label="Centroid")
    plt.plot(moments.snapshot_times, moments.spreads, marker="s", label="Spread")
    plt.plot(moments.snapshot_times, moments.masses, marker="^", label="Mass")
    plt.xlabel("Snapshot time")
    plt.ylabel("Value")
    plt.title(f"{site}: snapshot moments")
    plt.legend()
    _savefig(outdir / f"{site}_snapshot_moments.png", dpi=dpi)


def plot_effective_signals(site: str, sig: EffectiveSignals, outdir: Path, dpi: int = 160) -> None:
    """
    Plot the new decomposed signals:
      - smoothed BTC
      - advective/velocity kernel g(t)
      - retention kernel h(t)
      - residual/direct signal
      - reconstructed BTC from g*h
    """
    reconstructed = _reconstruct_btc_from_kernels(
        sig.time,
        sig.velocity_signal,
        sig.retention_signal,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(sig.time, sig.btc_smooth, label="Smoothed BTC", linewidth=2)
    plt.plot(sig.time, reconstructed, label="Reconstructed BTC = g*h", linestyle="--", linewidth=2)
    plt.plot(sig.time, sig.velocity_signal, label="Advective kernel g(t)")
    plt.plot(sig.time, sig.retention_signal, label="Retention kernel h(t)")
    plt.plot(sig.time, sig.direct_kernel_signal, label="Residual / direct signal", alpha=0.9)
    plt.xlabel("Time")
    plt.ylabel("Signal amplitude")
    plt.title(f"{site}: BTC decomposition into advective and retention components")
    plt.legend()
    _savefig(outdir / f"{site}_effective_signals.png", dpi=dpi)


def plot_kernels(site: str, kernels: KernelSet, outdir: Path, dpi: int = 160) -> None:
    """
    Plot recovered kernels using new interpretation:
      - velocity_kernel = advective kernel g(t)
      - retention_kernel = retention kernel h(t)
      - direct_kernel = optional residual/direct kernel
    """
    reconstructed = _reconstruct_btc_from_kernels(
        kernels.time,
        kernels.velocity_kernel,
        kernels.retention_kernel,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(kernels.time, kernels.velocity_kernel, label="Advective kernel g(t)")
    plt.plot(kernels.time, kernels.retention_kernel, label="Retention kernel h(t)")
    plt.plot(kernels.time, kernels.direct_kernel, label="Residual / direct kernel")
    plt.plot(kernels.time, reconstructed, label="Reconstructed BTC = g*h", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Kernel value")
    plt.title(f"{site}: kernel comparison")
    plt.legend()
    _savefig(outdir / f"{site}_kernel_comparison.png", dpi=dpi)


def plot_single_kernel(site: str, time: np.ndarray, kernel: np.ndarray, name: str, outdir: Path, dpi: int = 160) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(time, kernel, label=name)
    plt.xlabel("Time")
    plt.ylabel("Kernel value")
    plt.title(f"{site}: {name}")
    plt.legend()
    _savefig(outdir / f"{site}_{name.lower().replace(' ', '_')}.png", dpi=dpi)


def plot_btc_reconstruction(
    site: str,
    time: np.ndarray,
    btc_smooth: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    outdir: Path,
    dpi: int = 160,
) -> None:
    """
    Dedicated plot for BTC reconstruction from g(t) and h(t).
    """
    reconstructed = _reconstruct_btc_from_kernels(time, g, h)

    plt.figure(figsize=(10, 5))
    plt.plot(time, btc_smooth, label="Smoothed BTC", linewidth=2)
    plt.plot(time, reconstructed, label="Reconstructed BTC = g*h", linestyle="--", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(f"{site}: observed vs reconstructed BTC")
    plt.legend()
    _savefig(outdir / f"{site}_btc_reconstruction.png", dpi=dpi)


def plot_equation_fit(site: str, time: np.ndarray, kernel: np.ndarray, fit: FitResult, tag: str, outdir: Path, dpi: int = 160) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(time, kernel, label=f"Recovered {tag}")
    plt.plot(time, fit.y_fit, label=f"Best-fit equation: {fit.family}")
    plt.xlabel("Time")
    plt.ylabel("Kernel value")
    plt.title(f"{site}: {tag} equation fit\n{fit.equation}")
    plt.legend()
    _savefig(outdir / f"{site}_{tag.lower().replace(' ', '_')}_equation_fit.png", dpi=dpi)


def plot_cross_site_family_parameters(parameter_table, outdir: Path, dpi: int = 160) -> None:
    if parameter_table.empty:
        return

    numeric_cols = [c for c in parameter_table.columns if c not in {"site", "kernel_name", "family", "equation"}]
    for col in numeric_cols:
        vals = parameter_table[col].astype(float)
        plt.figure(figsize=(8, 4.5))
        plt.plot(parameter_table["site"], vals, marker="o")
        plt.xlabel("Site")
        plt.ylabel(col)
        plt.title(f"Cross-site fitted parameter: {col}")
        _savefig(outdir / f"cross_site_param_{col}.png", dpi=dpi)
