from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .inference import build_effective_signals, compute_snapshot_moments
from .io_utils import ensure_dir, load_btc_csv, load_snapshot_csv, save_json, split_by_site
from .kernels import recover_kernels
from .plotting import (
    plot_btc_preprocessing,
    plot_btc_reconstruction,
    plot_cross_site_family_parameters,
    plot_effective_signals,
    plot_equation_fit,
    plot_kernels,
    plot_single_kernel,
    plot_snapshot_moments,
    plot_snapshot_preprocessing,
)
from .preprocess import smooth_btc, smooth_snapshot_profiles
from .symbolic_fit import fit_all_signal_equations, fit_shared_family_across_sites


def _safe_scalar_mean_velocity(time: np.ndarray, g: np.ndarray) -> float:
    """Compute an effective velocity from g(t), assuming normalized travel distance L=1."""
    time = np.asarray(time, dtype=float)
    g = np.asarray(g, dtype=float)
    area = np.trapezoid(g, time)
    if area <= 0:
        return 0.0
    g_norm = g / area
    mean_t = np.trapezoid(time * g_norm, time)
    if mean_t <= 0:
        return 0.0
    return float(1.0 / mean_t)


def run_pipeline(config: Dict) -> pd.DataFrame:
    """
    Full pipeline with TRUE multi-site shared-family discovery.

    For the retention kernel, this now calculates:

        For each candidate family m:
            For each site i:
                fit K_m(t; theta_i) to recovered h_i(t)
            L_m = sum_i sum_j [h_i(t_j) - K_m(t_j; theta_i)]^2
            J_m = L_m + alpha * Complexity(m)

        best family = argmin_m J_m
    """
    btc_df = load_btc_csv(config["btc_csv"])
    snapshot_df = load_snapshot_csv(config["snapshot_csv"])
    outdir = ensure_dir(config["output_dir"])

    site_data = split_by_site(btc_df, snapshot_df)
    requested_sites = config.get("sites")
    if requested_sites:
        site_data = {k: v for k, v in site_data.items() if k in requested_sites}

    preprocess_cfg = config["preprocess"]
    kernel_cfg = config["kernel"]
    dpi = config.get("plots", {}).get("dpi", 160)

    rows: List[Dict] = []
    recovered_site_kernels: Dict[str, Dict[str, np.ndarray]] = {}

    for site, (btc_site, snapshot_site) in site_data.items():
        site_out = ensure_dir(Path(outdir) / site)

        btc_smoothed = smooth_btc(
            btc_site,
            uniform_points=preprocess_cfg["btc_uniform_points"],
            sg_window=preprocess_cfg["sg_window"],
            sg_polyorder=preprocess_cfg["sg_polyorder"],
            gaussian_sigma=preprocess_cfg["gaussian_sigma"],
            clip_negative_to_zero=preprocess_cfg["clip_negative_to_zero"],
        )

        snapshot_smoothed = smooth_snapshot_profiles(
            snapshot_site,
            uniform_points=preprocess_cfg["snapshot_uniform_points"],
            sg_window=max(7, preprocess_cfg["sg_window"] - 4),
            sg_polyorder=preprocess_cfg["sg_polyorder"],
            gaussian_sigma=max(1.0, preprocess_cfg["gaussian_sigma"] - 0.5),
            clip_negative_to_zero=preprocess_cfg["clip_negative_to_zero"],
        )

        moments = compute_snapshot_moments(snapshot_smoothed)
        signals = build_effective_signals(btc_smoothed, moments)

        kernels = recover_kernels(
            signals,
            regularization_eps=kernel_cfg["regularization_eps"],
            normalize_kernels=kernel_cfg["normalize_kernels"],
            min_positive_floor=kernel_cfg["min_positive_floor"],
        )

        recovered_site_kernels[site] = {
            "time": np.asarray(kernels.time, dtype=float),
            "velocity_kernel": np.asarray(kernels.velocity_kernel, dtype=float),
            "retention_kernel": np.asarray(kernels.retention_kernel, dtype=float),
            "direct_kernel": np.asarray(kernels.direct_kernel, dtype=float),
        }

        # Individual site-level fits. These are diagnostics only.
        fit_results = fit_all_signal_equations(
            kernels.time,
            {
                "velocity_kernel": kernels.velocity_kernel,
                "retention_kernel": kernels.retention_kernel,
                "direct_kernel": kernels.direct_kernel,
            },
        )

        plot_btc_preprocessing(site, btc_smoothed, site_out, dpi=dpi)
        plot_snapshot_preprocessing(site, snapshot_smoothed, site_out, dpi=dpi)
        plot_snapshot_moments(site, moments, site_out, dpi=dpi)
        plot_effective_signals(site, signals, site_out, dpi=dpi)

        plot_single_kernel(site, kernels.time, kernels.velocity_kernel, "Advective Kernel g(t)", site_out, dpi=dpi)
        plot_single_kernel(site, kernels.time, kernels.retention_kernel, "Retention Kernel h(t)", site_out, dpi=dpi)
        plot_single_kernel(site, kernels.time, kernels.direct_kernel, "Residual / Direct Signal", site_out, dpi=dpi)

        plot_kernels(site, kernels, site_out, dpi=dpi)
        plot_btc_reconstruction(site, kernels.time, signals.btc_smooth, kernels.velocity_kernel, kernels.retention_kernel, site_out, dpi=dpi)

        pretty_name_map = {
            "velocity_kernel": "Advective Kernel g(t)",
            "retention_kernel": "Retention Kernel h(t)",
            "direct_kernel": "Residual / Direct Signal",
        }

        for key, fit in fit_results.items():
            label = pretty_name_map.get(key, key.replace("_", " ").title())
            kernel_y = getattr(kernels, key)
            plot_equation_fit(site, kernels.time, kernel_y, fit, label, site_out, dpi=dpi)

            row = {"site": site, "kernel_name": key, "family": fit.family, "equation": fit.equation, "mse": fit.mse, "score": fit.score}
            row.update(fit.params)
            rows.append(row)

        effective_velocity = _safe_scalar_mean_velocity(kernels.time, kernels.velocity_kernel)
        site_summary = {
            "site": site,
            "effective_velocity_from_g": effective_velocity,
            "reconstruction_mse": kernels.reconstruction_mse,
            "best_individual_fits": {
                k: {"family": v.family, "params": v.params, "equation": v.equation, "mse": v.mse, "score": v.score, "label": getattr(v, "label", "K")}
                for k, v in fit_results.items()
            },
        }
        save_json(site_summary, site_out / f"{site}_summary.json")

    # Summary of independent single-site fits.
    result_df = pd.DataFrame(rows)
    result_df.to_csv(Path(outdir) / "cross_site_kernel_fit_summary_individual.csv", index=False)
    plot_cross_site_family_parameters(result_df, Path(outdir), dpi=dpi)

    # TRUE shared-family discovery.
    shared_cfg = config.get("shared_family", {})
    shared_kernel_name = shared_cfg.get("kernel_name", "retention_kernel")
    family_names = shared_cfg.get("families", ["exponential", "stretched_exponential", "power_law", "tempered_power_law", "gamma_type"])
    complexity_weight = float(shared_cfg.get("complexity_weight", 0.01))

    shared_result = fit_shared_family_across_sites(
        recovered_site_kernels,
        kernel_name=shared_kernel_name,
        family_names=family_names,
        complexity_weight=complexity_weight,
    )

    shared_result.family_summary.to_csv(Path(outdir) / f"shared_family_global_summary_{shared_kernel_name}.csv", index=False)
    shared_result.best_site_parameters.to_csv(Path(outdir) / f"shared_family_best_site_parameters_{shared_kernel_name}.csv", index=False)
    shared_result.all_site_parameters.to_csv(Path(outdir) / f"shared_family_all_site_parameters_{shared_kernel_name}.csv", index=False)

    shared_json = {
        "kernel_name": shared_kernel_name,
        "best_shared_family": shared_result.best_family,
        "best_objective": shared_result.best_objective,
        "complexity_weight": complexity_weight,
        "formula": shared_result.best_formula,
        "family_summary": shared_result.family_summary.to_dict(orient="records"),
        "best_site_parameters": shared_result.best_site_parameters.to_dict(orient="records"),
    }
    save_json(shared_json, Path(outdir) / f"shared_family_result_{shared_kernel_name}.json")

    print("\nTRUE SHARED-FAMILY DISCOVERY RESULT")
    print("-----------------------------------")
    print(f"Kernel tested: {shared_kernel_name}")
    print(f"Best shared family: {shared_result.best_family}")
    print(f"Best formula: {shared_result.best_formula}")
    print(f"Best objective: {shared_result.best_objective:.6g}")
    print(f"Saved results in: {outdir}")

    return result_df
