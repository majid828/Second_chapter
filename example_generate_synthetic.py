from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42

def normalize_area(y, x):
    y = np.nan_to_num(np.asarray(y, float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0.0)
    area = np.trapezoid(y, x)
    return y / area if area > 0 else y

def gaussian_advective(t, mu, sigma):
    return normalize_area(np.exp(-((t - mu) ** 2) / (2 * sigma ** 2)), t)

def exp_k(t, a, b):
    return a * np.exp(-b * t)

def stretched_k(t, a, b, beta):
    return a * np.exp(-((b * t) ** beta))

def power_k(t, a, lam, c):
    return a * (t + c) ** (-lam)

def tempered_k(t, a, lam, b, c):
    return a * (t + c) ** (-lam) * np.exp(-b * t)

def gamma_k(t, a, k, theta):
    tp = np.maximum(t, 1e-12)
    return a * (tp ** (k - 1)) * np.exp(-tp / theta)

def two_exp_k(t, a1, b1, a2, b2):
    return a1 * np.exp(-b1 * t) + a2 * np.exp(-b2 * t)

def exp_plus_power_k(t, a1, b1, a2, lam, c):
    return a1 * np.exp(-b1 * t) + a2 * (t + c) ** (-lam)

def tempered_plus_exp_k(t, a1, lam, b1, c, a2, b2):
    return a1 * (t + c) ** (-lam) * np.exp(-b1 * t) + a2 * np.exp(-b2 * t)

def softplus_exp_exp_k(t, a, c, b, q):
    return a * np.log(c + np.exp(-b * np.exp(q * t)))

def lognormal_like_k(t, a, mu, sigma, c):
    tp = np.maximum(t + c, 1e-12)
    return a * np.exp(-((np.log(tp) - mu) ** 2) / (2 * sigma ** 2)) / tp

def retention_kernel(t, site):
    if site == "site_A":
        return normalize_area(exp_k(t, 1.0, 1.20), t), "exponential_short_memory"
    if site == "site_B":
        return normalize_area(stretched_k(t, 1.0, 0.55, 0.65), t), "stretched_exponential"
    if site == "site_C":
        return normalize_area(gamma_k(t, 1.0, 2.2, 1.5), t), "gamma_type"
    if site == "site_D":
        return normalize_area(power_k(t, 1.0, 0.85, 0.25), t), "power_law_long_memory"
    if site == "site_E":
        return normalize_area(tempered_k(t, 1.0, 0.65, 0.12, 0.20), t), "tempered_power_law"
    if site == "site_F":
        return normalize_area(two_exp_k(t, 0.75, 0.35, 0.25, 2.40), t), "two_exponential"
    if site == "site_G":
        return normalize_area(exp_plus_power_k(t, 0.55, 0.85, 0.45, 1.10, 0.30), t), "exp_plus_power"
    if site == "site_H":
        return normalize_area(tempered_plus_exp_k(t, 0.70, 0.75, 0.18, 0.25, 0.30, 1.80), t), "tempered_plus_exp"
    if site == "site_I":
        return normalize_area(softplus_exp_exp_k(t, 1.0, 1.08, 0.18, 0.95), t), "softplus_exp_exp"
    if site == "site_J":
        return normalize_area(lognormal_like_k(t, 1.0, 1.05, 0.55, 0.15), t), "lognormal_like"
    raise ValueError(site)

def convolve_btc(t, g, h):
    dt = float(np.mean(np.diff(t)))
    return normalize_area(np.convolve(g, h, mode="full")[:len(t)] * dt, t)

def make_snapshots(site, snapshot_times, x_grid, velocity, dispersion, retention_strength, rng):
    rows = []
    for tm in snapshot_times:
        center = velocity * tm
        spread = max(np.sqrt(2 * dispersion * tm + retention_strength * tm), 0.05)
        c = np.exp(-((x_grid - center) ** 2) / (2 * spread ** 2))
        c = c / (np.trapezoid(c, x_grid) + 1e-12)
        c = np.maximum(c * np.exp(-0.03 * tm) * (1 + 0.04 * rng.normal()), 0.0)
        for x, val in zip(x_grid, c):
            rows.append({"site": site, "time": tm, "x": x, "concentration": val})
    return rows

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    Path("examples").mkdir(exist_ok=True)

    sites = [f"site_{ch}" for ch in "ABCDEFGHIJ"]
    t = np.linspace(0.05, 20.0, 450)
    x_grid = np.linspace(0.0, 8.0, 220)

    btc_rows, truth_rows, snapshot_rows = [], [], []

    for idx, site in enumerate(sites):
        g = gaussian_advective(t, mu=2.0 + 0.22 * idx, sigma=0.35 + 0.035 * idx)
        h, true_type = retention_kernel(t, site)
        f = convolve_btc(t, g, h)

        noise_level = 0.010 + 0.002 * (idx % 4)
        f = np.maximum(f + rng.normal(0, noise_level * np.max(f), size=len(f)), 0.0)

        for tt, cc in zip(t, f):
            btc_rows.append({"site": site, "time": tt, "concentration": cc})

        truth_rows.append({
            "site": site,
            "true_retention_type": true_type,
            "advective_mu": 2.0 + 0.22 * idx,
            "advective_sigma": 0.35 + 0.035 * idx,
            "noise_level": noise_level
        })

        snapshot_rows.extend(make_snapshots(
            site, [3.0, 6.0, 10.0, 15.0], x_grid,
            velocity=0.25 + 0.015 * idx,
            dispersion=0.035 + 0.004 * idx,
            retention_strength=0.020 + 0.006 * idx,
            rng=rng
        ))

    pd.DataFrame(btc_rows).to_csv("examples/synthetic_btc.csv", index=False)
    pd.DataFrame(snapshot_rows).to_csv("examples/synthetic_snapshots.csv", index=False)
    pd.DataFrame(truth_rows).to_csv("examples/synthetic_truth_10_sites.csv", index=False)

    print("Created 10-site synthetic dataset.")
    print(pd.DataFrame(truth_rows))

if __name__ == "__main__":
    main()
