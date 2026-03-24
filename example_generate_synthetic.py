from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RNG = np.random.default_rng(123)
OUTDIR = Path("examples")
OUTDIR.mkdir(parents=True, exist_ok=True)


def tempered_power_law(t, a, lam, gamma, c):
    return a * (t + c) ** (-lam) * np.exp(-gamma * t)


def make_btc_for_site(site: str, a: float, lam: float, gamma: float, c: float):
    t = np.linspace(0.2, 30.0, 70)
    clean = tempered_power_law(t, a, lam, gamma, c)
    # Shape BTC more realistically as pulse-like: rise then fall.
    pulse = (t ** 2) * np.exp(-0.35 * t)
    pulse = pulse / pulse.max()
    clean = clean * pulse
    noise = RNG.normal(0.0, 0.05 * np.max(clean), size=t.size)
    obs = np.maximum(clean + noise, 0.0)
    return pd.DataFrame({"site": site, "time": t, "concentration": obs})


def make_snapshots_for_site(site: str, speed: float, spread_scale: float):
    rows = []
    times = [4, 8, 12, 18, 24]
    x = np.linspace(0, 120, 45)
    for t in times:
        center = speed * t
        spread = spread_scale * np.sqrt(t + 1)
        clean = np.exp(-0.5 * ((x - center) / spread) ** 2)
        tail = 0.35 * np.exp(-0.04 * np.maximum(x - center, 0))
        clean = clean + tail
        clean = clean / clean.max()
        noise = RNG.normal(0.0, 0.07, size=x.size)
        obs = np.maximum(clean + noise, 0.0)
        rows.append(pd.DataFrame({
            "site": site,
            "time": t,
            "distance": x,
            "concentration": obs,
        }))
    return pd.concat(rows, ignore_index=True)


def main():
    btc = pd.concat([
        make_btc_for_site("site_A", 1.4, 0.9, 0.08, 0.3),
        make_btc_for_site("site_B", 1.6, 1.1, 0.06, 0.4),
        make_btc_for_site("site_C", 1.3, 0.8, 0.10, 0.25),
    ], ignore_index=True)

    snapshots = pd.concat([
        make_snapshots_for_site("site_A", speed=3.0, spread_scale=4.0),
        make_snapshots_for_site("site_B", speed=2.6, spread_scale=5.0),
        make_snapshots_for_site("site_C", speed=3.3, spread_scale=4.5),
    ], ignore_index=True)

    btc.to_csv(OUTDIR / "synthetic_btc.csv", index=False)
    snapshots.to_csv(OUTDIR / "synthetic_snapshots.csv", index=False)
    print("Synthetic example files written to examples/")


if __name__ == "__main__":
    main()
