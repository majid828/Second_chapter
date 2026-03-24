from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


REQUIRED_BTC_COLUMNS = {"site", "time", "concentration"}
REQUIRED_SNAPSHOT_COLUMNS = {"site", "time", "distance", "concentration"}


def load_config(config_path: str | Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_btc_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_BTC_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"BTC CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["site"] = df["site"].astype(str)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["concentration"] = pd.to_numeric(df["concentration"], errors="coerce")
    df = df.dropna(subset=["site", "time", "concentration"]).sort_values(["site", "time"])
    return df


def load_snapshot_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_SNAPSHOT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Snapshot CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["site"] = df["site"].astype(str)
    for col in ["time", "distance", "concentration"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["site", "time", "distance", "concentration"]).sort_values(["site", "time", "distance"])
    return df


def split_by_site(
    btc_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    sites = sorted(set(btc_df["site"]).intersection(set(snapshot_df["site"])))
    if not sites:
        raise ValueError("No common sites found between BTC and snapshot files.")
    out = {}
    for site in sites:
        out[site] = (
            btc_df.loc[btc_df["site"] == site].copy(),
            snapshot_df.loc[snapshot_df["site"] == site].copy(),
        )
    return out


def save_json(data: Dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
