#!/usr/bin/env python3
"""Gera features tabulares combinando SST, gradiente e clorofila (MODIS)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent

if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))

try:
    from scripts.utils import project_root
except ModuleNotFoundError:  # fallback quando chamado diretamente
    from utils_config import project_root


ROOT = project_root()
PROC_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


LAT_NAMES = {"lat", "latitude"}
LON_NAMES = {"lon", "longitude"}


def _normalize_coords(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    if "latitude" in df.columns and "lat" not in df.columns:
        rename_map["latitude"] = "lat"
    if "longitude" in df.columns and "lon" not in df.columns:
        rename_map["longitude"] = "lon"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _extract_date_from_ds(ds: xr.Dataset, nc_path: Path) -> str:
    if "time" in ds and ds["time"].size > 0:
        dt64 = np.array(ds["time"]).ravel()[0]
        date_iso = pd.to_datetime(dt64).strftime("%Y-%m-%d")
    else:
        token = nc_path.name.split("JPL")[0][:8]
        date_iso = pd.to_datetime(token, format="%Y%m%d").strftime("%Y-%m-%d")

    source = nc_path.name.lower()
    if "sstfnd-mur" in source:
        date_iso = (pd.to_datetime(date_iso) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    return date_iso


def _load_dataframe(nc_path: Path) -> tuple[str, Dict[str, pd.DataFrame]]:
    ds = xr.open_dataset(nc_path)
    data: Dict[str, pd.DataFrame] = {}

    date_iso = _extract_date_from_ds(ds, nc_path)

    if {"sst", "sst_gradient"}.issubset(ds.data_vars):
        df = ds[["sst", "sst_gradient"]].to_dataframe().reset_index()
        df = _normalize_coords(df)
        df["date"] = pd.to_datetime(df.get("time", date_iso)).dt.strftime("%Y-%m-%d")
        df = df.drop(columns=["time"], errors="ignore")
        data["sst"] = df
    if "chlor_a" in ds.data_vars:
        df_chl = ds[["chlor_a"]].to_dataframe().reset_index()
        df_chl = _normalize_coords(df_chl)
        df_chl["date"] = pd.to_datetime(df_chl.get("time", date_iso)).dt.strftime("%Y-%m-%d")
        df_chl = df_chl.drop(columns=["time"], errors="ignore")
        data["chl"] = df_chl

    if not data:
        raise ValueError(f"Nenhuma variavel reconhecida em {nc_path.name}")

    return date_iso, data


def main() -> None:
    files = sorted(PROC_DIR.glob("*_proc.nc"))
    if not files:
        raise FileNotFoundError("Nenhum arquivo processado em data/processed/")

    per_date: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
    sources: Dict[str, str] = {}

    for nc_path in files:
        try:
            date_iso, data = _load_dataframe(nc_path)
        except Exception as exc:
            print(f"[features] Falha ao ler {nc_path.name}: {exc}")
            continue

        for key, df in data.items():
            per_date[date_iso][key] = df
            if key == "sst":
                sources[date_iso] = nc_path.name

    if not per_date:
        raise RuntimeError("Nenhum dataset válido encontrado para gerar features.")

    for date_iso, group in per_date.items():
        if "sst" not in group:
            print(f"[features] Ignorando {date_iso}: arquivo de SST ausente.")
            continue

        sst_df = group["sst"].copy()
        sst_df["lat_round"] = sst_df["lat"].round(4)
        sst_df["lon_round"] = sst_df["lon"].round(4)

        if "chl" in group:
            chl_df = group["chl"].copy()
            chl_df["lat_round"] = chl_df["lat"].round(4)
            chl_df["lon_round"] = chl_df["lon"].round(4)
            merged = sst_df.merge(
                chl_df[["lat_round", "lon_round", "date", "chlor_a"]],
                on=["lat_round", "lon_round", "date"],
                how="left",
            )
        else:
            merged = sst_df
            merged["chlor_a"] = np.nan

        merged = merged.drop(columns=["lat_round", "lon_round"])
        merged["source_file"] = sources.get(date_iso, "")

        out_csv = FEATURES_DIR / f"{date_iso.replace('-', '')}_features.csv"
        merged.to_csv(out_csv, index=False)
        print(f"[features] {date_iso} -> {out_csv} ({len(merged)} linhas)")


if __name__ == "__main__":
    main()
