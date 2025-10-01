#!/usr/bin/env python3
"""Gera features tabulares unificadas: SST + gradiente + CHLOR_A (MODIS), com interpolação."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------
# Setup de paths
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------
def _extract_date_from_ds(ds: xr.Dataset, nc_path: Path) -> str:
    """Extrai a data do dataset ou do nome do arquivo."""
    if "time" in ds and ds["time"].size > 0:
        dt64 = np.array(ds["time"]).ravel()[0]
        date_iso = pd.to_datetime(dt64).strftime("%Y-%m-%d")
    else:
        token = "".join([c for c in nc_path.name if c.isdigit()])
        date_iso = pd.to_datetime(token[:8], format="%Y%m%d").strftime("%Y-%m-%d")

    source = nc_path.name.lower()
    if "sstfnd-mur" in source:
        # Ajuste do MUR: arquivos vêm com timestamp de +1 dia
        date_iso = (pd.to_datetime(date_iso) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    return date_iso


def interpolate_chlor_to_sst(sst_path: Path, chl_path: Path, dropna: bool) -> pd.DataFrame:
    """Interpola CHLOR_A (MODIS) para o grid do SST (MUR) e retorna DataFrame unificado (apenas oceano)."""
    sst_ds = xr.open_dataset(sst_path)
    chl_ds = xr.open_dataset(chl_path)

    date_iso = _extract_date_from_ds(sst_ds, sst_path)

    # Interpola chlor_a para o grid do SST
    chl_interp = chl_ds.interp(lat=sst_ds.lat, lon=sst_ds.lon, method="linear")

    # Garante exatamente o mesmo grid do SST (preenche CHL com NaN onde faltar)
    chl_on_sst = chl_interp.reindex_like(sst_ds, method=None)

    # Junta variáveis preservando todo grid do SST
    merged = xr.merge(
        [sst_ds[["sst", "sst_gradient"]], chl_on_sst[["chlor_a"]]],
        join="outer",
        combine_attrs="override"
    )

    # Converte para DataFrame
    df = merged.to_dataframe().reset_index()

    # Adiciona data
    df["date"] = date_iso

    # Mantém só oceano: remove linhas sem SST
    if dropna:
        # Dataset enxuto: exige SST e CHL válidos
        df = df.dropna(subset=["sst", "chlor_a"], how="any")
    else:
        # Padrão: mantém todos os pontos com SST válido (oceano), CHL pode ser NaN
        df = df.dropna(subset=["sst"], how="any")

    # Conversão para float32 (mais leve)
    for col in ["sst", "sst_gradient", "chlor_a", "lat", "lon"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    sst_ds.close()
    chl_ds.close()

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropna", action="store_true", help="Remove linhas sem chlor_a (default: mantém NaN)")
    args = parser.parse_args()

    sst_files = sorted(PROC_DIR.glob("*SSTfnd*.nc"))
    chl_files = sorted(PROC_DIR.glob("*CHL*.nc"))

    if not sst_files or not chl_files:
        raise FileNotFoundError("Arquivos de SST ou CHL não encontrados em data/processed/")

    # Processa em pares (assumindo ordenação por data equivalente)
    for sst_path, chl_path in zip(sst_files, chl_files):
        try:
            df = interpolate_chlor_to_sst(sst_path, chl_path, args.dropna)
        except Exception as exc:
            print(f"[features] Falha ao processar {sst_path.name} + {chl_path.name}: {exc}")
            continue

        if df.empty:
            print(f"[features] {sst_path.name} + {chl_path.name} -> sem dados válidos após interpolação")
            continue

        date_iso = df["date"].iloc[0]
        out_file = FEATURES_DIR / f"{date_iso.replace('-', '')}_features.csv"
        df.to_csv(out_file, index=False, float_format="%.6f")

        # Print compacto informando fontes e total de linhas
        print(
            f"[features] {date_iso} -> Features salvas em {out_file} "
            f"({len(df)} linhas) | SST: {sst_path.name} | CHL: {chl_path.name}"
        )


if __name__ == "__main__":
    main()
