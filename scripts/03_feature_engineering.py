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
def _extract_date_from_filename(nc_path: Path) -> str:
    """Extrai a data do nome do arquivo processado.
    
    Os arquivos já vêm com a data lógica correta do script 01:
    - 20250926_SSTfnd-MUR_proc.nc -> 2025-09-26
    - 20250926_CHL-MODIS_proc.nc -> 2025-09-26
    """
    # Pega os primeiros 8 dígitos do nome
    digits = "".join([c for c in nc_path.name if c.isdigit()])[:8]
    return pd.to_datetime(digits, format="%Y%m%d").strftime("%Y-%m-%d")


def interpolate_chlor_to_sst(sst_path: Path, chl_path: Path, dropna: bool) -> pd.DataFrame:
    """Interpola CHLOR_A (MODIS) para o grid do SST (MUR) e retorna DataFrame unificado."""
    # Abrir com chunks para evitar sobrecarga de memória
    sst_ds = xr.open_dataset(sst_path, chunks={'time': 1})
    chl_ds = xr.open_dataset(chl_path, chunks={'time': 1})

    # Data já está correta no nome do arquivo
    date_iso = _extract_date_from_filename(sst_path)

    # Squeeze time se necessário
    if 'time' in sst_ds.dims and sst_ds.sizes.get('time', 0) == 1:
        sst_ds = sst_ds.squeeze('time', drop=True)
    if 'time' in chl_ds.dims and chl_ds.sizes.get('time', 0) == 1:
        chl_ds = chl_ds.squeeze('time', drop=True)

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

    # Remove colunas desnecessárias que podem ter NaN
    cols_to_keep = ["lat", "lon", "sst", "sst_gradient", "chlor_a"]
    df = df[[col for col in cols_to_keep if col in df.columns]]

    # Adiciona data
    df["date"] = date_iso

    # Mantém só oceano: remove linhas sem SST
    if dropna:
        # Dataset enxuto: exige SST e CHL válidos
        df = df.dropna(subset=["sst", "chlor_a"], how="any")
    else:
        # Padrão: mantém todos os pontos com SST válido (oceano), CHL pode ser NaN
        df = df.dropna(subset=["sst"], how="any")

    # Remove linhas com lat/lon inválidos
    df = df.dropna(subset=["lat", "lon"], how="any")

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

    sst_files = sorted(PROC_DIR.glob("*SSTfnd*_proc.nc"))
    chl_files = sorted(PROC_DIR.glob("*CHL*_proc.nc"))

    if not sst_files:
        raise FileNotFoundError("Arquivos de SST não encontrados em data/processed/")
    if not chl_files:
        raise FileNotFoundError("Arquivos de CHL não encontrados em data/processed/")

    print(f"\n{'='*60}")
    print(f"Encontrados {len(sst_files)} arquivo(s) SST e {len(chl_files)} arquivo(s) CHL")
    print(f"{'='*60}\n")

    # Agrupa por data (usando o prefixo YYYYMMDD do nome)
    sst_by_date = {_extract_date_from_filename(f): f for f in sst_files}
    chl_by_date = {_extract_date_from_filename(f): f for f in chl_files}

    # Processa apenas datas que têm ambos SST e CHL
    common_dates = set(sst_by_date.keys()) & set(chl_by_date.keys())
    
    if not common_dates:
        print("❌ Nenhuma data com SST e CHL correspondentes encontrada!")
        return

    processed = 0
    failed = 0

    for date_iso in sorted(common_dates):
        sst_path = sst_by_date[date_iso]
        chl_path = chl_by_date[date_iso]

        try:
            df = interpolate_chlor_to_sst(sst_path, chl_path, args.dropna)
        except Exception as exc:
            print(f"❌ Falha ao processar {date_iso}: {exc}")
            failed += 1
            continue

        if df.empty:
            print(f"⚠️  {date_iso} -> sem dados válidos após interpolação")
            failed += 1
            continue

        out_file = FEATURES_DIR / f"{date_iso.replace('-', '')}_features.csv"
        
        # Reordenar colunas para melhor visualização
        col_order = ["date", "lat", "lon", "sst", "sst_gradient", "chlor_a"]
        df = df[[col for col in col_order if col in df.columns]]
        
        # Salvar com tratamento especial para NaN
        df.to_csv(out_file, index=False, float_format="%.6f", na_rep="NaN")

        print(
            f"✅ {date_iso} -> {out_file.name} "
            f"({len(df):,} linhas) | SST: {sst_path.name} | CHL: {chl_path.name}"
        )
        processed += 1

    print(f"\n{'='*60}")
    print(f"Geração de features concluída!")
    print(f"   ✅ Processados: {processed}")
    print(f"   ❌ Falhas: {failed}")
    print(f"   📂 Arquivos em: {FEATURES_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()