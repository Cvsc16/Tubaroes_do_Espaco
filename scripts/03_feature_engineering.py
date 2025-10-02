#!/usr/bin/env python3
"""Gera features tabulares unificadas: SST + gradiente + CHLOR_A (MODIS + PACE), com interpolação."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
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
except ModuleNotFoundError:
    from utils_config import project_root

ROOT = project_root()
PROC_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def _extract_date_from_filename(nc_path: Path) -> str:
    digits = "".join([c for c in nc_path.name if c.isdigit()])[:8]
    return pd.to_datetime(digits, format="%Y%m%d").strftime("%Y-%m-%d")


def interpolate_to_sst(sst_ds: xr.Dataset, chl_ds: xr.Dataset, var: str, new_name: str) -> xr.DataArray:
    """Interpola CHL (MODIS ou PACE) para o grid do SST e renomeia variável."""
    chl_interp = chl_ds.interp(lat=sst_ds.lat, lon=sst_ds.lon, method="linear")
    chl_on_sst = chl_interp.reindex_like(sst_ds, method=None)
    chl_on_sst = chl_on_sst.rename({var: new_name})
    return chl_on_sst[new_name]


def merge_datasets(sst_path: Path, modis_path: Path, pace_path: Path, dropna: bool) -> pd.DataFrame:
    date_iso = _extract_date_from_filename(sst_path)

    sst_ds = xr.open_dataset(sst_path).squeeze(drop=True)
    modis_ds = xr.open_dataset(modis_path).squeeze(drop=True)
    pace_ds = xr.open_dataset(pace_path).squeeze(drop=True)

    # MODIS já vem como chlor_a_modis
    modis_var = "chlor_a_modis" if "chlor_a_modis" in modis_ds.variables else "chlor_a"
    # PACE já vem como chlor_a_pace
    pace_var = "chlor_a_pace" if "chlor_a_pace" in pace_ds.variables else "chlor_a"

    # Interpolar MODIS e PACE para grid do SST
    modis_interp = modis_ds[modis_var].interp(lat=sst_ds.lat, lon=sst_ds.lon, method="linear")
    pace_interp  = pace_ds[pace_var].interp(lat=sst_ds.lat, lon=sst_ds.lon, method="linear")

    merged = xr.merge(
        [sst_ds[["sst", "sst_gradient"]],
         modis_interp.to_dataset(name="chlor_a_modis"),
         pace_interp.to_dataset(name="chlor_a_pace")],
        join="outer", combine_attrs="override"
    )

    df = merged.to_dataframe().reset_index()
    df["date"] = date_iso

    if dropna:
        df = df.dropna(subset=["sst", "chlor_a_modis", "chlor_a_pace"], how="any")
    else:
        df = df.dropna(subset=["sst"], how="any")

    cols = ["date", "lat", "lon", "sst", "sst_gradient", "chlor_a_modis", "chlor_a_pace"]
    df = df[[c for c in cols if c in df.columns]]

    for col in ["sst", "sst_gradient", "chlor_a_modis", "chlor_a_pace", "lat", "lon"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    sst_ds.close(); modis_ds.close(); pace_ds.close()
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropna", action="store_true", help="Remove linhas sem MODIS/PACE (default: mantém NaN)")
    args = parser.parse_args()

    sst_files = sorted(PROC_DIR.glob("*SSTfnd*_proc.nc"))
    modis_files = sorted(PROC_DIR.glob("*CHL-MODIS*_proc.nc"))
    pace_files = sorted(PROC_DIR.glob("*CHL-PACE*_proc.nc"))

    if not sst_files or not modis_files or not pace_files:
        raise FileNotFoundError("Arquivos processados de SST, MODIS ou PACE não encontrados em data/processed/")

    sst_by_date = {_extract_date_from_filename(f): f for f in sst_files}
    modis_by_date = {_extract_date_from_filename(f): f for f in modis_files}
    pace_by_date = {_extract_date_from_filename(f): f for f in pace_files}

    common_dates = set(sst_by_date) & set(modis_by_date) & set(pace_by_date)
    if not common_dates:
        print("❌ Nenhuma data com SST, MODIS e PACE correspondentes encontrada!")
        return

    processed, failed = 0, 0
    for date_iso in sorted(common_dates):
        try:
            df = merge_datasets(sst_by_date[date_iso], modis_by_date[date_iso], pace_by_date[date_iso], args.dropna)
            if df.empty:
                print(f"⚠️  {date_iso} -> sem dados válidos")
                failed += 1
                continue

            out_file = FEATURES_DIR / f"{date_iso.replace('-', '')}_features.csv"
            df.to_csv(out_file, index=False, float_format="%.6f", na_rep="NaN")
            print(f"✅ {date_iso} -> {out_file.name} ({len(df):,} linhas)")
            processed += 1
        except Exception as exc:
            print(f"❌ Falha em {date_iso}: {exc}")
            failed += 1

    print(f"\n{'='*60}")
    print("🏁 Geração de features concluída")
    print(f"   ✅ Processados: {processed}")
    print(f"   ❌ Falhas: {failed}")
    print(f"   📂 Arquivos em: {FEATURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
