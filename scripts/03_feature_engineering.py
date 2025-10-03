#!/usr/bin/env python3
"""Gera features tabulares unificadas:
SST + gradiente + CHLOR_A (MODIS + PACE) + SSH (SWOT) sem extrapolar valores.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import xarray as xr
import numpy as np

# ---------------- bootstrap ----------------
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


def _compute_gradient(field: xr.DataArray) -> xr.DataArray:
    if "lat" not in field.dims or "lon" not in field.dims:
        return xr.full_like(field, np.nan)

    grad_y, grad_x = np.gradient(field.values)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return xr.DataArray(
        grad, dims=field.dims, coords=field.coords, name=f"{field.name}_gradient"
    )


def _estimate_step(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.1
    diffs = np.diff(np.sort(values))
    diffs = np.abs(diffs[diffs != 0])
    if diffs.size == 0:
        return 0.1
    return float(diffs.min())


def _swot_points_to_grid(swot_ds: xr.Dataset, target_lat: xr.DataArray, target_lon: xr.DataArray,
                         lat_tol_factor: float = 0.6, lon_tol_factor: float = 0.6) -> tuple[xr.DataArray, xr.DataArray]:
    """Converte pontos SWOT para a grade do SST sem extrapolar."""
    lat_vals = target_lat.values
    lon_vals = target_lon.values

    if lat_vals.ndim != 1 or lon_vals.ndim != 1:
        raise ValueError("lat/lon alvo devem ser 1D para projeção da SWOT.")

    lat_step = _estimate_step(lat_vals)
    lon_step = _estimate_step(lon_vals)
    lat_tol = max(lat_step * lat_tol_factor, 1e-3)
    lon_tol = max(lon_step * lon_tol_factor, 1e-3)

    ssh_sum = np.zeros((lat_vals.size, lon_vals.size), dtype=np.float32)
    counts = np.zeros((lat_vals.size, lon_vals.size), dtype=np.uint16)

    lat_points = swot_ds["lat"].values
    lon_points = swot_ds["lon"].values
    ssh_points = swot_ds["ssh"].values

    for lat_pt, lon_pt, ssh_val in zip(lat_points, lon_points, ssh_points):
        if not np.isfinite(ssh_val):
            continue
        i = int(np.argmin(np.abs(lat_vals - lat_pt)))
        if abs(lat_vals[i] - lat_pt) > lat_tol:
            continue
        j = int(np.argmin(np.abs(lon_vals - lon_pt)))
        if abs(lon_vals[j] - lon_pt) > lon_tol:
            continue
        ssh_sum[i, j] += ssh_val
        counts[i, j] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        ssh_mean = np.where(counts > 0, ssh_sum / counts, np.nan).astype(np.float32)

    mask = (counts > 0).astype(np.float32)

    ssh_da = xr.DataArray(
        ssh_mean,
        dims=("lat", "lon"),
        coords={"lat": target_lat, "lon": target_lon},
        name="ssh_swot"
    )
    mask_da = xr.DataArray(
        mask,
        dims=("lat", "lon"),
        coords={"lat": target_lat, "lon": target_lon},
        name="swot_mask"
    )
    return ssh_da, mask_da


def merge_datasets(
    sst_path: Path,
    modis_path: Path,
    pace_path: Path,
    swot_path: Path | None,
    dropna: bool,
    swot_distance_factor: float = 0.6
) -> pd.DataFrame:
    date_iso = _extract_date_from_filename(sst_path)

    sst_ds = xr.open_dataset(sst_path).squeeze(drop=True)
    modis_ds = xr.open_dataset(modis_path).squeeze(drop=True)
    pace_ds = xr.open_dataset(pace_path).squeeze(drop=True)

    modis_var = "chlor_a_modis" if "chlor_a_modis" in modis_ds.variables else "chlor_a"
    pace_var = "chlor_a_pace" if "chlor_a_pace" in pace_ds.variables else "chlor_a"

    modis_interp = modis_ds[modis_var].interp(lat=sst_ds.lat, lon=sst_ds.lon, method="linear")
    pace_interp = pace_ds[pace_var].interp(lat=sst_ds.lat, lon=sst_ds.lon, method="linear")

    datasets = [
        sst_ds[["sst", "sst_gradient"]],
        modis_interp.to_dataset(name="chlor_a_modis"),
        pace_interp.to_dataset(name="chlor_a_pace"),
    ]

    if swot_path and swot_path.exists():
        swot_ds = xr.open_dataset(swot_path).squeeze(drop=True)
        try:
            if "points" in swot_ds.dims:
                ssh_da, mask_da = _swot_points_to_grid(swot_ds, sst_ds.lat, sst_ds.lon, swot_distance_factor, swot_distance_factor)
                datasets.append(ssh_da.to_dataset())
                datasets.append(mask_da.to_dataset())
                ssh_grad = _compute_gradient(ssh_da.where(mask_da > 0))
                datasets.append(ssh_grad.to_dataset())
            elif "ssh_swot" in swot_ds.variables and {"lat", "lon"}.issubset(swot_ds["ssh_swot"].dims):
                ssh_da = swot_ds["ssh_swot"]
                ssh_da = ssh_da.sel(lat=sst_ds.lat, lon=sst_ds.lon, method="nearest")
                mask_da = xr.where(~np.isnan(ssh_da), 1.0, 0.0).astype(np.float32)
                datasets.append(ssh_da.to_dataset(name="ssh_swot"))
                datasets.append(mask_da.to_dataset(name="swot_mask"))
                ssh_grad = _compute_gradient(ssh_da.where(mask_da > 0))
                datasets.append(ssh_grad.to_dataset())
            else:
                print(f"[SWOT] {swot_path.name}: formato não suportado, ignorando")
        finally:
            swot_ds.close()

    merged = xr.merge(datasets, join="outer", combine_attrs="override")

    df = merged.to_dataframe().reset_index()
    df["date"] = date_iso

    required_vars = ["sst", "chlor_a_modis", "chlor_a_pace"]
    if dropna:
        df = df.dropna(subset=required_vars, how="any")
    else:
        df = df.dropna(subset=["sst"], how="any")

    cols = [
        "date", "lat", "lon", "sst", "sst_gradient",
        "chlor_a_modis", "chlor_a_pace",
        "ssh_swot", "ssh_swot_gradient",
        "swot_mask"
    ]
    df = df[[c for c in cols if c in df.columns]]

    for col in [
        "sst", "sst_gradient", "chlor_a_modis", "chlor_a_pace",
        "ssh_swot", "ssh_swot_gradient", "lat", "lon", "swot_mask"
    ]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    sst_ds.close(); modis_ds.close(); pace_ds.close()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropna", action="store_true",
                        help="Remove linhas sem MODIS/PACE (default: mantém NaN)")
    parser.add_argument("--swot-distance-factor", type=float, default=0.6,
                        help="Fator multiplicativo do passo da grade para aceitar pontos SWOT (default: 0.6)")
    args = parser.parse_args()

    sst_files = sorted(PROC_DIR.glob("*SSTfnd*_proc.nc"))
    modis_files = sorted(PROC_DIR.glob("*CHL-MODIS*_proc.nc"))
    pace_files = sorted(PROC_DIR.glob("*CHL-PACE*_proc.nc"))
    swot_files = sorted(PROC_DIR.glob("*SSH-SWOT*_points.nc"))
    if not swot_files:
        swot_files = sorted(PROC_DIR.glob("*SSH-SWOT*_proc.nc"))

    if not sst_files or not modis_files or not pace_files:
        raise FileNotFoundError("Arquivos processados de SST, MODIS ou PACE não encontrados em data/processed/")

    sst_by_date = {_extract_date_from_filename(f): f for f in sst_files}
    modis_by_date = {_extract_date_from_filename(f): f for f in modis_files}
    pace_by_date = {_extract_date_from_filename(f): f for f in pace_files}
    swot_by_date = {_extract_date_from_filename(f): f for f in swot_files}

    common_dates = set(sst_by_date) & set(modis_by_date) & set(pace_by_date)
    if not common_dates:
        print("❌ Nenhuma data com SST, MODIS e PACE correspondentes encontrada!")
        return

    processed, failed = 0, 0
    for date_iso in sorted(common_dates):
        try:
            swot_file = swot_by_date.get(date_iso)
            df = merge_datasets(
                sst_by_date[date_iso],
                modis_by_date[date_iso],
                pace_by_date[date_iso],
                swot_file,
                args.dropna,
                args.swot_distance_factor
            )
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
