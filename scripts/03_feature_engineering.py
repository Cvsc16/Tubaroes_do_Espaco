#!/usr/bin/env python3
"""Gera features tabulares unificadas:
SST + gradiente + CHLOR_A (MODIS + PACE) + produtos MOANA + SSH (SWOT).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr

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

LAT_CANDIDATES = ("lat", "latitude")
LON_CANDIDATES = ("lon", "longitude")


def detect_coordinate(da: xr.DataArray, candidates: tuple[str, ...]) -> str:
    """Retorna o nome da coordenada lat/lon mais provavel."""

    for name in candidates:
        if name in da.coords:
            return name
    for dim in da.dims:
        low = dim.lower()
        if any(low.startswith(prefix[:3]) for prefix in candidates):
            return dim
    raise KeyError(f"Nenhuma coordenada encontrada para {candidates} em {da.dims}")


def _standardize_lat_lon(ds: xr.Dataset, sample_var: str) -> tuple[xr.Dataset, xr.DataArray, xr.DataArray]:
    """Garante que as dimensoes espaciais sejam chamadas de lat/lon."""

    da = ds[sample_var]
    lat_name = detect_coordinate(da, LAT_CANDIDATES)
    lon_name = detect_coordinate(da, LON_CANDIDATES)
    rename_map: Dict[str, str] = {}
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds, ds["lat"], ds["lon"]


def _rename_da_dims(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    rename_map: Dict[str, str] = {}
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if rename_map:
        da = da.rename(rename_map)
    return da


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


def _swot_points_to_grid(
    swot_ds: xr.Dataset,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
    lat_tol_factor: float = 0.6,
    lon_tol_factor: float = 0.6,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Converte pontos SWOT para a grade do SST sem extrapolar."""

    lat_vals = target_lat.values
    lon_vals = target_lon.values

    if lat_vals.ndim != 1 or lon_vals.ndim != 1:
        raise ValueError("lat/lon alvo devem ser 1D para projecao da SWOT.")

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
        name="ssh_swot",
    )
    mask_da = xr.DataArray(
        mask,
        dims=("lat", "lon"),
        coords={"lat": target_lat, "lon": target_lon},
        name="swot_mask",
    )
    return ssh_da, mask_da


def merge_datasets(
    sst_path: Path,
    modis_path: Path,
    pace_path: Path,
    moana_path: Path | None,
    swot_path: Path | None,
    dropna: bool,
    swot_distance_factor: float = 0.6,
) -> pd.DataFrame:
    sst_ds = xr.open_dataset(sst_path).squeeze(drop=True)
    modis_ds = xr.open_dataset(modis_path).squeeze(drop=True)
    pace_ds = xr.open_dataset(pace_path).squeeze(drop=True)
    moana_ds = (
        xr.open_dataset(moana_path).squeeze(drop=True)
        if moana_path is not None and moana_path.exists()
        else None
    )

    try:
        sst_var = "sst" if "sst" in sst_ds.data_vars else list(sst_ds.data_vars.keys())[0]
        sst_ds, target_lat, target_lon = _standardize_lat_lon(sst_ds, sst_var)

        datasets = [sst_ds[["sst", "sst_gradient"]]]

        modis_var = "chlor_a_modis" if "chlor_a_modis" in modis_ds.variables else "chlor_a"
        modis_da = modis_ds[modis_var]
        lat_modis = detect_coordinate(modis_da, LAT_CANDIDATES)
        lon_modis = detect_coordinate(modis_da, LON_CANDIDATES)
        modis_interp = modis_da.interp({lat_modis: target_lat, lon_modis: target_lon}, method="linear")
        modis_interp = _rename_da_dims(modis_interp, lat_modis, lon_modis)
        datasets.append(modis_interp.astype(np.float32).to_dataset(name="chlor_a_modis"))

        pace_var = "chlor_a_pace" if "chlor_a_pace" in pace_ds.variables else "chlor_a"
        pace_da = pace_ds[pace_var]
        lat_pace = detect_coordinate(pace_da, LAT_CANDIDATES)
        lon_pace = detect_coordinate(pace_da, LON_CANDIDATES)
        pace_interp = pace_da.interp({lat_pace: target_lat, lon_pace: target_lon}, method="linear")
        pace_interp = _rename_da_dims(pace_interp, lat_pace, lon_pace)
        datasets.append(pace_interp.astype(np.float32).to_dataset(name="chlor_a_pace"))

        if moana_ds is not None:
            for var_name, da in moana_ds.data_vars.items():
                if da.ndim < 2:
                    continue
                lat_moana = detect_coordinate(da, LAT_CANDIDATES)
                lon_moana = detect_coordinate(da, LON_CANDIDATES)
                moana_interp = da.interp({lat_moana: target_lat, lon_moana: target_lon}, method="nearest")
                moana_interp = _rename_da_dims(moana_interp, lat_moana, lon_moana)
                datasets.append(moana_interp.astype(np.float32).to_dataset(name=var_name))

        if swot_path and swot_path.exists():
            swot_ds = xr.open_dataset(swot_path).squeeze(drop=True)
            try:
                if "points" in swot_ds.dims:
                    ssh_da, mask_da = _swot_points_to_grid(
                        swot_ds,
                        target_lat,
                        target_lon,
                        swot_distance_factor,
                        swot_distance_factor,
                    )
                    datasets.append(ssh_da.to_dataset())
                    datasets.append(mask_da.to_dataset())
                    ssh_grad = _compute_gradient(ssh_da.where(mask_da > 0))
                    datasets.append(ssh_grad.to_dataset())
                elif "ssh_swot" in swot_ds.variables and {"lat", "lon"}.issubset(swot_ds["ssh_swot"].dims):
                    ssh_da = swot_ds["ssh_swot"].sel(lat=target_lat, lon=target_lon, method="nearest")
                    mask_da = xr.where(~np.isnan(ssh_da), 1.0, 0.0).astype(np.float32)
                    datasets.append(ssh_da.astype(np.float32).to_dataset(name="ssh_swot"))
                    datasets.append(mask_da.to_dataset(name="swot_mask"))
                    ssh_grad = _compute_gradient(ssh_da.where(mask_da > 0))
                    datasets.append(ssh_grad.to_dataset())
                else:
                    print(f"[warn] {swot_path.name}: formato SWOT nao reconhecido, ignorando")
            finally:
                swot_ds.close()

        merged = xr.merge(datasets, join="outer", combine_attrs="override")
        df = merged.to_dataframe().reset_index()
        df["date"] = _extract_date_from_filename(sst_path)

        required_vars = ["sst", "chlor_a_modis", "chlor_a_pace"]
        if dropna:
            df = df.dropna(subset=required_vars, how="any")
        else:
            df = df.dropna(subset=["sst"], how="any")

        if "chlor_a" not in df.columns:
            if "chlor_a_pace" in df.columns and "chlor_a_modis" in df.columns:
                df["chlor_a"] = df["chlor_a_pace"].fillna(df["chlor_a_modis"])
            elif "chlor_a_pace" in df.columns:
                df["chlor_a"] = df["chlor_a_pace"]
            elif "chlor_a_modis" in df.columns:
                df["chlor_a"] = df["chlor_a_modis"]

        moana_cols = [col for col in df.columns if col.startswith("moana_")]
        cols = [
            "date",
            "lat",
            "lon",
            "sst",
            "sst_gradient",
            "chlor_a_modis",
            "chlor_a_pace",
            "chlor_a",
            "ssh_swot",
            "ssh_swot_gradient",
            "swot_mask",
            *moana_cols,
        ]
        df = df[[c for c in cols if c in df.columns]]

        for col in [
            "sst",
            "sst_gradient",
            "chlor_a_modis",
            "chlor_a_pace",
            "chlor_a",
            "ssh_swot",
            "ssh_swot_gradient",
            "lat",
            "lon",
            "swot_mask",
            *moana_cols,
        ]:
            if col in df.columns:
                df[col] = df[col].astype("float32")

        return df
    finally:
        sst_ds.close()
        modis_ds.close()
        pace_ds.close()
        if moana_ds is not None:
            moana_ds.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dropna",
        action="store_true",
        help="Remove linhas sem MODIS/PACE/MOANA (default: mantem NaN)",
    )
    parser.add_argument(
        "--swot-distance-factor",
        type=float,
        default=0.6,
        help="Fator multiplicativo do passo da grade para aceitar pontos SWOT (default: 0.6)",
    )
    args = parser.parse_args()

    sst_files = sorted(PROC_DIR.glob("*SSTfnd*_proc.nc"))
    modis_files = sorted(PROC_DIR.glob("*CHL-MODIS*_proc.nc"))
    pace_files = sorted(PROC_DIR.glob("*CHL-PACE*_proc.nc"))
    moana_files = sorted(PROC_DIR.glob("*MOANA-PACE*_proc.nc"))
    swot_files = sorted(PROC_DIR.glob("*SSH-SWOT*_points.nc"))
    if not swot_files:
        swot_files = sorted(PROC_DIR.glob("*SSH-SWOT*_proc.nc"))

    if not sst_files or not modis_files or not pace_files:
        raise FileNotFoundError("Arquivos processados de SST, MODIS ou PACE nao encontrados em data/processed/")

    sst_by_date = {_extract_date_from_filename(f): f for f in sst_files}
    modis_by_date = {_extract_date_from_filename(f): f for f in modis_files}
    pace_by_date = {_extract_date_from_filename(f): f for f in pace_files}
    moana_by_date = {_extract_date_from_filename(f): f for f in moana_files}
    swot_by_date = {_extract_date_from_filename(f): f for f in swot_files}

    common_dates = set(sst_by_date) & set(modis_by_date) & set(pace_by_date)
    if moana_by_date:
        common_dates &= set(moana_by_date)
    else:
        print("[warn] Nenhum arquivo MOANA encontrado. Seguiremos sem essas variaveis.")
    if not common_dates:
        print("[warn] Nenhuma data com SST, MODIS e PACE (e MOANA) correspondentes encontrada!")
        return

    processed, failed = 0, 0
    for date_iso in sorted(common_dates):
        try:
            moana_file = moana_by_date.get(date_iso)
            swot_file = swot_by_date.get(date_iso)
            df = merge_datasets(
                sst_by_date[date_iso],
                modis_by_date[date_iso],
                pace_by_date[date_iso],
                moana_file,
                swot_file,
                args.dropna,
                args.swot_distance_factor,
            )
            if df.empty:
                print(f"[warn] {date_iso} -> sem dados validos")
                failed += 1
                continue

            out_file = FEATURES_DIR / f"{date_iso.replace('-', '')}_features.csv"
            df.to_csv(out_file, index=False, float_format="%.6f", na_rep="NaN")
            print(f"[ok] {date_iso} -> {out_file.name} ({len(df):,} linhas)")
            processed += 1
        except Exception as exc:
            print(f"[error] Falha em {date_iso}: {exc}")
            failed += 1

    print("\n" + "=" * 60)
    print("Geracao de features concluida")
    print(f"   [ok] Processados: {processed}")
    print(f"   [warn] Falhas: {failed}")
    print(f"   Arquivos em: {FEATURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
