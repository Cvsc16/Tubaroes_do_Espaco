#!/usr/bin/env python3
"""Gera GeoTIFFs de probabilidade a partir do modelo treinado (SST + gradiente + CHL)."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re
from typing import Dict, Tuple

import numpy as np
import xarray as xr
import joblib
import rioxarray  # noqa: F401

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent

import sys
if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))


ROOT = _PROJECT_ROOT_FALLBACK
PROCESSED_DIR = ROOT / "data" / "processed"
TILES_DIR = ROOT / "data" / "tiles"
TILES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROCESSED_DIR / "model_xgb.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError("Modelo nao encontrado. Rode 04_train_model.py antes de exportar tiles.")

model = joblib.load(MODEL_PATH)

LAT_CANDIDATES = ("lat", "latitude")
LON_CANDIDATES = ("lon", "longitude")


def detect_coordinate(da: xr.DataArray, candidates: Tuple[str, ...]) -> str:
    for name in candidates:
        if name in da.coords:
            return name
    for dim in da.dims:
        lower = dim.lower()
        if any(lower.startswith(prefix[:3]) for prefix in candidates):
            return dim
    raise KeyError(f"Nao encontrei coordenada correspondente aos candidatos {candidates}")


def ensure_sorted(da: xr.DataArray, coord: str) -> xr.DataArray:
    values = da[coord]
    if values[0] > values[-1]:
        return da.sortby(coord)
    return da


def extract_date(ds: xr.Dataset, path: Path) -> str:
    if "time" in ds and ds["time"].size:
        timestamp = np.array(ds["time"].values).ravel()[0]
        try:
            return np.datetime_as_string(np.datetime64(timestamp), unit="D")
        except Exception:
            pass
    match = re.search(r"(20\d{6})", path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")
    raise ValueError(f"Nao foi possivel determinar a data para {path.name}")


def list_processed_files() -> Tuple[list[Path], Dict[str, Path]]:
    sst_files: list[Path] = []
    chl_map: Dict[str, Path] = {}
    for path in sorted(PROCESSED_DIR.glob("*_proc.nc")):
        with xr.open_dataset(path) as ds:
            if "sst" in ds.data_vars:
                sst_files.append(path)
            elif "chlor_a" in ds.data_vars:
                date_iso = extract_date(ds, path)
                chl_map[date_iso] = Path(path)
    if not sst_files:
        raise FileNotFoundError("Nenhum arquivo de SST processado encontrado. Rode 02_preprocess.py primeiro.")
    return sst_files, chl_map


SST_FILES, CHL_BY_DATE = list_processed_files()


def prepare_slice(ds: xr.Dataset, time_index: int | None):
    sst = ds["sst"]
    grad = ds["sst_gradient"]

    if time_index is not None and "time" in sst.dims:
        sst = sst.isel(time=time_index, drop=True)
        grad = grad.isel(time=time_index, drop=True)
    elif "time" in sst.dims:
        sst = sst.isel(time=0, drop=True)
        grad = grad.isel(time=0, drop=True)

    lat_name = detect_coordinate(sst, LAT_CANDIDATES)
    lon_name = detect_coordinate(sst, LON_CANDIDATES)

    sst = ensure_sorted(sst, lon_name)
    sst = ensure_sorted(sst, lat_name)
    grad = grad.sel({lat_name: sst[lat_name], lon_name: sst[lon_name]})
    grad = ensure_sorted(grad, lon_name)
    grad = ensure_sorted(grad, lat_name)

    return sst, grad, lat_name, lon_name


def load_chl_for_date(date_iso: str, target_lat: xr.DataArray, target_lon: xr.DataArray, lat_name: str, lon_name: str):
    chl_path = CHL_BY_DATE.get(date_iso)
    if not chl_path:
        return None

    with xr.open_dataset(chl_path) as ds:
        chl = ds["chlor_a"].load()

    chl_lat = detect_coordinate(chl, LAT_CANDIDATES)
    chl_lon = detect_coordinate(chl, LON_CANDIDATES)

    chl = ensure_sorted(chl, chl_lon)
    chl = ensure_sorted(chl, chl_lat)

    # interpolar para a grade alvo (nearest)
    chl_interp = chl.interp({chl_lat: target_lat, chl_lon: target_lon}, method="nearest")
    return chl_interp


def predict_probability(sst_da: xr.DataArray, grad_da: xr.DataArray, chl_da: xr.DataArray | None, lat_name: str, lon_name: str):
    arr_sst = sst_da.values
    arr_grad = grad_da.values
    mask = ~np.isfinite(arr_sst) | ~np.isfinite(arr_grad)

    if chl_da is not None:
        arr_chl = chl_da.values
    else:
        arr_chl = np.full_like(arr_sst, np.nan, dtype=float)

    mask = mask | ~np.isfinite(arr_chl)
    stack = np.stack([arr_sst, arr_grad, arr_chl], axis=-1)
    stack[mask] = 0.0

    proba = model.predict_proba(stack.reshape(-1, stack.shape[-1]))[:, 1]
    proba = proba.reshape(arr_sst.shape)
    proba = np.where(mask, np.nan, proba)

    da = xr.DataArray(
        proba,
        coords={lat_name: sst_da[lat_name], lon_name: sst_da[lon_name]},
        dims=(lat_name, lon_name),
        name="hotspot_probability",
    )
    da = da.rename({lat_name: "y", lon_name: "x"}).rio.write_crs("EPSG:4326")
    return da


for sst_path in SST_FILES:
    with xr.open_dataset(sst_path) as ds:
        times = ds["time"].values if "time" in ds.dims else [None]
        for idx, time_val in enumerate(times):
            sst_da, grad_da, lat_name, lon_name = prepare_slice(ds, idx if time_val is not None else None)
            if time_val is not None:
                timestamp = np.datetime_as_string(time_val, unit="s")
            else:
                timestamp = sst_path.stem.replace("_proc", "")
            date_iso = timestamp.split("T")[0]

            chl_da = load_chl_for_date(date_iso, sst_da[lat_name], sst_da[lon_name], lat_name, lon_name)
            proba_da = predict_probability(sst_da, grad_da, chl_da, lat_name, lon_name)

            safe_timestamp = re.sub(r"[:\]", "-", timestamp)
            out_path = TILES_DIR / f"hotspots_probability_{safe_timestamp}.tif"
            proba_da.rio.to_raster(out_path, dtype="float32")
            print(f"Tile salvo em {out_path}")
