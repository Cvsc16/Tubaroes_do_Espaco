#!/usr/bin/env python3
"""Gera rasters de probabilidade a partir dos NetCDF processados e do modelo treinado."""

from pathlib import Path
import re
import numpy as np
import xarray as xr
import joblib
import rioxarray  # noqa: F401 - necessario para habilitar metodos .rio

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TILES_DIR = ROOT / "data" / "tiles"
TILES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROCESSED_DIR / "model_xgb.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError("Modelo nao encontrado. Rode 04_train_model.py antes de exportar tiles.")

model = joblib.load(MODEL_PATH)

PROC_FILES = sorted(PROCESSED_DIR.glob("*_proc.nc"))
if not PROC_FILES:
    raise FileNotFoundError("Nenhum NetCDF processado encontrado. Rode 02_preprocess.py primeiro.")


def prepare_slice(ds: xr.Dataset, time_index: int | None):
    sst = ds["sst"]
    grad = ds["sst_gradient"]

    if time_index is not None and "time" in sst.dims:
        sst = sst.isel(time=time_index, drop=True)
        grad = grad.isel(time=time_index, drop=True)
    elif "time" in sst.dims:
        sst = sst.isel(time=0, drop=True)
        grad = grad.isel(time=0, drop=True)

    for coord in ("lat", "lon"):
        if coord not in sst.coords:
            raise KeyError(f"Coordenada '{coord}' nao encontrada em {ds}")

    if sst["lat"].values[0] > sst["lat"].values[-1]:
        sst = sst.sortby("lat")
        grad = grad.sortby("lat")
    if sst["lon"].values[0] > sst["lon"].values[-1]:
        sst = sst.sortby("lon")
        grad = grad.sortby("lon")

    return sst, grad


def predict_probability(sst_da: xr.DataArray, grad_da: xr.DataArray):
    arr_sst = sst_da.values
    arr_grad = grad_da.values

    mask = ~np.isfinite(arr_sst) | ~np.isfinite(arr_grad)
    features = np.stack([arr_sst, arr_grad], axis=-1)
    features[mask] = 0.0

    proba = model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1]
    proba = proba.reshape(arr_sst.shape)
    proba = np.where(mask, np.nan, proba)

    proba_da = xr.DataArray(
        proba,
        coords={"y": sst_da["lat"].values, "x": sst_da["lon"].values},
        dims=("y", "x"),
        name="hotspot_probability",
    )
    proba_da = proba_da.rio.write_crs("EPSG:4326")
    return proba_da


for nc_path in PROC_FILES:
    with xr.open_dataset(nc_path) as ds:
        times = ds["time"].values if "time" in ds.dims else [None]
        for idx, time_val in enumerate(times):
            sst_da, grad_da = prepare_slice(ds, idx if time_val is not None else None)
            proba_da = predict_probability(sst_da, grad_da)

            if time_val is not None:
                timestamp = np.datetime_as_string(time_val, unit="s")
            else:
                timestamp = nc_path.stem.replace("_proc", "")

            safe_timestamp = re.sub(r"[:\\]", "-", timestamp)
            out_path = TILES_DIR / f"hotspots_probability_{safe_timestamp}.tif"
            proba_da.rio.to_raster(out_path, dtype="float32")
            print(f"Tile salvo em {out_path}")
