#!/usr/bin/env python3
"""Interactive side-by-side comparison of MODIS True Color versus SST/gradient."""

from __future__ import annotations

import base64
import datetime as dt
import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import xarray as xr
from PIL import Image

from scripts.utils import get_bbox, load_config, project_root

ROOT = project_root()
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "compare"
OUT.mkdir(parents=True, exist_ok=True)

CFG = load_config()
BBOX = get_bbox(CFG) or [-80.0, 25.0, -60.0, 40.0]
MODIS_SIZE = 1024  # target resolution for the MODIS tile
TARGET_PIXELS = 2000  # approximate grid size for SST/gradient heatmaps


def download_modis_truecolor(date: str, out_file: Path) -> Path:
    """Fetch MODIS Aqua True Color via Worldview WMS and cache on disk."""

    base_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    layer = "MODIS_Aqua_CorrectedReflectance_TrueColor"
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/jpeg",
        "BBOX": ",".join(map(str, BBOX)),
        "WIDTH": MODIS_SIZE,
        "HEIGHT": MODIS_SIZE,
        "SRS": "EPSG:4326",
        "TIME": date,
    }

    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        Image.open(io.BytesIO(response.content)).convert("RGB").save(out_file)
    except Exception as exc:
        print(f"WARNING: MODIS download failed for {date}: {exc}")
        if not out_file.exists():
            raise
    return out_file


def image_to_array(image_path: Path) -> np.ndarray:
    """Load an image file into a numpy RGB array."""

    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"))


def get_lat_lon(ds: xr.Dataset) -> tuple[str, str]:
    """Return the names of latitude and longitude dimensions/coords."""

    for lat_name in ("lat", "latitude"):
        if lat_name in ds.coords:
            break
    else:
        lat_name = next((dim for dim in ds.dims if dim.lower().startswith("lat")), None)

    for lon_name in ("lon", "longitude"):
        if lon_name in ds.coords:
            break
    else:
        lon_name = next((dim for dim in ds.dims if dim.lower().startswith("lon")), None)

    if lat_name is None or lon_name is None:
        raise KeyError("Could not identify lat/lon coordinates in dataset")

    return lat_name, lon_name


def downsample_field(field: xr.DataArray) -> xr.DataArray:
    """Reduce the grid size to keep the HTML lighter."""

    if TARGET_PIXELS is None or TARGET_PIXELS <= 0:
        return field

    lat_name, lon_name = get_lat_lon(field.to_dataset(name="tmp"))
    size_lat = field.sizes[lat_name]
    size_lon = field.sizes[lon_name]

    factor_lat = max(1, int(np.ceil(size_lat / TARGET_PIXELS)))
    factor_lon = max(1, int(np.ceil(size_lon / TARGET_PIXELS)))

    if factor_lat == 1 and factor_lon == 1:
        return field

    return field.coarsen({lat_name: factor_lat, lon_name: factor_lon}, boundary="trim").mean()


def extract_field(ds: xr.Dataset, var: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lon, lat, and field arrays ready for plotting."""

    if var not in ds:
        return None, None, None  # type: ignore[misc]

    arr = ds[var].squeeze()
    arr = downsample_field(arr)
    lat_name, lon_name = get_lat_lon(arr.to_dataset(name="tmp"))
    lat_vals = arr[lat_name].values
    lon_vals = arr[lon_name].values
    return lon_vals, lat_vals, arr.values


def compute_color_limits(files: list[Path]) -> dict[str, tuple[float, float]]:
    limits = {"sst": (None, None), "sst_gradient": (None, None)}  # type: ignore[assignment]

    for path in files:
        ds = xr.open_dataset(path)
        for var in ("sst", "sst_gradient"):
            if var not in ds:
                continue
            _, _, values = extract_field(ds, var)
            if values is None:
                continue
            valid = values[np.isfinite(values)]
            if valid.size == 0:
                continue
            vmin, vmax = float(valid.min()), float(valid.max())
            current = limits[var]
            limits[var] = (
                vmin if current[0] is None or vmin < current[0] else current[0],
                vmax if current[1] is None or vmax > current[1] else current[1],
            )
    return limits


def build_slider() -> None:
    files = sorted(PROC.glob("*_proc.nc"))
    if not files:
        raise FileNotFoundError("No processed files available in data/processed/")

    color_limits = compute_color_limits(files)

    # Prepare initial traces using the first file
    first = files[0]
    first_ds = xr.open_dataset(first)
    first_lon, first_lat, first_sst = extract_field(first_ds, "sst")
    first_lon_g, first_lat_g, first_grad = extract_field(first_ds, "sst_gradient")

    date_str = first.name.split("JPL")[0][:8]
    date_iso = dt.datetime.strptime(date_str, "%Y%m%d").date().isoformat()
    modis_path = OUT / f"MODIS_truecolor_{date_iso}.jpg"
    if not modis_path.exists():
        download_modis_truecolor(date_iso, modis_path)
    first_img = image_to_array(modis_path)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("MODIS True Color", "Scientific Layer"), horizontal_spacing=0.06)

    fig.add_trace(go.Image(z=first_img, colormodel="rgb"), row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            z=first_sst,
            x=first_lon,
            y=first_lat,
            colorscale="Turbo",
            zmin=color_limits["sst"][0],
            zmax=color_limits["sst"][1],
            colorbar=dict(title="degC"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=first_grad,
            x=first_lon_g,
            y=first_lat_g,
            colorscale="Viridis",
            zmin=color_limits["sst_gradient"][0],
            zmax=color_limits["sst_gradient"][1],
            colorbar=dict(title="gradT"),
            visible=False,
        ),
        row=1,
        col=2,
    )

    frames = []
    for path in files:
        date_str = path.name.split("JPL")[0][:8]
        date_iso = dt.datetime.strptime(date_str, "%Y%m%d").date().isoformat()

        modis_path = OUT / f"MODIS_truecolor_{date_iso}.jpg"
        if not modis_path.exists():
            print(f"Downloading MODIS True Color for {date_iso}...")
            download_modis_truecolor(date_iso, modis_path)
        modis_img = image_to_array(modis_path)

        ds = xr.open_dataset(path)
        lon_vals, lat_vals, sst_vals = extract_field(ds, "sst")
        lon_vals_g, lat_vals_g, grad_vals = extract_field(ds, "sst_gradient")

        frames.append(
            go.Frame(
                data=[
                    go.Image(z=modis_img, colormodel="rgb"),
                    go.Heatmap(
                        z=sst_vals,
                        x=lon_vals,
                        y=lat_vals,
                        colorscale="Turbo",
                        zmin=color_limits["sst"][0],
                        zmax=color_limits["sst"][1],
                        colorbar=dict(title="degC"),
                    ),
                    go.Heatmap(
                        z=grad_vals,
                        x=lon_vals_g,
                        y=lat_vals_g,
                        colorscale="Viridis",
                        zmin=color_limits["sst_gradient"][0],
                        zmax=color_limits["sst_gradient"][1],
                        colorbar=dict(title="gradT"),
                        visible=False,
                    ),
                ],
                name=date_iso,
            )
        )

    fig.frames = frames

    fig.update_layout(
        title="MODIS vs Scientific Layers",
        sliders=[
            dict(
                steps=[
                    dict(
                        args=[[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        label=frame.name,
                        method="animate",
                    )
                    for frame in frames
                ],
                active=0,
                x=0.1,
                y=-0.05,
                xanchor="left",
                yanchor="top",
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.55,
                y=1.15,
                buttons=[
                    dict(label="SST", method="update", args=[{"visible": [True, True, False]}]),
                    dict(label="Gradient", method="update", args=[{"visible": [True, False, True]}]),
                ],
            )
        ],
    )

    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None)

    out_file = OUT / "compare_side_by_side_slider.html"
    fig.write_html(out_file, auto_open=False)
    print(f"Saved interactive comparison to {out_file}")


def main() -> None:
    build_slider()


if __name__ == "__main__":
    main()
