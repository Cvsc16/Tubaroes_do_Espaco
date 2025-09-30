#!/usr/bin/env python3
"""Dashboard interativo (MODIS, SST, gradiente, probabilidade) em HTML."""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import xarray as xr
import rioxarray as rxr
from PIL import Image

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils import get_bbox, load_config, project_root

ROOT = project_root()
PROC_DIR = ROOT / "data" / "processed"
TILES_DIR = ROOT / "data" / "tiles"
OUT_DIR = ROOT / "data" / "compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CFG = load_config()
BBOX = get_bbox(CFG) or [-80.0, 25.0, -60.0, 40.0]

MODIS_SIZE = 512  # reduz peso do HTML


def download_modis(date_iso: str, out_file: Path, size: int = MODIS_SIZE) -> Path:
    base_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": "MODIS_Aqua_CorrectedReflectance_TrueColor",
        "STYLES": "",
        "FORMAT": "image/jpeg",
        "BBOX": ",".join(map(str, BBOX)),
        "WIDTH": size,
        "HEIGHT": size,
        "SRS": "EPSG:4326",
        "TIME": date_iso,
    }
    if out_file.exists():
        return out_file
    response = requests.get(base_url, params=params, timeout=60)
    response.raise_for_status()
    Image.open(io.BytesIO(response.content)).convert("RGB").save(out_file)
    return out_file


def infer_timestamp(ds: xr.Dataset, nc_path: Path) -> str:
    if "time" in ds and ds["time"].size:
        time_val = np.datetime_as_string(ds["time"].values[0], unit="s")
    else:
        raw = nc_path.name.split("JPL")[0][:8]
        time_val = f"{dt.datetime.strptime(raw, '%Y%m%d').date().isoformat()}T00:00:00"
    return time_val.replace(":", "-")


def load_day(nc_path: Path) -> dict[str, np.ndarray]:
    with xr.open_dataset(nc_path) as ds:
        sst = ds["sst"].squeeze()
        grad = ds["sst_gradient"].squeeze()
        lon = sst["lon"].values
        lat = sst["lat"].values
        timestamp = infer_timestamp(ds, nc_path)

    date_iso = timestamp.split("T")[0]
    modis_path = OUT_DIR / f"MODIS_truecolor_{date_iso}.jpg"
    download_modis(date_iso, modis_path)
    modis_img = np.array(Image.open(modis_path).convert("RGB"))

    tile_path = TILES_DIR / f"hotspots_probability_{timestamp}.tif"
    if not tile_path.exists():
        raise FileNotFoundError(f"GeoTIFF {tile_path.name} nao encontrado. Rode 05_export_tiles.py.")

    prob_da = rxr.open_rasterio(tile_path).squeeze()
    if prob_da.coords["y"][0] > prob_da.coords["y"][-1]:
        prob_da = prob_da.sortby("y")
    if prob_da.coords["x"][0] > prob_da.coords["x"][-1]:
        prob_da = prob_da.sortby("x")
    prob_interp = prob_da.interp(x=sst["lon"], y=sst["lat"], method="nearest")

    return {
        "label": date_iso,
        "lon": lon,
        "lat": lat,
        "sst": sst.values,
        "grad": grad.values,
        "prob": prob_interp.values,
        "modis": modis_img,
    }


def build_dashboard(datasets: list[dict[str, np.ndarray]], out_path: Path) -> None:
    first = datasets[0]

    def calc_limits(key: str, default_range: tuple[float, float]) -> tuple[float, float]:
        values = np.concatenate([
            data[key][np.isfinite(data[key])] for data in datasets if np.isfinite(data[key]).any()
        ])
        if values.size == 0:
            return default_range
        return float(values.min()), float(values.max())

    sst_min, sst_max = calc_limits("sst", (0.0, 1.0))
    grad_min, grad_max = calc_limits("grad", (0.0, 1.0))

    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=("MODIS", "SST", "Gradiente", "Probabilidade"),
        column_widths=[0.25, 0.28, 0.28, 0.28],
    )

    fig.add_trace(go.Image(z=first["modis"], colormodel="rgb"), row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            x=first["lon"],
            y=first["lat"],
            z=first["sst"],
            coloraxis="coloraxis",
            hovertemplate="Lat=%{y:.2f}<br>Lon=%{x:.2f}<br>SST=%{z:.2f} degC<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            x=first["lon"],
            y=first["lat"],
            z=first["grad"],
            coloraxis="coloraxis2",
            hovertemplate="Lat=%{y:.2f}<br>Lon=%{x:.2f}<br>|dT|=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Heatmap(
            x=first["lon"],
            y=first["lat"],
            z=first["prob"],
            coloraxis="coloraxis3",
            hovertemplate="Lat=%{y:.2f}<br>Lon=%{x:.2f}<br>Prob=%{z:.2f}<extra></extra>",
        ),
        row=1,
        col=4,
    )

    if len(datasets) > 1:
        frames = []
        for data in datasets:
            frames.append(
                go.Frame(
                    data=[
                        go.Image(z=data["modis"], colormodel="rgb"),
                        go.Heatmap(z=data["sst"], x=data["lon"], y=data["lat"], coloraxis="coloraxis"),
                        go.Heatmap(z=data["grad"], x=data["lon"], y=data["lat"], coloraxis="coloraxis2"),
                        go.Heatmap(z=data["prob"], x=data["lon"], y=data["lat"], coloraxis="coloraxis3"),
                    ],
                    name=data["label"],
                )
            )
        fig.frames = frames

        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "Data: "},
                    "pad": {"t": 60},
                    "steps": [
                        {
                            "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                            "label": frame.name,
                            "method": "animate",
                        }
                        for frame in frames
                    ],
                }
            ],
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0.5,
                    "y": 1.15,
                    "direction": "left",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False}}],
                        },
                    ],
                }
            ],
        )

    fig.update_layout(
        coloraxis=dict(colorscale="Turbo", cmin=sst_min, cmax=sst_max, colorbar=dict(title="SST (degC)", x=0.43, len=0.7)),
        coloraxis2=dict(colorscale="Inferno", cmin=grad_min, cmax=grad_max, colorbar=dict(title="|dT|", x=0.69, len=0.7)),
        coloraxis3=dict(colorscale="Viridis", cmin=0, cmax=1, colorbar=dict(title="Probabilidade", x=0.95, len=0.7)),
        height=600,
        width=1500,
        title="Dashboard Interativo - MODIS, SST, Gradiente, Probabilidade",
        margin=dict(l=50, r=50, t=80, b=40),
    )

    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None)

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Dashboard salvo em {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Gera dashboard interativo a partir dos GeoTIFFs gerados")
    parser.add_argument("--date", help="Data alvo (YYYY-MM-DD). Default: processa todos os dias disponiveis.")
    args = parser.parse_args()

    nc_files = sorted(PROC_DIR.glob("*_proc.nc"))
    if not nc_files:
        raise FileNotFoundError("Nenhum arquivo processado em data/processed. Rode 02_preprocess.py.")

    datasets: list[dict[str, np.ndarray]] = []

    for nc_file in nc_files:
        try:
            data = load_day(nc_file)
        except FileNotFoundError as exc:
            print(exc)
            continue

        if args.date and data["label"] != args.date:
            continue

        datasets.append(data)

        if args.date:
            break

    if not datasets:
        msg = "Nenhum dataset encontrado"
        if args.date:
            msg += f" para a data {args.date}"
        raise RuntimeError(msg + ".")

    suffix = f"_{datasets[0]['label']}" if len(datasets) == 1 else ""
    out_html = OUT_DIR / f"compare_modis_sst_probability_interativo{suffix}.html"
    build_dashboard(datasets, out_html)


if __name__ == "__main__":
    main()
