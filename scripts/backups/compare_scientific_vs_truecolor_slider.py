#!/usr/bin/env python3
"""
Comparação científica (SST MUR) vs MODIS True Color com slider temporal
"""

import datetime
from pathlib import Path
import requests
from PIL import Image
import io
import numpy as np
import xarray as xr
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "compare"
OUT.mkdir(parents=True, exist_ok=True)

# Área de interesse (vai do config.yaml idealmente)
bbox = [-80.0, 25.0, -60.0, 40.0]  # [west, south, east, north]

def download_modis_truecolor(date: str, out_file: Path):
    """
    Baixa imagem MODIS Aqua True Color via Worldview WMS
    """
    base_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    layer = "MODIS_Aqua_CorrectedReflectance_TrueColor"
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/jpeg",
        "BBOX": ",".join(map(str, bbox)),
        "WIDTH": 512,
        "HEIGHT": 512,
        "SRS": "EPSG:4326",
        "TIME": date,
    }

    r = requests.get(base_url, params=params, timeout=60)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    img.save(out_file)
    return out_file

def build_slider():
    files = sorted(PROC.glob("*_proc.nc"))
    if not files:
        raise FileNotFoundError("Nenhum arquivo processado em data/processed/")

    steps = []
    frames = []

    for idx, nc_file in enumerate(files):
        # Data do nome do arquivo
        date_str = nc_file.name.split("JPL")[0][:8]
        date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
        date_iso = date.isoformat()

        # MODIS True Color
        modis_path = OUT / f"MODIS_truecolor_{date_iso}.jpg"
        if not modis_path.exists():
            print(f"📥 Baixando MODIS True Color para {date_iso}...")
            download_modis_truecolor(date_iso, modis_path)

        # Carrega SST
        ds = xr.open_dataset(nc_file)
        sst = ds["sst"].squeeze().values
        lat = ds["lat"].values
        lon = ds["lon"].values

        # Limpa valores inválidos
        valid_values = sst[~np.isnan(sst)]
        if valid_values.size > 0:
            zmin = float(valid_values.min())
            zmax = float(valid_values.max())
        else:
            zmin, zmax = 20.0, 35.0  # fallback típico oceano

        # Frame para slider
        frames.append(
    go.Frame(
        data=[
            go.Heatmap(
                z=sst,
                x=lon,
                y=lat,
                colorscale="Turbo",
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title="°C"),
            )
        ],
        layout=go.Layout(
            images=[
                dict(
                    source=Image.open(modis_path),
                    xref="x",
                    yref="y",
                    x=bbox[0],
                    y=bbox[3],
                    sizex=bbox[2] - bbox[0],
                    sizey=bbox[3] - bbox[1],
                    sizing="stretch",
                    opacity=0.6,
                    layer="below",
                )
            ]
        ),
        name=date_iso,
    )
)

        # Passo do slider
        steps.append(
            dict(
                method="animate",
                args=[[date_iso], {"frame": {"duration": 1000, "redraw": True}, "mode": "immediate"}],
                label=date_iso,
            )
        )

    # Figura base
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title="Comparação SST (científica) vs MODIS True Color",
            xaxis=dict(title="Longitude"),
            yaxis=dict(title="Latitude"),
            sliders=[dict(active=0, steps=steps)],
            updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True, "mode": "immediate"}])])],
        ),
        frames=frames,
    )

    out_html = OUT / "compare_scientific_vs_truecolor_slider.html"
    fig.write_html(out_html)
    print(f"✅ Slider salvo em {out_html}")

if __name__ == "__main__":
    build_slider()
