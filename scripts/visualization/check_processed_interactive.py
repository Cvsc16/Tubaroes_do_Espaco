#!/usr/bin/env python3
# Visualização interativa dos dados processados (SST + gradiente) com Plotly

import xarray as xr
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT  = ROOT / "data"

def visualize_interactive(file_path):
    print(f"📂 Abrindo {file_path}")
    ds = xr.open_dataset(file_path)

    # SST
    fig_sst = px.imshow(
        ds["sst"].values,
        origin="lower",
        labels=dict(color="SST (°C)"),
        x=ds["lon"].values,
        y=ds["lat"].values,
        color_continuous_scale="Turbo",
        title="🌡️ Temperatura da Superfície do Mar (SST)"
    )
    fig_sst.update_xaxes(title="Longitude")
    fig_sst.update_yaxes(title="Latitude")
    sst_file = OUT / "sst_interactive.html"
    fig_sst.write_html(sst_file)

    # Gradiente
    if "sst_gradient" in ds:
        fig_grad = px.imshow(
            ds["sst_gradient"].values,
            origin="lower",
            labels=dict(color="Gradiente SST"),
            x=ds["lon"].values,
            y=ds["lat"].values,
            color_continuous_scale="Inferno",
            title="🌊 Gradiente de SST (Frentes Oceânicas)"
        )
        fig_grad.update_xaxes(title="Longitude")
        fig_grad.update_yaxes(title="Latitude")
        grad_file = OUT / "sst_gradient_interactive.html"
        fig_grad.write_html(grad_file)

    print("✅ Versões interativas salvas em:")
    print(f" - {sst_file}")
    if "sst_gradient" in ds:
        print(f" - {grad_file}")

if __name__ == "__main__":
    files = sorted(PROC.glob("*.nc"))
    if not files:
        print("Nenhum arquivo encontrado em data/processed/")
    else:
        visualize_interactive(files[0])
