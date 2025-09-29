#!/usr/bin/env python3
"""
Mapa interativo das features (lat, lon, SST, gradiente)
"""

import pandas as pd
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
OUT  = ROOT / "data" / "viz"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    # Carrega o primeiro arquivo de features
    files = sorted(FEAT.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Nenhum arquivo encontrado em data/features/")
    file = files[0]

    print(f"📂 Carregando {file.name} ...")
    df = pd.read_csv(file)

    # Remove NaN para evitar erro no plot
    df = df.dropna(subset=["sst", "sst_gradient"])

    # Cria scattermapbox com SST
    fig = px.scatter_mapbox(
        df.sample(20000),  # pega amostra para não travar
        lat="lat",
        lon="lon",
        color="sst",
        size_max=4,
        zoom=4,
        mapbox_style="carto-positron",
        title=f"SST Interativa - {file.name}"
    )

    out_html = OUT / f"{file.stem}_sst_map.html"
    fig.write_html(out_html)
    print(f"✅ Mapa interativo salvo em {out_html}")

    # Cria scattermapbox com gradiente
    fig2 = px.scatter_mapbox(
        df.sample(20000),
        lat="lat",
        lon="lon",
        color="sst_gradient",
        size_max=4,
        zoom=4,
        mapbox_style="carto-positron",
        title=f"Gradiente Interativo - {file.name}"
    )

    out_html2 = OUT / f"{file.stem}_gradient_map.html"
    fig2.write_html(out_html2)
    print(f"✅ Mapa interativo salvo em {out_html2}")

if __name__ == "__main__":
    main()
