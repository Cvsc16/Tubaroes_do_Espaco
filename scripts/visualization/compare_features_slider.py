#!/usr/bin/env python3
"""
Comparação interativa com slider temporal
Mostra SST e Gradiente de SST ao longo dos dias
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Caminho dos arquivos
ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features"
OUT = ROOT / "data" / "compare"
OUT.mkdir(parents=True, exist_ok=True)

# Limite de pontos para não travar
MAX_POINTS = 10000

# Carregar todos os arquivos CSV de features
files = sorted(FEATURES.glob("*.csv"))
if not files:
    raise FileNotFoundError("Nenhum arquivo encontrado em data/features/")

frames = []
for f in files:
    df = pd.read_csv(f)
    # Amostragem se o arquivo for muito grande
    if len(df) > MAX_POINTS:
        df = df.sample(MAX_POINTS, random_state=42)
    df["source_file"] = f.name
    frames.append(df)

# Concatenar tudo
data = pd.concat(frames, ignore_index=True)

# Criar figura com slider
fig = go.Figure()

# Adicionar cada dia como um frame
frames_plotly = []
steps = []

for i, fname in enumerate(sorted(data["source_file"].unique())):
    df_day = data[data["source_file"] == fname]

    frame = go.Frame(
        data=[
            go.Scattermapbox(
                lon=df_day["lon"],
                lat=df_day["lat"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=df_day["sst"],
                    colorscale="Turbo",
                    colorbar=dict(title="SST (°C)"),
                ),
                text=[f"SST: {t:.2f} °C<br>Gradiente: {g:.3f}" for t, g in zip(df_day["sst"], df_day["sst_gradient"])],
            )
        ],
        name=fname
    )
    frames_plotly.append(frame)

    step = dict(
        method="animate",
        args=[
            [fname],
            dict(mode="immediate", frame=dict(duration=500, redraw=True), transition=dict(duration=0)),
        ],
        label=fname[:8],  # só a data
    )
    steps.append(step)

fig.frames = frames_plotly

# Layout inicial
first_day = sorted(data["source_file"].unique())[0]
df_init = data[data["source_file"] == first_day]

fig.add_trace(
    go.Scattermapbox(
        lon=df_init["lon"],
        lat=df_init["lat"],
        mode="markers",
        marker=dict(
            size=4,
            color=df_init["sst"],
            colorscale="Turbo",
            colorbar=dict(title="SST (°C)"),
        ),
        text=[f"SST: {t:.2f} °C<br>Gradiente: {g:.3f}" for t, g in zip(df_init["sst"], df_init["sst_gradient"])],
    )
)

fig.update_layout(
    title="🌍 Evolução da SST (Surface Sea Temperature) com Slider",
    mapbox=dict(style="carto-positron", zoom=4, center=dict(lat=32.5, lon=-70)),
    sliders=[dict(steps=steps, active=0, x=0.1, y=0, xanchor="left", yanchor="top")],
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=800, redraw=True), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))]),
            ],
        )
    ]
)

# Salvar HTML
out_file = OUT / "sst_slider.html"
fig.write_html(out_file)
print(f"✅ Slider interativo salvo em {out_file}")
