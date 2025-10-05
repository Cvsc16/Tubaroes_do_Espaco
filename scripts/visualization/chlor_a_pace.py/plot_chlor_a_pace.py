#!/usr/bin/env python3
"""Visualiza mapa de clorofila (chlor_a) a partir de CSV (PACE OCI)."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === CONFIGURAÇÕES ===
CSV_FILE = Path("data/features/20250926_features.csv")  # ajuste para o dia desejado
OUT_PNG = CSV_FILE.with_name(CSV_FILE.stem + "_PACE.png")

CHL_COLUMN = "chlor_a_pace"  # campo de clorofila do PACE

def main():
    print(f"Lendo {CSV_FILE} ...")
    df = pd.read_csv(CSV_FILE)

    # Garante que a coluna existe
    if CHL_COLUMN not in df.columns:
        print(f"❌ Coluna {CHL_COLUMN} não encontrada no CSV!")
        print(f"Colunas disponíveis: {list(df.columns)}")
        return

    # Remove valores nulos
    df = df.dropna(subset=[CHL_COLUMN])
    if df.empty:
        print(f"Nenhum dado válido em {CHL_COLUMN} encontrado!")
        return

    # Gridizar dados (lat, lon) → matriz para plot 2D
    lats = np.sort(df["lat"].unique())
    lons = np.sort(df["lon"].unique())
    grid = df.pivot(index="lat", columns="lon", values=CHL_COLUMN)

    # === PLOT ===
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.plasma  # pode trocar para viridis
    mesh = plt.pcolormesh(lons, lats, grid.values, cmap=cmap, shading="auto")

    plt.colorbar(mesh, label="Clorofila (PACE, mg m⁻³)")
    plt.title(f"Clorofila PACE OCI - {df['date'].iloc[0]}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.savefig(OUT_PNG, dpi=200)
    plt.close()
    print(f"✅ Figura salva em {OUT_PNG}")

if __name__ == "__main__":
    main()
