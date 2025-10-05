#!/usr/bin/env python3
"""Visualiza mapa de clorofila (chlor_a) a partir de CSV."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === CONFIGURAÇÕES ===
CSV_FILE = Path("data/features/20250926_chlor_a.csv")  # ajuste para o dia desejado
OUT_PNG = CSV_FILE.with_suffix(".png")

def main():
    print(f"Lendo {CSV_FILE} ...")
    df = pd.read_csv(CSV_FILE)

    # Remove valores nulos
    df = df.dropna(subset=["chlor_a"])
    if df.empty:
        print("Nenhum dado válido de chlor_a encontrado!")
        return

    # Gridizar dados (lat, lon) → matriz para plot 2D
    lats = np.sort(df["lat"].unique())
    lons = np.sort(df["lon"].unique())
    grid = df.pivot(index="lat", columns="lon", values="chlor_a")

    # === PLOT ===
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.plasma  # ou viridis
    mesh = plt.pcolormesh(lons, lats, grid.values, cmap=cmap, shading="auto")

    plt.colorbar(mesh, label="Chlor_a (mg m⁻³)")
    plt.title(f"Clorofila (chlor_a) - {df['date'].iloc[0]}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.savefig(OUT_PNG, dpi=200)
    plt.close()
    print(f"✅ Figura salva em {OUT_PNG}")

if __name__ == "__main__":
    main()
