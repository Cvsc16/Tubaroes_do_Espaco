#!/usr/bin/env python3
"""Visualiza mapa de SST (Sea Surface Temperature) a partir de CSV."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === CONFIGURAÇÕES ===
CSV_FILE = Path("data/features/20250926_sst.csv")  # ajuste para o dia desejado
OUT_PNG = CSV_FILE.with_suffix(".png")

def main():
    print(f"Lendo {CSV_FILE} ...")
    df = pd.read_csv(CSV_FILE)

    # Remove valores nulos
    df = df.dropna(subset=["sst"])
    if df.empty:
        print("Nenhum dado válido de SST encontrado!")
        return

    # Gridizar dados (lat, lon) → matriz para plot 2D
    lats = np.sort(df["lat"].unique())
    lons = np.sort(df["lon"].unique())
    grid = df.pivot(index="lat", columns="lon", values="sst")

    # Calcule os limites dinamicamente
    vmin_data = df["sst"].min()
    vmax_data = df["sst"].max()

    # Para dar uma margem, use um pequeno delta (opcional)
    delta = 1.0 
    vmin = vmin_data - delta
    vmax = vmax_data + delta 
    # Opcionalmente, pode-se usar percentis para ignorar valores extremos (outliers)

    # === PLOT ===
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.jet  # mesma paleta tradicional para SST
    mesh = plt.pcolormesh(lons, lats, grid.values, cmap=cmap, shading="auto",
                          vmin=vmin, vmax=vmax)  # escala fixa em °C (oceano global)

    plt.colorbar(mesh, label="SST (°C)")
    plt.title(f"Temperatura da Superfície do Mar (SST) - {df['date'].iloc[0]}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.savefig(OUT_PNG, dpi=200)
    plt.close()
    print(f"✅ Figura salva em {OUT_PNG}")

if __name__ == "__main__":
    main()
