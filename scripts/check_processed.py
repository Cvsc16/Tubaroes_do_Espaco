#!/usr/bin/env python3
# Visualização rápida dos dados processados (SST + gradiente)

import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

def visualize_processed(file_path):
    print(f"📂 Abrindo {file_path}")
    ds = xr.open_dataset(file_path)

    # Plot SST
    plt.figure(figsize=(8, 5))
    ds["sst"].plot(cmap="turbo")  # "turbo" dá cores boas para temperatura
    plt.title("🌡️ Temperatura da Superfície do Mar (SST)")
    plt.savefig(ROOT / "data" / "sst_preview.png", dpi=150)
    plt.close()

    # Plot gradiente
    if "sst_gradient" in ds:
        plt.figure(figsize=(8, 5))
        ds["sst_gradient"].plot(cmap="inferno")
        plt.title("🌊 Gradiente de SST (Frentes Oceânicas)")
        plt.savefig(ROOT / "data" / "sst_gradient_preview.png", dpi=150)
        plt.close()

    print("✅ Pré-visualizações salvas em:")
    print(f" - {ROOT/'data'/'sst_preview.png'}")
    print(f" - {ROOT/'data'/'sst_gradient_preview.png'}")

if __name__ == "__main__":
    # Pega o primeiro arquivo processado
    files = sorted(PROC.glob("*.nc"))
    if not files:
        print("Nenhum arquivo encontrado em data/processed/")
    else:
        visualize_processed(files[0])
