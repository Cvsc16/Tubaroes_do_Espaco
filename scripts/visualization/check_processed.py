#!/usr/bin/env python3
"""Visualizacao rapida dos dados processados (SST + gradiente)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils import project_root


ROOT = project_root()
PROC = ROOT / "data" / "processed"


def visualize_processed(file_path: Path) -> None:
    print(f"[viz] Abrindo {file_path}")
    ds = xr.open_dataset(file_path)

    plt.figure(figsize=(8, 5))
    ds["sst"].plot(cmap="turbo")
    plt.title("Temperatura da Superficie do Mar (SST)")
    plt.savefig(ROOT / "data" / "sst_preview.png", dpi=150)
    plt.close()

    if "sst_gradient" in ds:
        plt.figure(figsize=(8, 5))
        ds["sst_gradient"].plot(cmap="inferno")
        plt.title("Gradiente de SST (frentes)")
        plt.savefig(ROOT / "data" / "sst_gradient_preview.png", dpi=150)
        plt.close()

    print("[viz] Arquivos salvos:")
    print(f" - {ROOT / 'data' / 'sst_preview.png'}")
    if "sst_gradient" in ds:
        print(f" - {ROOT / 'data' / 'sst_gradient_preview.png'}")


if __name__ == "__main__":
    files = sorted(PROC.glob("*.nc"))
    if not files:
        print("Nenhum arquivo encontrado em data/processed/")
    else:
        visualize_processed(files[0])
