#!/usr/bin/env python3
"""Visualiza mapa do gradiente de SSH (derivado do SWOT) com referência cartográfica."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- bootstrap projeto ---
_THIS_FILE = Path(__file__).resolve()
for parent in _THIS_FILE.parents:
    if parent.name == "scripts":
        PROJECT_ROOT = parent.parent
        break
else:
    PROJECT_ROOT = _THIS_FILE.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import load_config, get_bbox

# === CONFIGURAÇÕES ===
FEATURES_DIR = Path(PROJECT_ROOT, "data/features")
GRAD_COLUMN = "ssh_swot_gradient"
MASK_COLUMN = "swot_mask"
CFG = load_config()
DEFAULT_BBOX = get_bbox(CFG) or [-80.0, 25.0, -60.0, 40.0]


def _compute_extent(lons: np.ndarray, lats: np.ndarray, pad: float = 0.3) -> list[float]:
    lon_min = float(np.nanmin(lons)) if lons.size else DEFAULT_BBOX[0]
    lon_max = float(np.nanmax(lons)) if lons.size else DEFAULT_BBOX[2]
    lat_min = float(np.nanmin(lats)) if lats.size else DEFAULT_BBOX[1]
    lat_max = float(np.nanmax(lats)) if lats.size else DEFAULT_BBOX[3]

    lon_min = min(lon_min, DEFAULT_BBOX[0])
    lon_max = max(lon_max, DEFAULT_BBOX[2])
    lat_min = min(lat_min, DEFAULT_BBOX[1])
    lat_max = max(lat_max, DEFAULT_BBOX[3])

    return [lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad]


def _plot_bbox(ax):
    west, south, east, north = DEFAULT_BBOX
    xs = [west, east, east, west, west]
    ys = [south, south, north, north, south]
    (line,) = ax.plot(xs, ys, color="red", linestyle="--", linewidth=1.0, transform=ccrs.PlateCarree(), label="BBOX")
    return line


def plot_file(csv_file: Path):
    print(f"Lendo {csv_file} ...")
    df = pd.read_csv(csv_file)

    if GRAD_COLUMN not in df.columns:
        print(f"❌ Coluna {GRAD_COLUMN} não encontrada em {csv_file.name}")
        return

    df = df.dropna(subset=[GRAD_COLUMN])
    if df.empty:
        print(f"Nenhum dado válido em {GRAD_COLUMN} encontrado em {csv_file.name}")
        return

    grid = df.pivot(index="lat", columns="lon", values=GRAD_COLUMN)
    lats = grid.index.values
    lons = grid.columns.values
    data = np.ma.masked_invalid(grid.values)

    extent = _compute_extent(lons, lats)

    fig = plt.figure(figsize=(8.5, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    cmap = plt.cm.inferno
    mesh = ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("SSH Gradient (m/km)")

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, linestyle=":")

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4, alpha=0.6, color="gray")
    gl.top_labels = False
    gl.right_labels = False

    handles = []
    labels = []
    if MASK_COLUMN in df.columns:
        df_swot = df[df[MASK_COLUMN] == 1]
        if not df_swot.empty:
            scatter = ax.scatter(
                df_swot["lon"],
                df_swot["lat"],
                c="white",
                edgecolors="black",
                linewidths=0.2,
                s=5,
                alpha=0.6,
                transform=ccrs.PlateCarree(),
                label="Faixa SWOT real",
            )
            handles.append(scatter)
            labels.append("Faixa SWOT real")

    bbox_line = _plot_bbox(ax)
    handles.append(bbox_line)
    labels.append("BBOX")

    ax.legend(handles, labels, loc="lower right")
    ax.set_title(f"Gradiente SSH SWOT - {df['date'].iloc[0]}")

    out_png = csv_file.with_name(csv_file.stem + "_SSHgradient.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figura salva em {out_png}")


def main():
    csv_files = sorted(FEATURES_DIR.glob("*_features.csv"))
    if not csv_files:
        print("❌ Nenhum CSV encontrado em data/features/")
        return

    for csv_file in csv_files:
        plot_file(csv_file)


if __name__ == "__main__":
    main()
