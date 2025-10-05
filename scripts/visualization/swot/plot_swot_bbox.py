#!/usr/bin/env python3
"""Plota cobertura SWOT vs BBOX definido em config.yaml, com mapa de fundo."""

import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# Bootstrap simples
_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent
if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))

from scripts.utils import get_bbox, project_root, load_config

ROOT = project_root()
RAW_DIR = ROOT / "data" / "raw" / "swot"
OUT_DIR = ROOT / "data" / "viz" / "swot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config
CFG = load_config()
BBOX = get_bbox(CFG) or [-80, 25, -60, 40]  # west, south, east, north


def normalize_lon(lon):
    """Converte [0,360] → [-180,180] se necessário."""
    return (lon + 180) % 360 - 180


def plot_swot_file(file_path: Path):
    print(f"[plot] {file_path.name}")

    # Cria figura com projeção geográfica
    fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Pega bbox e expande um pouco para contexto
    west, south, east, north = BBOX
    margin_lon = (east - west) * 0.3
    margin_lat = (north - south) * 0.3
    ax.set_extent([west - margin_lon, east + margin_lon,
                   south - margin_lat, north + margin_lat],
                  crs=ccrs.PlateCarree())

    # Fundo do mapa
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    # BBOX
    ax.plot([west, east, east, west, west],
            [south, south, north, north, south],
            "r--", lw=2, label="BBOX", transform=ccrs.PlateCarree())

    # SWOT ground tracks
    for side, color in zip(["left", "right"], ["dodgerblue", "orange"]):
        try:
            ds = xr.open_dataset(file_path, group=side)
            lat = ds["latitude"].values
            lon = normalize_lon(ds["longitude"].values)
            ax.scatter(lon, lat, s=2, c=color, label=f"{side}", alpha=0.6,
                       transform=ccrs.PlateCarree())
        except Exception as e:
            print(f"  [warn] {side} não disponível: {e}")

    ax.set_title(f"SWOT Coverage — {file_path.stem}", fontsize=12)
    ax.legend(loc="upper right", markerscale=5)

    # Salvar
    out_file = OUT_DIR / f"{file_path.stem}.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Figura salva em {out_file}")

def main():
    files = sorted(RAW_DIR.glob("*.nc"))
    if not files:
        print("Nenhum arquivo SWOT encontrado em data/raw/swot/")
        return
    for f in files:
        plot_swot_file(f)


if __name__ == "__main__":
    main()
