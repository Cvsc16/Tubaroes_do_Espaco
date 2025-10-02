#!/usr/bin/env python3
"""Visualização de Chlor_a (MODIS) com True Color + Mapa Científico."""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Caminhos
ROOT = Path(__file__).resolve().parents[3] 
PROC_DIR = ROOT / "data" / "processed"
IMG_DIR = ROOT / "data" / "viz"
TRUECOLOR_FILE = ROOT / "data" / "compare" / "MODIS_truecolor_2025-09-26.jpg"
IMG_DIR.mkdir(parents=True, exist_ok=True)

def plot_chlor_a(nc_file: Path, truecolor_file: Path | None = None):
    ds = xr.open_dataset(nc_file)

    if "chlor_a" not in ds:
        raise ValueError(f"{nc_file} não contém chlor_a!")

    chl = ds["chlor_a"]

    # Detectar coordenadas
    lat_name = "lat" if "lat" in chl.dims else "latitude"
    lon_name = "lon" if "lon" in chl.dims else "longitude"

    lats = ds[lat_name].values
    lons = ds[lon_name].values
    chl_vals = chl.values

    # Máscara de valores inválidos
    chl_vals = np.where((chl_vals <= 0) | np.isnan(chl_vals), np.nan, chl_vals)

    # Nome/data
    date = nc_file.stem.split(".")[1] if "AQUA_MODIS" in nc_file.name else "desconhecido"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Painel 1: True Color (se fornecido)
    ax1 = axes[0]
    if truecolor_file and Path(truecolor_file).exists():
        import matplotlib.image as mpimg
        img = mpimg.imread(truecolor_file)

        # calcular limites geográficos do dataset (mesmo do chlor_a)
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())
        extent = [lon_min, lon_max, lat_min, lat_max]

        ax1.imshow(img, origin="upper", extent=extent, transform=ccrs.PlateCarree())
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax1.set_title(f"MODIS True Color {date}")
    else:
        ax1.text(0.5, 0.5, "True Color não disponível",
                 ha="center", va="center", fontsize=12)
        ax1.set_title("True Color")

    # Painel 2: Chlor_a científico
    ax2 = axes[1]
    cmap = plt.cm.plasma.copy()
    cmap.set_bad("white")

    mesh = ax2.pcolormesh(lons, lats, chl_vals,
                          cmap=cmap, shading="auto",
                          norm=LogNorm(vmin=0.05, vmax=10))

    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax2.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")

    cbar = plt.colorbar(mesh, ax=ax2, orientation="vertical", shrink=0.8)
    cbar.set_label("Chlor_a (mg m⁻³, escala log)")

    ax2.set_title(f"Chlor_a MODIS {date}")

    out_path = IMG_DIR / f"chlor_a_map_{date}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Mapa salvo em {out_path}")

if __name__ == "__main__":
    # Exemplo: usar um arquivo chlor_a processado
    print("PROC_DIR ->", PROC_DIR.resolve())
    nc_file = next(PROC_DIR.glob("**/*chlor_a*proc.nc"))
    truecolor_file = TRUECOLOR_FILE
    plot_chlor_a(nc_file, truecolor_file)
