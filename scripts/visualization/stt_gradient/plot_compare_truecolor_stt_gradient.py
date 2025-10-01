#!/usr/bin/env python3
"""Visualização de Gradiente de SST (MUR) com True Color + Mapa Científico."""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# Caminhos
ROOT = Path(__file__).resolve().parents[3]
PROC_DIR = ROOT / "data" / "processed"
IMG_DIR = ROOT / "data" / "viz"
TRUECOLOR_FILE = ROOT / "data" / "compare" / "MODIS_truecolor_2025-09-26.jpg"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def plot_sst_gradient(nc_file: Path, truecolor_file: Path | None = None):
    ds = xr.open_dataset(nc_file)

    if "sst_gradient" not in ds:
        raise ValueError(f"{nc_file} não contém variável sst_gradient!")

    grad = ds["sst_gradient"]

    # Detectar coordenadas
    lat_name = "lat" if "lat" in grad.dims else "latitude"
    lon_name = "lon" if "lon" in grad.dims else "longitude"

    lats = ds[lat_name].values
    lons = ds[lon_name].values
    grad_vals = grad.values

    # Se tiver dimensão extra (ex: time=1), remover
    if grad_vals.ndim == 3:
        grad_vals = np.squeeze(grad_vals)

    # Máscara de valores inválidos
    grad_vals = np.where(np.isnan(grad_vals), np.nan, grad_vals)

    # Nome/data
    try:
        raw = nc_file.name.split("-")[0][:8]  # pega AAAAMMDD
        date = datetime.strptime(raw, "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        date = "desconhecido"

    # Extent geográfico do dataset
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())
    extent = [lon_min, lon_max, lat_min, lat_max]

    # Criação dos painéis
    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    ax1, ax2 = axes

    # Painel 1: True Color
    if truecolor_file and Path(truecolor_file).exists():
        import matplotlib.image as mpimg
        img = mpimg.imread(truecolor_file)
        ax1.imshow(img, origin="upper", extent=extent, transform=ccrs.PlateCarree())
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax1.set_title(f"MODIS True Color {date}")
    else:
        ax1.text(0.5, 0.5, "True Color não disponível",
                 ha="center", va="center", fontsize=12)
        ax1.set_title("True Color")

    # Painel 2: SST Gradient
    cmap = plt.cm.seismic.copy()  # azul-negativo, vermelho-positivo
    cmap.set_bad("white")

    vmax = np.nanpercentile(np.abs(grad_vals), 98)  # escala simétrica
    mesh = ax2.pcolormesh(lons, lats, grad_vals,
                          cmap=cmap, shading="auto",
                          vmin=-vmax, vmax=vmax,
                          transform=ccrs.PlateCarree())

    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax2.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")

    cbar = plt.colorbar(mesh, ax=ax2, orientation="vertical", shrink=0.8)
    cbar.set_label("Gradiente de SST (°C/km)")  # ajuste unidade conforme cálculo

    ax2.set_title(f"SST Gradient {date}")

    out_path = IMG_DIR / f"sst_gradient_map_{date}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Mapa salvo em {out_path}")


if __name__ == "__main__":
    print("PROC_DIR ->", PROC_DIR.resolve())
    nc_file = next(PROC_DIR.glob("**/*sst*proc.nc"))  # mesmo arquivo do SST
    truecolor_file = TRUECOLOR_FILE
    plot_sst_gradient(nc_file, truecolor_file)
