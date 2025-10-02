#!/usr/bin/env python3
"""Visualização de Chlor_a (PACE OCI) com True Color + Mapa Científico."""

import requests
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
COMPARE_DIR = ROOT / "data" / "compare"
IMG_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

# NASA Worldview Snapshots endpoint
SNAPSHOT_URL = (
    "https://wvs.earthdata.nasa.gov/api/v1/snapshot"
    "?REQUEST=GetSnapshot"
    "&TIME={date}"
    "&BBOX={bbox}"
    "&CRS=EPSG:4326"
    "&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor"
    "&FORMAT=image/jpeg"
    "&WIDTH=1024&HEIGHT=1024"
)


def download_truecolor(date: str, bbox: list[float], out_path: Path):
    """Baixa imagem True Color via NASA Worldview Snapshots."""
    url = SNAPSHOT_URL.format(
        date=date,
        bbox=",".join(map(str, bbox))
    )
    print(f"🌍 Baixando True Color MODIS para {date} ...")
    resp = requests.get(url, timeout=60)
    if resp.status_code == 200:
        out_path.write_bytes(resp.content)
        print(f"✅ True Color salva em {out_path}")
        return True
    else:
        print(f"❌ Falha ao baixar True Color ({resp.status_code})")
        return False


def plot_chlor_a_pace(nc_file: Path, truecolor_file: Path | None = None):
    ds = xr.open_dataset(nc_file)

    if "chlor_a_pace" not in ds:
        raise ValueError(f"{nc_file} não contém chlor_a_pace! Variáveis disponíveis: {list(ds.variables)}")

    chl = ds["chlor_a_pace"]

    # Detectar coordenadas
    lat_name = "lat" if "lat" in chl.dims else "latitude"
    lon_name = "lon" if "lon" in chl.dims else "longitude"

    lats = ds[lat_name].values
    lons = ds[lon_name].values
    chl_vals = chl.values

    chl_vals = np.where((chl_vals <= 0) | np.isnan(chl_vals), np.nan, chl_vals)

    # Data
    date = nc_file.name.split("_")[0]

    # Caminho da imagem True Color
    if truecolor_file is None:
        truecolor_file = COMPARE_DIR / f"PACE_truecolor_{date}.jpg"

    if not Path(truecolor_file).exists():
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())
        bbox = [lon_min, lat_min, lon_max, lat_max]
        download_truecolor(date[:4] + "-" + date[4:6] + "-" + date[6:], bbox, Path(truecolor_file))

    # === PLOT ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Painel 1: True Color
    ax1 = axes[0]
    if Path(truecolor_file).exists():
        import matplotlib.image as mpimg
        img = mpimg.imread(truecolor_file)

        extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
        ax1.imshow(img, origin="upper", extent=extent, transform=ccrs.PlateCarree())
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax1.set_title(f"PACE True Color {date}")
    else:
        ax1.text(0.5, 0.5, "True Color não disponível",
                 ha="center", va="center", fontsize=12)
        ax1.set_title("True Color")

    # Painel 2: Chlor_a
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
    cbar.set_label("Clorofila PACE (mg m⁻³, escala log)")

    ax2.set_title(f"Clorofila PACE {date}")

    out_path = IMG_DIR / f"chlor_a_map_PACE_{date}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Mapa salvo em {out_path}")


if __name__ == "__main__":
    print("PROC_DIR ->", PROC_DIR.resolve())
    nc_file = next(PROC_DIR.glob("*CHL-PACE_proc.nc"))
    plot_chlor_a_pace(nc_file)
