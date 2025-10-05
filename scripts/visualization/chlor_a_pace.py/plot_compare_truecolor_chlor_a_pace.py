#!/usr/bin/env python3
"""Visualização de Chlor_a (PACE OCI) com True Color + Mapa Científico (via GIBS WMS)."""

import io
import requests
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image

# Paths
ROOT = Path(__file__).resolve().parents[3]
PROC_DIR = ROOT / "data" / "processed"
IMG_DIR = ROOT / "data" / "viz"
COMPARE_DIR = ROOT / "data" / "compare"
IMG_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

# GIBS WMS endpoint
GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"

# Layers em ordem de prioridade
LAYER_CANDIDATES = [
    "MODIS_Aqua_CorrectedReflectance_TrueColor",
    "MODIS_Terra_CorrectedReflectance_TrueColor",
    "VIIRS_SNPP_CorrectedReflectance_TrueColor",
]


def download_truecolor(date_iso: str, bbox: list[float], out_file: Path, size: int = 1024) -> Path:
    """Baixa imagem True Color via GIBS WMS (Aqua/Terra/VIIRS)."""
    if out_file.exists():
        return out_file

    for layer in LAYER_CANDIDATES:
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.1.1",
            "LAYERS": layer,
            "STYLES": "",
            "FORMAT": "image/jpeg",
            "BBOX": ",".join(map(str, bbox)),  # [lon_min, lat_min, lon_max, lat_max]
            "WIDTH": size,
            "HEIGHT": size,
            "SRS": "EPSG:4326",
            "TIME": date_iso,
        }

        try:
            print(f"🌍 Tentando baixar True Color ({layer}) para {date_iso}")
            r = requests.get(GIBS_WMS, params=params, timeout=60)
            r.raise_for_status()

            Image.open(io.BytesIO(r.content)).save(out_file)
            print(f"✅ True Color salva em {out_file}")
            return out_file
        except Exception as e:
            print(f"⚠️  Falha em {layer}: {e}")

    print("❌ Nenhuma camada True Color disponível nesse dia")
    return out_file


def plot_chlor_a_pace(nc_file: Path, truecolor_file: Path | None = None):
    ds = xr.open_dataset(nc_file)

    # Seu pré-processado guarda como 'chlor_a_pace'
    if "chlor_a_pace" not in ds.variables:
        raise ValueError(f"{nc_file} não contém 'chlor_a_pace'. Variáveis: {list(ds.variables)}")

    chl = ds["chlor_a_pace"]

    # Detectar coords
    lat_name = "lat" if "lat" in chl.dims else "latitude"
    lon_name = "lon" if "lon" in chl.dims else "longitude"

    lats = ds[lat_name].values
    lons = ds[lon_name].values
    chl_vals = np.where((chl.values <= 0) | np.isnan(chl.values), np.nan, chl.values)

    # Data YYYYMMDD do nome
    date_token = nc_file.name.split("_")[0]
    date_iso = f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:]}"

    # Caminho para True Color
    if truecolor_file is None:
        truecolor_file = COMPARE_DIR / f"PACE_truecolor_{date_token}.jpg"

    # Se não existir → baixar via GIBS
    if not Path(truecolor_file).exists():
        lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
        lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
        bbox = [lon_min, lat_min, lon_max, lat_max]
        download_truecolor(date_iso, bbox, Path(truecolor_file))

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
        ax1.set_title(f"PACE True Color {date_token}")
    else:
        ax1.text(0.5, 0.5, "True Color não disponível",
                 ha="center", va="center", fontsize=12)

    # Painel 2: Chlor_a PACE
    ax2 = axes[1]
    cmap = plt.cm.plasma.copy()
    cmap.set_bad("white")

    mesh = ax2.pcolormesh(lons, lats, chl_vals,
                          cmap=cmap, shading="auto",
                          norm=LogNorm(vmin=0.05, vmax=10.0))

    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.right_labels = False
    gl.top_labels = False

    cbar = plt.colorbar(mesh, ax=ax2, orientation="vertical", shrink=0.8)
    cbar.set_label("Clorofila PACE (mg m⁻³, escala log)")

    ax2.set_title(f"Clorofila PACE {date_token}")

    out_path = IMG_DIR / f"chlor_a_map_PACE_{date_token}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Mapa salvo em {out_path}")


if __name__ == "__main__":
    print("PROC_DIR ->", PROC_DIR.resolve())
    nc_file = next(PROC_DIR.glob("*CHL-PACE_proc.nc"))
    plot_chlor_a_pace(nc_file)
