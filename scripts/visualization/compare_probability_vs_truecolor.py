#!/usr/bin/env python3
"""Gera comparacoes MODIS True Color vs SST vs probabilidade (GeoTIFF)."""

from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent

if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))

import sys
import argparse

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FALLBACK = THIS_DIR.parent.parent
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

import datetime as dt
import io

import matplotlib.pyplot as plt
import numpy as np
import requests
import rioxarray as rxr
import xarray as xr
from PIL import Image

from scripts.utils import get_bbox, load_config, project_root

ROOT = project_root()
PROC_DIR = ROOT / "data" / "processed"
TILES_DIR = ROOT / "data" / "tiles"
OUT_DIR = ROOT / "data" / "compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CFG = load_config()
BBOX = get_bbox(CFG) or [-80.0, 25.0, -60.0, 40.0]



def download_modis(date_iso: str, out_file: Path, size: int = 1024) -> Path:
    """Baixa imagem MODIS True Color via WMS e salva em disco."""

    base_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    layer = "MODIS_Aqua_CorrectedReflectance_TrueColor"
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/jpeg",
        "BBOX": ",".join(map(str, BBOX)),
        "WIDTH": size,
        "HEIGHT": size,
        "SRS": "EPSG:4326",
        "TIME": date_iso,
    }

    if out_file.exists():
        return out_file

    response = requests.get(base_url, params=params, timeout=60)
    response.raise_for_status()
    Image.open(io.BytesIO(response.content)).save(out_file)
    return out_file


def infer_timestamp(ds: xr.Dataset, nc_path: Path) -> str:
    """Gera timestamp ISO (com '-' no lugar de ':') para casar com o GeoTIFF."""

    if "time" in ds:
        time_val = np.datetime_as_string(ds["time"].values[0], unit="s")
    else:
        raw = nc_path.name.split("JPL")[0][:8]
        date = dt.datetime.strptime(raw, "%Y%m%d").date()
        time_val = f"{date.isoformat()}T00:00:00"
    return time_val.replace(":", "-")


def plot_comparison(nc_path: Path):
    with xr.open_dataset(nc_path) as ds:
        if "sst" not in ds:
            raise KeyError(f"Variavel 'sst' nao encontrada em {nc_path.name}")
        sst = ds["sst"].squeeze()

        lon = sst["lon"].values
        lat = sst["lat"].values

        timestamp_key = infer_timestamp(ds, nc_path)

    date_iso = timestamp_key.split("T")[0]
    modis_path = OUT_DIR / f"MODIS_truecolor_{date_iso}.jpg"
    download_modis(date_iso, modis_path)

    tile_path = TILES_DIR / f"hotspots_probability_{timestamp_key}.tif"
    if not tile_path.exists():
        raise FileNotFoundError(f"GeoTIFF {tile_path.name} nao encontrado. Gere com 05_export_tiles.py")

    tile_da = rxr.open_rasterio(tile_path).squeeze()
    if tile_da.coords['y'][0] > tile_da.coords['y'][-1]:
        tile_da = tile_da.sortby('y')
    if tile_da.coords['x'][0] > tile_da.coords['x'][-1]:
        tile_da = tile_da.sortby('x')

    proba_vals = tile_da.values
    lon_prob = tile_da['x'].values
    lat_prob = tile_da['y'].values

    grad = ds["sst_gradient"].squeeze()
    grad_vals = grad.values
    fig, axs = plt.subplots(1, 4, figsize=(22, 6))

    axs[0].imshow(Image.open(modis_path))
    axs[0].set_title(f"MODIS True Color\n{date_iso}")
    axs[0].axis("off")

    im1 = axs[1].pcolormesh(lon, lat, sst, cmap="turbo")
    axs[1].set_title("SST MUR (degC)")
    axs[1].set_xlabel("Longitude")
    axs[1].set_ylabel("Latitude")
    fig.colorbar(im1, ax=axs[1], shrink=0.75)

    im2 = axs[2].pcolormesh(lon, lat, grad_vals, cmap="inferno")
    axs[2].set_title("Gradiente SST")
    axs[2].set_xlabel("Longitude")
    axs[2].set_ylabel("Latitude")
    fig.colorbar(im2, ax=axs[2], shrink=0.75, label="abs(dT)")

    masked_proba = np.ma.masked_invalid(proba_vals)
    im3 = axs[3].pcolormesh(lon_prob, lat_prob, masked_proba, cmap="viridis", shading="auto", vmin=0, vmax=1)
    axs[3].set_title("Probabilidade de hotspot")
    axs[3].set_xlabel("Longitude")
    axs[3].set_ylabel("Latitude")
    fig.colorbar(im3, ax=axs[3], shrink=0.75, label="Probabilidade")

    fig.suptitle(f"Comparacao MODIS x Cientifico x Modelo - {date_iso}", fontsize=14)
    fig.tight_layout()

    out_png = OUT_DIR / f"compare_MODIS_SST_probability_{date_iso}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Figura salva em {out_png}")


def main(target_date: str | None = None) -> None:
    files = sorted(PROC_DIR.glob("*_proc.nc"))
    if not files:
        raise FileNotFoundError('Nenhum arquivo processado encontrado. Rode 02_preprocess.py.')

    if target_date:
        target_str = target_date.replace('-', '')
        files = [f for f in files if f.name.startswith(target_str)]
        if not files:
            raise FileNotFoundError(f'Nenhum arquivo processado para a data {target_date}.')

    for nc_file in files:
        try:
            plot_comparison(nc_file)
        except Exception as exc:
            print(f'[compare] Falha em {nc_file.name}: {exc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compara MODIS, SST e probabilidade (GeoTIFF).')
    parser.add_argument('--date', help='Data alvo no formato YYYY-MM-DD (opcional).')
    args = parser.parse_args()
    main(args.date)
