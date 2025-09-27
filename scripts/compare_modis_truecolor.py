#!/usr/bin/env python3
"""
Comparação MODIS True Color vs SST MUR
"""

import datetime
from pathlib import Path
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import yaml
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "compare"
OUT.mkdir(parents=True, exist_ok=True)

CFG = yaml.safe_load(open(ROOT/"config"/"config.yaml"))
bbox = CFG["aoi"]["bbox"]  # [west, south, east, north]

def download_modis_truecolor(date: str, out_file: Path):
    """
    Baixa imagem MODIS Aqua True Color via Worldview WMS
    """
    base_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    layer = "MODIS_Aqua_CorrectedReflectance_TrueColor"
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/jpeg",
        "BBOX": ",".join(map(str, bbox)),
        "WIDTH": 1024,
        "HEIGHT": 1024,
        "SRS": "EPSG:4326",
        "TIME": date,
    }

    r = requests.get(base_url, params=params, timeout=60)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    img.save(out_file)
    return out_file

def compare_day(nc_file: Path):
    """
    Pega SST MUR de um arquivo processado e plota lado a lado com MODIS True Color
    """
    # Data vem do nome do arquivo
    date_str = nc_file.name.split("JPL")[0][:8]
    date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
    date_iso = date.isoformat()

    # Baixa MODIS
    modis_path = OUT / f"MODIS_truecolor_{date_iso}.jpg"
    if not modis_path.exists():
        print(f"📥 Baixando MODIS True Color para {date_iso}...")
        download_modis_truecolor(date_iso, modis_path)

    # Carrega SST
    ds = xr.open_dataset(nc_file)
    sst = ds["sst"].squeeze()

    # Plot lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(Image.open(modis_path))
    axs[0].set_title(f"MODIS True Color {date_iso}")
    axs[0].axis("off")

    im = axs[1].pcolormesh(sst["lon"], sst["lat"], sst, cmap="turbo")
    axs[1].set_title(f"SST MUR {date_iso}")
    fig.colorbar(im, ax=axs[1], label="°C")

    out_img = OUT / f"compare_MODIS_SST_{date_iso}.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"✅ Comparação salva em {out_img}")

if __name__ == "__main__":
    # Pega o primeiro arquivo processado
    files = sorted(PROC.glob("*_proc.nc"))
    if not files:
        raise FileNotFoundError("Nenhum arquivo processado em data/processed/")
    compare_day(files[0])
