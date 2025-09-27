#!/usr/bin/env python3
# Pré-processamento de dados de SST: recorte por bbox, cálculo de gradiente e exportação

import xarray as xr
import numpy as np
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT/"config"/"config.yaml"))

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

bbox = CFG["aoi"]["bbox"]  # [west, south, east, north]

def preprocess_file(file_path, out_dir):
    print(f"Processando {file_path.name} ...")
    
    ds = xr.open_dataset(file_path)

    # identifica variável de SST
    if "analysed_sst" in ds.variables:
        sst = ds["analysed_sst"]
    elif "sst" in ds.variables:
        sst = ds["sst"]
    else:
        raise ValueError("Não encontrei variável de SST no arquivo")

    # converte para °C se estiver em Kelvin
    if sst.max() > 200:
        sst = sst - 273.15

    # recorte por bbox
    west, south, east, north = bbox
    sst_clip = sst.sel(lon=slice(west, east), lat=slice(south, north))

    # calcula gradiente espacial (proxy de frentes térmicas)
    # garante 2D (lat x lon)
    if "time" in sst_clip.dims:
        sst2d = sst_clip.isel(time=0).values
    else:
        sst2d = sst_clip.values

    # gradiente espacial
    dTdy, dTdx = np.gradient(sst2d)
    grad = np.sqrt(dTdx**2 + dTdy**2)

    # cria dataset de saída
    out = xr.Dataset(
    {
        "sst": (("lat", "lon"), sst2d),
        "sst_gradient": (("lat", "lon"), grad),
    },
    coords={"lat": sst_clip.lat, "lon": sst_clip.lon}
)

    # salva em NetCDF
    out_file = out_dir / file_path.name.replace(".nc", "_proc.nc")
    out.to_netcdf(out_file)
    print(f"Arquivo processado salvo em {out_file}")

if __name__ == "__main__":
    files = sorted(RAW.glob("*.nc"))
    if not files:
        print("Nenhum arquivo bruto encontrado em data/raw/")
        exit()

    for f in files:
        preprocess_file(f, PROC)

    print("✅ Pré-processamento concluído. Arquivos em data/processed/")
