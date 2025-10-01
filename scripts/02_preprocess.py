#!/usr/bin/env python3
"""Pre-processamento de SST e MODIS: recorte por bbox, conversão para Celsius e gradiente por timestep."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import xarray as xr
from typing import Iterable

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent

if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils import get_bbox, load_config, project_root


ROOT = project_root()
CFG = load_config()

RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

BBOX = get_bbox(CFG) or [-80.0, 25.0, -60.0, 40.0]

LAT_CANDIDATES = ("lat", "latitude")
LON_CANDIDATES = ("lon", "longitude")


def detect_coordinate(data: xr.DataArray, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in data.coords:
            return name
    for dim in data.dims:
        lower = dim.lower()
        if any(lower.startswith(prefix[:3]) for prefix in candidates):
            return dim
    raise KeyError(f"Nenhuma coordenada encontrada para candidatos {candidates} em {data.dims}")


def ensure_sorted(da: xr.DataArray, coord: str) -> xr.DataArray:
    values = da[coord]
    if values[0] > values[-1]:
        return da.sortby(coord)
    return da


def clip_bbox(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    west, south, east, north = BBOX
    da = ensure_sorted(da, lon_name)
    da = ensure_sorted(da, lat_name)
    return da.sel({lon_name: slice(west, east), lat_name: slice(south, north)})


def convert_to_celsius(da: xr.DataArray) -> xr.DataArray:
    max_val = float(da.max().compute() if hasattr(da.data, "compute") else da.max())
    if max_val > 200.0:
        da = da - 273.15
        da.attrs["units"] = "degree_Celsius"
    return da


def gradient_magnitude(field: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    """Calcular |gradiente| evitando erro em grades muito pequenas."""
    if field.sizes[lat_name] <= 1 or field.sizes[lon_name] <= 1:
        return xr.full_like(field, np.nan)

    def _gradient(arr: np.ndarray) -> np.ndarray:
        d_dy, d_dx = np.gradient(arr)
        return np.sqrt(d_dx**2 + d_dy**2)

    field_rechunked = field.chunk({lat_name: -1, lon_name: -1})

    grad = xr.apply_ufunc(
        _gradient,
        field_rechunked,
        input_core_dims=[[lat_name, lon_name]],
        output_core_dims=[[lat_name, lon_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    grad.name = "sst_gradient"
    grad.attrs["description"] = "Modulo do gradiente calculado com np.gradient"
    return grad


def preprocess_file(file_path: Path) -> Path:
    print(f"Processando {file_path.name} ...")

    with xr.open_dataset(file_path, chunks={"time": 1}, decode_timedelta=False) as ds:
        # 🔧 Corrige lat/lon se estiverem como variáveis
        if "lat" in ds.data_vars and "lon" in ds.data_vars:
            ds = ds.set_coords(["lat", "lon"])

        if "analysed_sst" in ds.variables or "sst" in ds.variables:
            var_name = "analysed_sst" if "analysed_sst" in ds.variables else "sst"
            field = ds[var_name]

            lat_name = detect_coordinate(field, LAT_CANDIDATES)
            lon_name = detect_coordinate(field, LON_CANDIDATES)

            field = clip_bbox(field, lat_name, lon_name)
            field = convert_to_celsius(field)
            grad = gradient_magnitude(field, lat_name, lon_name)

            out = xr.Dataset({"sst": field, "sst_gradient": grad})

        elif "chlor_a" in ds.variables:
            field = ds["chlor_a"]
            lat_name = detect_coordinate(field, LAT_CANDIDATES)
            lon_name = detect_coordinate(field, LON_CANDIDATES)

            field = clip_bbox(field, lat_name, lon_name)
            field.attrs.setdefault("units", ds["chlor_a"].attrs.get("units", "mg m-3"))
            out = xr.Dataset({"chlor_a": field})

        else:
            raise ValueError("Variável reconhecida (SST ou chlor_a) não encontrada no arquivo")

        out.attrs["source_file"] = file_path.name
        out.attrs["bbox"] = BBOX

        # CORREÇÃO: Manter o nome do arquivo (sem adicionar _proc novamente)
        # Arquivos já vêm renomeados: 20250926_SSTfnd-MUR.nc
        out_name = file_path.stem + "_proc.nc"
        out_path = PROC_DIR / out_name
        
        encoding = {var: {"zlib": True, "complevel": 4} for var in out.data_vars}
        out.to_netcdf(out_path, encoding=encoding, compute=True)

    print(f"✅ Arquivo processado salvo em {out_path}")
    return out_path


def main() -> None:
    files = sorted(RAW_DIR.glob("*.nc"))
    if not files:
        print("Nenhum arquivo bruto encontrado em data/raw/")
        return

    print(f"\n{'='*60}")
    print(f"Encontrados {len(files)} arquivo(s) para processar")
    print(f"{'='*60}\n")

    processed = 0
    failed = 0

    for file_path in files:
        try:
            preprocess_file(file_path)
            processed += 1
        except Exception as exc:
            print(f"❌ Falha ao processar {file_path.name}: {exc}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"🏁 Pre-processamento concluído!")
    print(f"   ✅ Processados: {processed}")
    print(f"   ❌ Falhas: {failed}")
    print(f"   📂 Arquivos em: {PROC_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()