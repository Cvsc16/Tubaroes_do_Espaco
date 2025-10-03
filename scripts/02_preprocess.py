#!/usr/bin/env python3
"""Pre-processamento de SST, MODIS, PACE e SWOT:
- Recorte por bbox
- Conversão para Celsius (SST)
- Gradiente por timestep (SST) com Dask (rechunk lat/lon)
- Mantém SWOT como pontos brutos (sem interpolação para grid)
- Junta múltiplos passes SWOT por dia em 1 arquivo diário
- Exporta NetCDF compactado em data/processed/
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import xarray as xr
from typing import Iterable, Dict, List
from collections import defaultdict

# ---------------- bootstrap ----------------
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

# ============== helpers gerais ==============

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
    """|gradiente| com Dask; garante 1 chunk em lat/lon (evita erro core dimension)."""
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
        dask_gufunc_kwargs={"allow_rechunk": False},
        output_dtypes=[float],
    )
    grad.name = "sst_gradient"
    grad.attrs["description"] = "Modulo do gradiente calculado com np.gradient"
    return grad

# ============== SWOT helpers ==============

def normalize_lon_180(lon: np.ndarray) -> np.ndarray:
    """Converte longitudes de [0,360] para [-180,180]."""
    return ((lon + 180.0) % 360.0) - 180.0

def _bbox_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    west, south, east, north = BBOX
    return (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)

def load_swot_points(file_path: Path, stride: int = 6) -> xr.Dataset | None:
    """Extrai pontos (lat, lon, ssh) dos grupos left/right de um arquivo SWOT L2."""
    all_lat, all_lon, all_ssh = [], [], []

    for side in ["left", "right"]:
        with xr.open_dataset(file_path, group=side) as ds_side:
            lat = ds_side["latitude"].values[::stride, ::stride]
            lon = ds_side["longitude"].values[::stride, ::stride]
            ssh = ds_side["ssh_karin"].values[::stride, ::stride]

            lon = normalize_lon_180(lon)

            mask = _bbox_mask(lat, lon)
            mask &= np.isfinite(lat) & np.isfinite(lon) & np.isfinite(ssh)

            if not np.any(mask):
                continue

            all_lat.append(lat[mask].astype(np.float32))
            all_lon.append(lon[mask].astype(np.float32))
            all_ssh.append(ssh[mask].astype(np.float32))

    if not all_lat:
        return None

    lat_cat = np.concatenate(all_lat)
    lon_cat = np.concatenate(all_lon)
    ssh_cat = np.concatenate(all_ssh)

    if lat_cat.size < 3:
        return None

    print(f"[SWOT] {file_path.name}: {lat_cat.size} pontos válidos no bbox (lon normalizado)")
    return xr.Dataset(
        {
            "lat": (("points",), lat_cat),
            "lon": (("points",), lon_cat),
            "ssh": (("points",), ssh_cat),
        }
    )

# ============== processamento arquivo-a-arquivo ==============

def preprocess_file(file_path: Path):
    print(f"Processando {file_path.name} ...")

    if "SSTfnd-MUR" in file_path.name:
        with xr.open_dataset(file_path, chunks={"time": 1}, decode_timedelta=False) as ds:
            var_name = "analysed_sst" if "analysed_sst" in ds.variables else "sst"
            field = ds[var_name]
            lat_name = detect_coordinate(field, LAT_CANDIDATES)
            lon_name = detect_coordinate(field, LON_CANDIDATES)
            field = clip_bbox(field, lat_name, lon_name)
            field = convert_to_celsius(field)
            grad = gradient_magnitude(field, lat_name, lon_name)
            out = xr.Dataset({"sst": field, "sst_gradient": grad})

        out_path = PROC_DIR / (file_path.stem + "_proc.nc")
        encoding = {var: {"zlib": True, "complevel": 4} for var in out.data_vars}
        out.to_netcdf(out_path, encoding=encoding, compute=True)
        print(f"✅ Arquivo processado salvo em {out_path}")
        return out_path

    if "CHL-MODIS" in file_path.name or "CHL-PACE" in file_path.name:
        with xr.open_dataset(file_path) as ds:
            varname = list(ds.data_vars)[0]
            field = ds[varname]
            lat_name = detect_coordinate(field, LAT_CANDIDATES)
            lon_name = detect_coordinate(field, LON_CANDIDATES)
            field = clip_bbox(field, lat_name, lon_name)
            out = xr.Dataset({varname: field})

        out_path = PROC_DIR / (file_path.stem + "_proc.nc")
        encoding = {var: {"zlib": True, "complevel": 4} for var in out.data_vars}
        out.to_netcdf(out_path, encoding=encoding, compute=True)
        print(f"✅ Arquivo processado salvo em {out_path}")
        return out_path

    if "SSH-SWOT" in file_path.name:
        digits = "".join([c for c in file_path.name if c.isdigit()])[:8]
        ds_pts = load_swot_points(file_path, stride=6)
        if ds_pts is not None:
            print(f"[SWOT] adicionando {file_path.name} ao dia {digits}")
            return ("SWOT", digits, ds_pts)
        else:
            print(f"[SWOT] {file_path.name}: nenhum ponto válido no bbox")
            return None

    print(f"⚠️ Arquivo não reconhecido: {file_path.name}")
    return None

# ============== main ==============

def main():
    files = sorted(RAW_DIR.glob("**/*.nc"))
    if not files:
        print("Nenhum arquivo bruto encontrado em data/raw/")
        return

    print(f"\n{'='*60}")
    print(f"Encontrados {len(files)} arquivo(s) para processar")
    print(f"{'='*60}\n")

    processed, failed = 0, 0
    swot_by_day: Dict[str, List[xr.Dataset]] = defaultdict(list)

    for file_path in files:
        try:
            result = preprocess_file(file_path)
            if isinstance(result, tuple) and result[0] == "SWOT":
                _, digits, ds_pts = result
                swot_by_day[digits].append(ds_pts)
            elif result:
                processed += 1
        except Exception as exc:
            print(f"❌ Falha ao processar {file_path.name}: {exc}")
            failed += 1

    for date, datasets in swot_by_day.items():
        try:
            print(f"[SWOT] Combinando {len(datasets)} passes para {date}")
            ds_points = xr.concat(datasets, dim="points")
            out_path = PROC_DIR / f"{date}_SSH-SWOT_points.nc"
            encoding = {var: {"zlib": True, "complevel": 4} for var in ds_points.data_vars}
            ds_points.to_netcdf(out_path, encoding=encoding, compute=True)
            print(f"✅ SWOT pontos brutos salvo em {out_path}")
            processed += 1
        except Exception as exc:
            print(f"❌ Falha ao combinar SWOT {date}: {exc}")
            failed += 1

    print(f"\n{'='*60}")
    print("🏁 Pre-processamento concluído!")
    print(f"   ✅ Processados: {processed}")
    print(f"   ❌ Falhas: {failed}")
    print(f"   📂 Arquivos em: {PROC_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
