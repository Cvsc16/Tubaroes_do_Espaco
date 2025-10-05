#!/usr/bin/env python3
"""Generate True Color vs SST comparison panels for processed NetCDF files."""

from __future__ import annotations

import argparse
import io
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr
from PIL import Image


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Directory with processed SST NetCDF files")
    parser.add_argument("--truecolor-dir", default="data/compare",
                        help="Directory to read/save MODIS true color imagery")
    parser.add_argument("--pattern", default="*_SSTfnd-MUR_proc.nc",
                        help="Glob pattern used to find NetCDF inputs")
    parser.add_argument("--out-dir", default="data/viz/sst",
                        help="Output directory for comparison PNGs")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Fixed minimum for SST color scale")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Fixed maximum for SST color scale")
    parser.add_argument("--pctl", type=float, nargs=2, default=(2.0, 98.0),
                        help="Percentiles (min,max) used to derive scale when vmin/vmax are not provided")
    parser.add_argument("--time-index", type=int, default=0,
                        help="Time index to use when the NetCDF has a time dimension")
    parser.add_argument("--disable-download", action="store_true",
                        help="Do not attempt to download true color imagery when missing")
    parser.add_argument("--wms-base-url", default="https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
                        help="NASA GIBS WMS endpoint for true color requests")
    parser.add_argument("--wms-layer", default="MODIS_Aqua_CorrectedReflectance_TrueColor",
                        help="WMS layer name to request")
    parser.add_argument("--wms-width", type=int, default=1024,
                        help="Requested image width in pixels")
    parser.add_argument("--wms-height", type=int, default=1024,
                        help="Requested image height in pixels")
    parser.add_argument("--download-timeout", type=float, default=60.0,
                        help="Timeout (seconds) per download attempt")
    parser.add_argument("--download-retries", type=int, default=2,
                        help="Number of retry attempts when a download fails")
    return parser


def iter_files(directory: Path, pattern: str) -> Iterable[Path]:
    return sorted(directory.glob(pattern))


def detect_coord_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for option in candidates:
        if option in dataset.dims or option in dataset.coords:
            return option
    raise KeyError(f"Could not find coordinates for {candidates} in {list(dataset.dims)}")


def parse_date_from_name(path: Path) -> str:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if len(digits) >= 8:
        token = digits[:8]
        try:
            return datetime.strptime(token, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return ""
    return ""


def find_truecolor(date_iso: str, directory: Path) -> Path | None:
    if not date_iso:
        return None
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = directory / f"MODIS_truecolor_{date_iso}{ext}"
        if candidate.exists():
            return candidate
    matches = list(directory.glob(f"*{date_iso}*"))
    return matches[0] if matches else None


def download_truecolor(date_iso: str, directory: Path, bbox: tuple[float, float, float, float], cfg: dict) -> Path | None:
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"MODIS_truecolor_{date_iso}.jpg"

    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": cfg["layer"],
        "STYLES": "",
        "FORMAT": "image/jpeg",
        "BBOX": ",".join(f"{coord:.6f}" for coord in bbox),
        "WIDTH": str(cfg["width"]),
        "HEIGHT": str(cfg["height"]),
        "SRS": "EPSG:4326",
        "TIME": date_iso,
    }

    attempts = cfg["retries"] + 1
    for attempt in range(attempts):
        try:
            response = requests.get(cfg["base_url"], params=params, timeout=cfg["timeout"])
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            img.save(out_path)
            print(f"[download] true color saved to {out_path}")
            return out_path
        except Exception as exc:
            if attempt < attempts - 1:
                print(f"[warn] failed to download true color ({date_iso}) attempt {attempt + 1}/{attempts}: {exc}")
            else:
                print(f"[error] giving up on true color ({date_iso}): {exc}")
    return None


def ensure_truecolor(date_iso: str, directory: Path, bbox: tuple[float, float, float, float], cfg: dict) -> Path | None:
    existing = find_truecolor(date_iso, directory)
    if existing is not None:
        return existing
    if not cfg.get("enable", True):
        return None
    return download_truecolor(date_iso, directory, bbox, cfg)


def pick_scale(values: np.ndarray, vmin: float | None, vmax: float | None, pctl: tuple[float, float]) -> tuple[float, float]:
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return 0.0, 1.0
    if vmin is None or vmax is None:
        lo, hi = np.nanpercentile(finite_vals, [pctl[0], pctl[1]])
        vmin_eff = float(vmin if vmin is not None else lo)
        vmax_eff = float(vmax if vmax is not None else hi)
    else:
        vmin_eff, vmax_eff = float(vmin), float(vmax)
    if vmin_eff >= vmax_eff:
        vmin_eff = float(np.nanmin(finite_vals))
        vmax_eff = float(np.nanmax(finite_vals))
    return vmin_eff, vmax_eff


def plot_compare(nc_path: Path, truecolor_dir: Path, out_dir: Path,
                 vmin: float | None, vmax: float | None, pctl: tuple[float, float],
                 time_index: int, download_cfg: dict) -> Path | None:
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as exc:
        print(f"[error] failed to open {nc_path.name}: {exc}")
        return None

    if "sst" not in ds:
        print(f"[skip] {nc_path.name}: variable 'sst' missing")
        ds.close()
        return None

    lat_name = detect_coord_name(ds, ("lat", "latitude"))
    lon_name = detect_coord_name(ds, ("lon", "longitude"))

    sst = ds["sst"]
    if "time" in sst.dims:
        available = sst.sizes["time"]
        idx = max(0, min(time_index, available - 1))
        sst = sst.isel(time=idx)

    sst_vals = sst.values.squeeze()
    lats = sst[lat_name].values if lat_name in sst.coords else ds[lat_name].values
    lons = sst[lon_name].values if lon_name in sst.coords else ds[lon_name].values

    vmin_eff, vmax_eff = pick_scale(sst_vals, vmin, vmax, pctl)

    date_iso = parse_date_from_name(nc_path)
    title_date = f" - {date_iso}" if date_iso else ""

    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    extent = [lon_min, lon_max, lat_min, lat_max]
    bbox = (lon_min, lat_min, lon_max, lat_max)

    truecolor_path = ensure_truecolor(date_iso, truecolor_dir, bbox, download_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"compare_truecolor_sst_{nc_path.stem}.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": ccrs.PlateCarree()})
    ax_true, ax_sst = axes

    for ax in axes:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=1)
        grid = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
        grid.top_labels = False
        grid.right_labels = False

    if truecolor_path and truecolor_path.exists():
        try:
            img = mpimg.imread(truecolor_path)
            ax_true.imshow(img, origin="upper", extent=extent, transform=ccrs.PlateCarree())
            ax_true.set_title(f"MODIS True Color{title_date}")
        except Exception as exc:
            ax_true.text(0.5, 0.5, f"error reading true color\n{exc}", ha="center", va="center")
            ax_true.set_title("True Color unavailable")
    else:
        ax_true.text(0.5, 0.5, "True Color not available", ha="center", va="center")
        ax_true.set_title("True Color not available")

    cmap = plt.cm.jet.copy()
    cmap.set_bad("white")
    mesh = ax_sst.pcolormesh(lons, lats, sst_vals, cmap=cmap, shading="auto",
                             vmin=vmin_eff, vmax=vmax_eff, transform=ccrs.PlateCarree())
    ax_sst.set_title(f"SST MUR{title_date}")
    colorbar = plt.colorbar(mesh, ax=ax_sst, orientation="vertical", fraction=0.046, pad=0.04)
    colorbar.set_label("SST (deg C)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    ds.close()

    print(f"[ok] {out_png}")
    return out_png


def main() -> None:
    args = build_argparser().parse_args()
    processed_dir = Path(args.processed_dir)
    truecolor_dir = Path(args.truecolor_dir)
    out_dir = Path(args.out_dir)

    download_cfg = {
        "enable": not args.disable_download,
        "base_url": args.wms_base_url,
        "layer": args.wms_layer,
        "width": max(1, args.wms_width),
        "height": max(1, args.wms_height),
        "timeout": max(1e-3, args.download_timeout),
        "retries": max(0, args.download_retries),
    }

    files = list(iter_files(processed_dir, args.pattern))
    if not files:
        print(f"No NetCDF found in {processed_dir} matching {args.pattern}")
        return

    print(f"Generating comparisons for {len(files)} file(s) in {out_dir} ...")
    for nc_path in files:
        plot_compare(nc_path, truecolor_dir, out_dir, args.vmin, args.vmax,
                     tuple(args.pctl), args.time_index, download_cfg)


if __name__ == "__main__":
    main()
