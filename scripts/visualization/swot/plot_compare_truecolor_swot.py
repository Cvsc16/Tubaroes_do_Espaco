#!/usr/bin/env python3
"""Compare MODIS True Color imagery with SWOT sea-surface height maps."""

from __future__ import annotations

import argparse
import io
from datetime import datetime
from pathlib import Path
from typing import Iterable
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image

_THIS_FILE = Path(__file__).resolve()
for parent in _THIS_FILE.parents:
    if parent.name == "scripts":
        PROJECT_ROOT = parent.parent
        break
else:
    PROJECT_ROOT = _THIS_FILE.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import load_config, get_bbox


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/features",
                        help="Directory containing feature CSV files")
    parser.add_argument("--pattern", default="*_features.csv",
                        help="Glob pattern for feature CSV selection")
    parser.add_argument("--truecolor-dir", default="data/compare",
                        help="Directory to read/write MODIS true color images")
    parser.add_argument("--out-dir", default="data/viz/swot",
                        help="Directory for comparison PNGs")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Fixed minimum for SSH color scale")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Fixed maximum for SSH color scale")
    parser.add_argument("--pctl", type=float, nargs=2, default=(2.0, 98.0),
                        help="Percentiles (min,max) used when vmin/vmax are not provided")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Minimum swot_mask value to keep a pixel (default: 0.5)")
    parser.add_argument("--cmap", default="coolwarm",
                        help="Colormap for SSH visualization (default: coolwarm)")
    parser.add_argument("--disable-download", action="store_true",
                        help="Skip automatic true color download when image is missing")
    parser.add_argument("--wms-base-url", default="https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
                        help="NASA GIBS WMS endpoint")
    parser.add_argument("--wms-layer", default="MODIS_Aqua_CorrectedReflectance_TrueColor",
                        help="WMS layer name to request")
    parser.add_argument("--wms-width", type=int, default=1024,
                        help="Requested image width in pixels")
    parser.add_argument("--wms-height", type=int, default=1024,
                        help="Requested image height in pixels")
    parser.add_argument("--download-timeout", type=float, default=60.0,
                        help="Timeout (seconds) per download attempt")
    parser.add_argument("--download-retries", type=int, default=2,
                        help="Retry attempts when download fails")
    parser.add_argument("--auto-extent", action="store_true",
                        help="Derive extent from SWOT data instead of config bbox")
    return parser


def iter_files(directory: Path, pattern: str) -> Iterable[Path]:
    return sorted(directory.glob(pattern))


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ssh_swot" not in df.columns:
        raise ValueError(f"ssh_swot column missing in {csv_path.name}")
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError(f"{csv_path.name} missing lat/lon columns")
    return df


def combined_mask(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if "swot_mask" not in df.columns:
        return df
    mask = df["swot_mask"].astype(float).fillna(0.0)
    return df[mask > threshold]


CFG = load_config()
DEFAULT_BBOX = get_bbox(CFG) or [-80.0, 25.0, -60.0, 40.0]


def merge_extent(lons: np.ndarray, lats: np.ndarray, bbox: list[float], pad_factor: float = 0.15) -> list[float]:
    lon_min = float(np.nanmin(lons)) if lons.size else bbox[0]
    lon_max = float(np.nanmax(lons)) if lons.size else bbox[2]
    lat_min = float(np.nanmin(lats)) if lats.size else bbox[1]
    lat_max = float(np.nanmax(lats)) if lats.size else bbox[3]

    lon_min = min(lon_min, bbox[0])
    lon_max = max(lon_max, bbox[2])
    lat_min = min(lat_min, bbox[1])
    lat_max = max(lat_max, bbox[3])

    span_lon = max(lon_max - lon_min, 0.5)
    span_lat = max(lat_max - lat_min, 0.5)
    pad_lon = span_lon * pad_factor
    pad_lat = span_lat * pad_factor

    return [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]


def bbox_to_extent(bbox: list[float]) -> list[float]:
    west, south, east, north = bbox
    return [west, east, south, north]


def pick_scale(values: np.ndarray, vmin: float | None, vmax: float | None, pctl: tuple[float, float]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -0.1, 0.1
    if vmin is None or vmax is None:
        lo, hi = np.nanpercentile(finite, [pctl[0], pctl[1]])
        vmin_eff = float(vmin if vmin is not None else lo)
        vmax_eff = float(vmax if vmax is not None else hi)
    else:
        vmin_eff, vmax_eff = float(vmin), float(vmax)
    if vmin_eff >= vmax_eff:
        center = float(np.nanmean(finite))
        spread = float(np.nanstd(finite)) or 0.1
        vmin_eff, vmax_eff = center - spread, center + spread
    return vmin_eff, vmax_eff


def parse_date(df: pd.DataFrame) -> str:
    if "date" in df.columns and not df["date"].isna().all():
        value = str(df["date"].iloc[0])
        try:
            return datetime.fromisoformat(value).date().isoformat()
        except Exception:
            return value
    return ""


def parse_date_from_name(path: Path) -> str:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8], "%Y%m%d").strftime("%Y-%m-%d")
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


def plot_compare(csv_path: Path, truecolor_dir: Path, out_dir: Path,
                 vmin: float | None, vmax: float | None, pctl: tuple[float, float],
                 mask_threshold: float, cmap_name: str, download_cfg: dict,
                 auto_extent: bool) -> Path | None:
    try:
        df = load_csv(csv_path)
    except Exception as exc:
        print(f"[error] {csv_path.name}: {exc}")
        return None

    df = combined_mask(df, mask_threshold)
    df = df.dropna(subset=["ssh_swot"])
    if df.empty:
        print(f"[skip] {csv_path.name}: no valid ssh_swot")
        return None

    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    values = df["ssh_swot"].to_numpy()
    vmin_eff, vmax_eff = pick_scale(values, vmin, vmax, pctl)

    date_iso = parse_date(df)
    if not date_iso:
        date_iso = parse_date_from_name(csv_path)
    title_date = f" - {date_iso}" if date_iso else ""

    if auto_extent:
        lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
        lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
        pad_lon = max((lon_max - lon_min) * 0.1, 0.2)
        pad_lat = max((lat_max - lat_min) * 0.1, 0.2)
        extent_list = [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]
    else:
        extent_list = DEFAULT_BBOX
    extent = bbox_to_extent(extent_list)
    bbox = (extent_list[0], extent_list[2], extent_list[1], extent_list[3])

    truecolor_path = ensure_truecolor(date_iso, truecolor_dir, bbox, download_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"compare_truecolor_swot_{csv_path.stem}.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": ccrs.PlateCarree()})
    ax_tc, ax_swot = axes

    for ax in axes:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=1)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

    if truecolor_path and truecolor_path.exists():
        try:
            img = mpimg.imread(truecolor_path)
            ax_tc.imshow(img, origin="upper", extent=extent, transform=ccrs.PlateCarree())
            ax_tc.set_title(f"MODIS True Color{title_date}")
        except Exception as exc:
            ax_tc.text(0.5, 0.5, f"error reading true color\n{exc}", ha="center", va="center")
            ax_tc.set_title("True Color unavailable")
    else:
        ax_tc.text(0.5, 0.5, "True Color not available", ha="center", va="center")
        ax_tc.set_title("True Color not available")

    scatter = ax_swot.scatter(lons, lats, c=values, cmap=cmap_name, s=12, alpha=0.9,
                              vmin=vmin_eff, vmax=vmax_eff, transform=ccrs.PlateCarree())
    ax_swot.set_title(f"SWOT Sea-Surface Height{title_date}")
    cbar = plt.colorbar(scatter, ax=ax_swot, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("SSH SWOT (m)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] {out_png}")
    return out_png


def main() -> None:
    args = build_argparser().parse_args()
    features_dir = Path(args.features_dir)
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

    files = list(iter_files(features_dir, args.pattern))
    if not files:
        print(f"No feature CSV found in {features_dir} matching {args.pattern}")
        return

    print(f"Generating SWOT comparisons for {len(files)} file(s) in {out_dir} ...")
    for csv_path in files:
        plot_compare(csv_path, truecolor_dir, out_dir, args.vmin, args.vmax,
                     tuple(args.pctl), args.mask_threshold, args.cmap, download_cfg,
                     args.auto_extent)


if __name__ == "__main__":
    main()
