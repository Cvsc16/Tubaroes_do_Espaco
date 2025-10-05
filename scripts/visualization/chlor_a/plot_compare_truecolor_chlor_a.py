#!/usr/bin/env python3
"""Compare MODIS True Color imagery with combined chlorophyll-a maps."""

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
import pandas as pd
import requests
from PIL import Image


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/features",
                        help="Directory containing feature CSV files")
    parser.add_argument("--pattern", default="*_features.csv",
                        help="Glob pattern for feature CSV selection")
    parser.add_argument("--truecolor-dir", default="data/compare",
                        help="Directory to read/write MODIS true color images")
    parser.add_argument("--out-dir", default="data/viz/chlor_a",
                        help="Directory for comparison PNGs")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Fixed minimum for chlor_a color scale")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Fixed maximum for chlor_a color scale")
    parser.add_argument("--pctl", type=float, nargs=2, default=(2.0, 98.0),
                        help="Percentiles (min,max) used when vmin/vmax are not provided")
    parser.add_argument("--time-column", default="date",
                        help="Column to extract date information (default: date)")
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
    parser.add_argument("--cmap", default="viridis",
                        help="Colormap for chlor_a visualization (default: viridis)")
    return parser


def iter_files(directory: Path, pattern: str) -> Iterable[Path]:
    return sorted(directory.glob(pattern))


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "chlor_a" not in df.columns and not ({"chlor_a_modis", "chlor_a_pace"} & set(df.columns)):
        raise ValueError(f"{csv_path.name} missing chlor_a columns")
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError(f"{csv_path.name} missing lat/lon columns")
    return df


def combined_chlor_a(df: pd.DataFrame) -> pd.Series:
    if "chlor_a" in df.columns:
        return df["chlor_a"]
    sources = {}
    for col in ("chlor_a_pace", "chlor_a_modis"):
        if col in df.columns:
            sources[col] = df[col]
    if not sources:
        raise ValueError("No chlor_a sources available")
    stacked = pd.DataFrame(sources)
    combined = stacked.sum(axis=1, skipna=True)
    combined[stacked.isna().all(axis=1)] = np.nan
    return combined.astype("float32")


def compute_grid(df: pd.DataFrame, values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_local = df.copy()
    df_local["chlor_a_combined"] = values
    grid = df_local.pivot(index="lat", columns="lon", values="chlor_a_combined")
    grid = grid.reindex(index=np.sort(grid.index.values), columns=np.sort(grid.columns.values))
    lats = grid.index.values
    lons = grid.columns.values
    return lats, lons, grid.values


def pick_scale(values: np.ndarray, vmin: float | None, vmax: float | None, pctl: tuple[float, float]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if vmin is None or vmax is None:
        lo, hi = np.nanpercentile(finite, [pctl[0], pctl[1]])
        vmin_eff = float(vmin if vmin is not None else lo)
        vmax_eff = float(vmax if vmax is not None else hi)
    else:
        vmin_eff, vmax_eff = float(vmin), float(vmax)
    if vmin_eff >= vmax_eff:
        vmin_eff = float(np.nanmin(finite))
        vmax_eff = float(np.nanmax(finite))
    return vmin_eff, vmax_eff


def parse_date(df: pd.DataFrame, column: str, fallback: str) -> str:
    if column in df.columns and not df[column].isna().all():
        value = str(df[column].iloc[0])
        try:
            return datetime.fromisoformat(value).date().isoformat()
        except Exception:
            return value
    return fallback


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
                 date_column: str, download_cfg: dict, cmap_name: str) -> Path | None:
    try:
        df = load_csv(csv_path)
    except Exception as exc:
        print(f"[error] {csv_path.name}: {exc}")
        return None

    combined = combined_chlor_a(df)
    df = df.assign(chlor_a_combined=combined).dropna(subset=["chlor_a_combined"])
    if df.empty:
        print(f"[skip] {csv_path.name}: no valid chlor_a")
        return None

    lats, lons, grid = compute_grid(df, df["chlor_a_combined"])
    vmin_eff, vmax_eff = pick_scale(grid, vmin, vmax, pctl)

    date_from_name = ""
    digits = "".join(ch for ch in csv_path.stem if ch.isdigit())
    if len(digits) >= 8:
        try:
            date_from_name = datetime.strptime(digits[:8], "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            date_from_name = ""

    date_iso = parse_date(df, date_column, date_from_name)
    title_date = f" - {date_iso}" if date_iso else ""

    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    extent = [lon_min, lon_max, lat_min, lat_max]
    bbox = (lon_min, lat_min, lon_max, lat_max)

    truecolor_path = ensure_truecolor(date_iso, truecolor_dir, bbox, download_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"compare_truecolor_chlor_a_{csv_path.stem}.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": ccrs.PlateCarree()})
    ax_tc, ax_chl = axes

    for ax in axes:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=1)
        gridlines = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
        gridlines.top_labels = False
        gridlines.right_labels = False

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

    mesh = ax_chl.pcolormesh(lons, lats, grid, cmap=cmap_name, shading="auto",
                              vmin=vmin_eff, vmax=vmax_eff, transform=ccrs.PlateCarree())
    ax_chl.set_title(f"Chlorophyll-a{title_date}")
    cbar = plt.colorbar(mesh, ax=ax_chl, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Chlor_a (mg m^-3)")

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

    print(f"Generating chlor_a comparisons for {len(files)} file(s) in {out_dir} ...")
    for csv_path in files:
        plot_compare(csv_path, truecolor_dir, out_dir, args.vmin, args.vmax,
                     tuple(args.pctl), args.time_column, download_cfg, args.cmap)


if __name__ == "__main__":
    main()
