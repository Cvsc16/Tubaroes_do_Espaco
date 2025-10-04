#!/usr/bin/env python3
"""Generate SWOT sea-surface height maps from feature CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
                        help="Glob pattern to select feature CSVs")
    parser.add_argument("--out-dir", default="data/viz/swot",
                        help="Output directory for PNG maps")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Fixed minimum value for SSH color scale")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Fixed maximum value for SSH color scale")
    parser.add_argument("--pctl", type=float, nargs=2, default=(2.0, 98.0),
                        help="Percentiles (min,max) to derive scale when vmin/vmax not provided")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Minimum swot_mask value to keep a pixel (default: 0.5)")
    parser.add_argument("--cmap", default="coolwarm",
                        help="Matplotlib colormap for SSH values (default: coolwarm)")
    parser.add_argument("--auto-extent", action="store_true",
                        help="Derive extent from SWOT data instead of config bbox")
    return parser


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"lat", "lon"}
    if "ssh_swot" not in df.columns:
        raise ValueError(f"ssh_swot column missing in {csv_path.name}")
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns {missing} in {csv_path.name}")
    return df


def apply_mask(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
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


def plot_map(csv_path: Path, out_dir: Path, cmap_name: str,
             vmin: float | None, vmax: float | None, pctl: tuple[float, float],
             mask_threshold: float, auto_extent: bool) -> Path | None:
    try:
        df = load_csv(csv_path)
        df = apply_mask(df, mask_threshold)
        df = df.dropna(subset=["ssh_swot"])
        if df.empty:
            print(f"[skip] {csv_path.name}: no valid ssh_swot values")
            return None

        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()
        values = df["ssh_swot"].to_numpy()
        vmin_eff, vmax_eff = pick_scale(values, vmin, vmax, pctl)

        date_str = ""
        if "date" in df.columns and not df["date"].isna().all():
            date_str = str(df["date"].iloc[0])

        if auto_extent:
            lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
            lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
            pad_lon = max((lon_max - lon_min) * 0.1, 0.2)
            pad_lat = max((lat_max - lat_min) * 0.1, 0.2)
            extent = [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]
        else:
            extent = bbox_to_extent(DEFAULT_BBOX)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{csv_path.stem}_ssh_swot_map.png"

        fig = plt.figure(figsize=(9, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        try:
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=1)
        except Exception:
            pass
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.right_labels = False
        gl.top_labels = False

        scatter = ax.scatter(lons, lats, c=values, cmap=cmap_name, s=12, alpha=0.9,
                             vmin=vmin_eff, vmax=vmax_eff, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label("SSH SWOT (m)")

        title = "SWOT Sea-Surface Height"
        if date_str:
            title += f" — {date_str}"
        ax.set_title(title)

        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] {out_png}")
        return out_png
    except Exception as exc:
        print(f"[error] failed for {csv_path.name}: {exc}")
        return None


def main() -> None:
    args = build_argparser().parse_args()
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    files = sorted(features_dir.glob(args.pattern))
    if not files:
        print(f"No feature CSV found in {features_dir} matching {args.pattern}")
        return

    print(f"Generating SWOT maps for {len(files)} file(s) into {out_dir} ...")
    for csv_path in files:
        plot_map(csv_path, out_dir, args.cmap, args.vmin, args.vmax,
                 tuple(args.pctl), args.mask_threshold, args.auto_extent)


if __name__ == "__main__":
    main()
