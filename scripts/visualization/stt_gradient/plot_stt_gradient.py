#!/usr/bin/env python3
"""Generate SST gradient maps from feature CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/features",
                        help="Directory containing feature CSV files")
    parser.add_argument("--pattern", default="*_features.csv",
                        help="Glob pattern used to locate feature CSVs")
    parser.add_argument("--out-dir", default="data/viz/sst_gradient",
                        help="Output directory for PNG maps")
    parser.add_argument("--pctl", type=float, nargs=2, default=(2.0, 98.0),
                        help="Percentiles (abs min, abs max) for symmetric scaling")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Override for absolute maximum of the gradient scale")
    parser.add_argument("--cmap", default="seismic",
                        help="Matplotlib colormap name (default: seismic)")
    parser.add_argument("--mask-chlor-a", action="store_true",
                        help="Mask pixels lacking chlorophyll data (when available)")
    return parser


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"lat", "lon", "sst_gradient"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns {missing} in {csv_path.name}")
    return df


def apply_masks(df: pd.DataFrame, mask_chlor_a: bool) -> pd.DataFrame:
    df = df.dropna(subset=["sst_gradient"])
    if mask_chlor_a:
        chlor_cols = [col for col in ("chlor_a", "chlor_a_pace", "chlor_a_modis") if col in df.columns]
        if chlor_cols:
            mask = ~pd.isna(df[chlor_cols]).all(axis=1)
            df = df[mask]
    return df


def compute_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = df.pivot(index="lat", columns="lon", values="sst_gradient")
    grid = grid.reindex(index=np.sort(grid.index.values), columns=np.sort(grid.columns.values))
    lats = grid.index.values
    lons = grid.columns.values
    return lats, lons, grid.values


def pick_scale(values: np.ndarray, vmax: float | None, pctl: tuple[float, float]) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.1
    if vmax is not None and vmax > 0:
        return float(vmax)
    lo, hi = np.nanpercentile(np.abs(finite), [pctl[0], pctl[1]])
    vmax_eff = max(hi, lo)
    if vmax_eff <= 0:
        vmax_eff = float(np.nanmax(np.abs(finite))) or 0.1
    return float(vmax_eff)


def plot_map(csv_path: Path, out_dir: Path, cmap_name: str,
             vmax: float | None, pctl: tuple[float, float], mask_chlor_a: bool) -> Path | None:
    try:
        df = load_csv(csv_path)
        df = apply_masks(df, mask_chlor_a)
        if df.empty:
            print(f"[skip] {csv_path.name}: no valid sst_gradient")
            return None

        lats, lons, grid = compute_grid(df)
        vmax_eff = pick_scale(grid, vmax, pctl)

        date_str = ""
        if "date" in df.columns and not df["date"].isna().all():
            date_str = str(df["date"].iloc[0])

        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{csv_path.stem}_sst_gradient_map.png"

        fig = plt.figure(figsize=(9, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        extent = [float(np.min(lons)), float(np.max(lons)), float(np.min(lats)), float(np.max(lats))]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        try:
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=1)
        except Exception:
            pass
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
        gl.right_labels = False
        gl.top_labels = False

        mesh = ax.pcolormesh(lons, lats, grid, cmap=cmap_name, shading="auto",
                              vmin=-vmax_eff, vmax=vmax_eff, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label("SST gradient (deg C per degree)")

        title = "SST Gradient"
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

    print(f"Generating SST gradient maps for {len(files)} file(s) into {out_dir} ...")
    for csv_path in files:
        plot_map(csv_path, out_dir, args.cmap, args.vmax, tuple(args.pctl), args.mask_chlor_a)


if __name__ == "__main__":
    main()

