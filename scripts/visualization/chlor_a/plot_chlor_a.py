#!/usr/bin/env python3
"""Generate chlorophyll-a maps from feature CSV files (MODIS + PACE combined)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/features",
                        help="Directory containing feature CSV files")
    parser.add_argument("--pattern", default="*_features.csv",
                        help="Glob pattern to select feature CSVs")
    parser.add_argument("--out-dir", default="data/viz/chlor_a",
                        help="Output directory for PNG maps")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Fixed minimum value for color scale")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Fixed maximum value for color scale")
    parser.add_argument("--pctl", type=float, nargs=2, default=(2.0, 98.0),
                        help="Percentiles (min,max) to derive color scale when vmin/vmax not provided")
    parser.add_argument("--cmap", default="viridis",
                        help="Matplotlib colormap name to use (default: viridis)")
    return parser


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"lat", "lon"}
    if "chlor_a" not in df.columns and not ({"chlor_a_modis", "chlor_a_pace"} & set(df.columns)):
        raise ValueError("CSV must contain 'chlor_a' or chlor_a_[modis|pace] columns")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {csv_path.name}")
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


def plot_map(csv_path: Path, out_dir: Path, cmap_name: str,
             vmin: float | None, vmax: float | None, pctl: tuple[float, float]) -> Path | None:
    try:
        df = load_csv(csv_path)
        combined = combined_chlor_a(df)
        df = df.assign(chlor_a_combined=combined).dropna(subset=["chlor_a_combined"])
        if df.empty:
            print(f"[skip] {csv_path.name}: no valid chlor_a")
            return None
        lats, lons, grid = compute_grid(df, df["chlor_a_combined"])
        vmin_eff, vmax_eff = pick_scale(grid, vmin, vmax, pctl)

        date_str = ""
        if "date" in df.columns and not df["date"].isna().all():
            date_str = str(df["date"].iloc[0])

        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{csv_path.stem}_chlor_a_map.png"

        if HAVE_CARTOPY:
            proj = ccrs.PlateCarree()
            fig = plt.figure(figsize=(9, 7))
            ax = plt.axes(projection=proj)
            extent = [float(np.min(lons)), float(np.max(lons)), float(np.min(lats)), float(np.max(lats))]
            ax.set_extent(extent, crs=proj)
            try:
                ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            except Exception:
                pass
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
            gl.right_labels = False
            gl.top_labels = False
            mesh = ax.pcolormesh(lons, lats, grid, cmap=cmap_name, shading="auto",
                                 vmin=vmin_eff, vmax=vmax_eff, transform=proj)
            cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02)
            cbar.set_label("Chlor_a (mg m^-3)")
            title = "Chlorophyll-a"
            if date_str:
                title += f" — {date_str}"
            ax.set_title(title)
            fig.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.figure(figsize=(8, 6))
            mesh = plt.pcolormesh(lons, lats, grid, cmap=cmap_name, shading="auto",
                                  vmin=vmin_eff, vmax=vmax_eff)
            plt.colorbar(mesh, label="Chlor_a (mg m^-3)")
            plt.title(f"Chlorophyll-a — {date_str}" if date_str else "Chlorophyll-a")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close()

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

    print(f"Generating chlor_a maps for {len(files)} file(s) into {out_dir} ...")
    for csv_path in files:
        plot_map(csv_path, out_dir, args.cmap, args.vmin, args.vmax, tuple(args.pctl))


if __name__ == "__main__":
    main()

