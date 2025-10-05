#!/usr/bin/env python3
"""Gera mapas de SST a partir de cada arquivo de features CSV e
adiciona mapa de referência (costas/continentes) como contexto geográfico.

Uso:
  python scripts/visualization/sst/plot_sst.py \
    --features-dir data/features \
    --pattern *_features.csv \
    --out-dir data/viz/sst
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", default="data/features", help="Diretório com CSVs de features")
    p.add_argument("--pattern", default="*_features.csv", help="Padrão dos arquivos CSV a processar")
    p.add_argument("--out-dir", default="data/viz/sst", help="Diretório de saída para PNGs")
    p.add_argument("--vmin", type=float, default=None, help="Valor mínimo fixo para a escala de cores")
    p.add_argument("--vmax", type=float, default=None, help="Valor máximo fixo para a escala de cores")
    p.add_argument("--pctl", type=float, nargs=2, default=(2.0, 98.0),
                   help="Percentis (min,max) para limitar a escala quando vmin/vmax não forem fornecidos")
    return p


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if {"lat", "lon", "sst"}.issubset(df.columns) is False:
        raise ValueError(f"Colunas necessárias ausentes em {csv_path.name} (precisa de lat, lon, sst)")
    df = df.dropna(subset=["sst"]).copy()
    return df


def compute_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Reindexa grade para ordem crescente em lat/lon
    grid = df.pivot(index="lat", columns="lon", values="sst")
    grid = grid.reindex(index=np.sort(grid.index.values), columns=np.sort(grid.columns.values))
    lats = grid.index.values
    lons = grid.columns.values
    return lats, lons, grid.values


def pick_scale(df: pd.DataFrame, vmin: float | None, vmax: float | None, pctl: tuple[float, float]) -> tuple[float, float]:
    vals = df["sst"].dropna().values
    if vmin is None or vmax is None:
        p_lo, p_hi = np.nanpercentile(vals, [pctl[0], pctl[1]])
        vmin_eff = float(p_lo if vmin is None else vmin)
        vmax_eff = float(p_hi if vmax is None else vmax)
    else:
        vmin_eff, vmax_eff = float(vmin), float(vmax)
    if not np.isfinite(vmin_eff) or not np.isfinite(vmax_eff) or vmin_eff >= vmax_eff:
        vmin_eff = float(np.nanmin(vals))
        vmax_eff = float(np.nanmax(vals))
    return vmin_eff, vmax_eff


def plot_sst_map(csv_path: Path, out_dir: Path, vmin: float | None, vmax: float | None, pctl: tuple[float, float]) -> Path | None:
    try:
        df = load_csv(csv_path)
        if df.empty:
            print(f"[skip] {csv_path.name}: sem SST válida")
            return None
        lats, lons, grid = compute_grid(df)
        vmin_eff, vmax_eff = pick_scale(df, vmin, vmax, pctl)

        date_str = ""
        if "date" in df.columns and not df["date"].isna().all():
            date_str = str(df["date"].iloc[0])

        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{csv_path.stem}_sst_map.png"

        if HAVE_CARTOPY:
            proj = ccrs.PlateCarree()
            fig = plt.figure(figsize=(9, 7))
            ax = plt.axes(projection=proj)
            ax.set_extent([float(np.min(lons)), float(np.max(lons)), float(np.min(lats)), float(np.max(lats))], crs=proj)
            try:
                ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            except Exception:
                # Caso recursos do NaturalEarth não estejam disponíveis em modo offline
                pass
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
            gl.right_labels = False; gl.top_labels = False
            mesh = ax.pcolormesh(lons, lats, grid, cmap='jet', shading='auto', vmin=vmin_eff, vmax=vmax_eff, transform=proj)
            cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('SST (°C)')
            ax.set_title(f"SST — {date_str}" if date_str else "SST")
            fig.savefig(out_png, dpi=200, bbox_inches='tight')
            plt.close(fig)
        else:
            # Fallback sem cartopy (sem mapa base)
            plt.figure(figsize=(8, 6))
            mesh = plt.pcolormesh(lons, lats, grid, cmap='jet', shading='auto', vmin=vmin_eff, vmax=vmax_eff)
            plt.colorbar(mesh, label='SST (°C)')
            plt.title(f"SST (sem basemap) — {date_str}" if date_str else "SST (sem basemap)")
            plt.xlabel('Longitude'); plt.ylabel('Latitude')
            plt.savefig(out_png, dpi=200, bbox_inches='tight')
            plt.close()

        print(f"[ok] {out_png}")
        return out_png
    except Exception as exc:
        print(f"[erro] {csv_path.name}: {exc}")
        return None


def main():
    args = build_argparser().parse_args()
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    pattern = args.pattern

    files = sorted(features_dir.glob(pattern))
    if not files:
        print(f"Nenhum arquivo encontrado em {features_dir} com padrão {pattern}")
        return

    print(f"Gerando mapas de SST para {len(files)} arquivo(s) em {out_dir} ...")
    for f in files:
        plot_sst_map(f, out_dir, args.vmin, args.vmax, tuple(args.pctl))


if __name__ == "__main__":
    main()

