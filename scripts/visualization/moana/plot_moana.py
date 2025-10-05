#!/usr/bin/env python3
"""Gera mapas das variáveis MOANA presentes nos CSVs de features."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False


_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent

if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))


DEFAULT_PATTERN = "*_features.csv"
DEFAULT_OUT_DIR = Path("data/viz/moana")
MOANA_PREFIX = "moana_"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plota mapas das variáveis MOANA dos CSVs de features.")
    parser.add_argument("--features-dir", default="data/features", help="Diretório onde estão os CSVs de features")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Padrão glob para localizar os arquivos (default: *_features.csv)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Diretório de saída para os PNG gerados")
    parser.add_argument("--columns", nargs="*", default=None, help="Lista explícita de colunas MOANA a plottar (default: todas começando com 'moana_')")
    parser.add_argument("--pctl", type=float, nargs=2, default=(5.0, 95.0), help="Percentis para limitar a escala de cores (default: 5 95)")
    parser.add_argument("--cmap", default="viridis", help="Mapa de cores a utilizar (default: viridis)")
    return parser


def discover_moana_columns(df: pd.DataFrame, explicit: Sequence[str] | None) -> list[str]:
    if explicit:
        cols = [col for col in explicit if col in df.columns]
        if not cols:
            raise ValueError("Nenhuma das colunas especificadas foi encontrada no CSV.")
        return cols
    cols = [col for col in df.columns if col.startswith(MOANA_PREFIX)]
    if not cols:
        raise ValueError("Nenhuma coluna MOANA encontrada (prefixo 'moana_').")
    return cols


def compute_grid(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = df[["lat", "lon", column]].dropna()
    if subset.empty:
        raise ValueError(f"Coluna {column} sem valores válidos")
    grid = subset.pivot(index="lat", columns="lon", values=column)
    grid = grid.reindex(index=np.sort(grid.index.values), columns=np.sort(grid.columns.values))
    lats = grid.index.values
    lons = grid.columns.values
    return lats, lons, grid.values


def pick_scale(values: np.ndarray, pctl: Iterable[float]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    p_lo, p_hi = np.nanpercentile(finite, list(pctl))
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_lo >= p_hi:
        return float(finite.min()), float(finite.max())
    return float(p_lo), float(p_hi)


def create_axes(n_plots: int):
    ncols = 2 if n_plots > 1 else 1
    nrows = math.ceil(n_plots / ncols)
    figsize = (6 * ncols, 5 * nrows)
    if HAVE_CARTOPY:
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=figsize,
            squeeze=False,
        )
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    return fig, axes.flatten()


def plot_single_map(ax, lons, lats, grid, title, cmap, vmin, vmax, show_basemap: bool) -> None:
    if HAVE_CARTOPY and show_basemap:
        ax.set_extent([
            float(np.min(lons)),
            float(np.max(lons)),
            float(np.min(lats)),
            float(np.max(lats)),
        ], crs=ccrs.PlateCarree())
        try:
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        except Exception:
            pass
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        mesh = ax.pcolormesh(lons, lats, grid, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    else:
        mesh = ax.pcolormesh(lons, lats, grid, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.set_title(title)
    cb = plt.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("cells ml$^{-1}")


def plot_moana_for_csv(csv_path: Path, out_dir: Path, columns: Sequence[str] | None, cmap: str, pctl: Sequence[float]) -> Path | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[erro] Falha ao ler {csv_path.name}: {exc}")
        return None

    required = {"lat", "lon"}
    if not required.issubset(df.columns):
        print(f"[skip] {csv_path.name}: faltam colunas lat/lon")
        return None

    try:
        columns = discover_moana_columns(df, columns)
    except ValueError as exc:
        print(f"[skip] {csv_path.name}: {exc}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = create_axes(len(columns))

    date_label = ""
    if "date" in df.columns and not df["date"].isna().all():
        date_label = str(df["date"].iloc[0])

    for ax, col in zip(axes, columns):
        try:
            lats, lons, grid = compute_grid(df, col)
        except ValueError as exc:
            ax.set_visible(False)
            print(f"[warn] {csv_path.name}: {exc}")
            continue
        vmin, vmax = pick_scale(grid, pctl)
        title = col.replace("moana_", "MOANA ")
        if date_label:
            title += f" — {date_label}"
        plot_single_map(ax, lons, lats, grid, title, cmap, vmin, vmax, HAVE_CARTOPY)

    # Desativar eixos extras (quando grid maior que colunas)
    for extra_ax in axes[len(columns):]:
        extra_ax.set_visible(False)

    fig.suptitle(f"Variáveis MOANA — {csv_path.stem}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = out_dir / f"{csv_path.stem}_moana.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_png}")
    return out_png


def main() -> None:
    args = build_argparser().parse_args()
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    pattern = args.pattern

    files = sorted(features_dir.glob(pattern))
    if not files:
        print(f"Nenhum arquivo encontrado em {features_dir} com padrão {pattern}")
        return

    print(f"Gerando mapas MOANA para {len(files)} arquivo(s)...")
    for csv_path in files:
        plot_moana_for_csv(csv_path, out_dir, args.columns, args.cmap, args.pctl)


if __name__ == "__main__":
    main()
