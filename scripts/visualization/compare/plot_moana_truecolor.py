#!/usr/bin/env python3
"""Painel comparando MODIS True Color com variáveis MOANA."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False

import matplotlib.image as mpimg


MOANA_PREFIX = "moana_"
DEFAULT_PATTERN = "*_features.csv"
DEFAULT_OUT_DIR = Path("data/viz/compare")
DEFAULT_TRUECOLOR_DIR = Path("data/compare")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plota True Color + variáveis MOANA para um CSV de features.")
    parser.add_argument("--features-dir", default="data/features", help="Diretório dos CSVs de features")
    parser.add_argument("--features-file", help="CSV específico a utilizar")
    parser.add_argument("--date", help="Data (YYYY-MM-DD) para localizar o CSV")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob para --all (default: *_features.csv)")
    parser.add_argument("--all", action="store_true", help="Processa todos os arquivos que casam com --pattern")
    parser.add_argument("--truecolor-dir", default=str(DEFAULT_TRUECOLOR_DIR), help="Diretório com as imagens True Color")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Diretório de saída para os PNGs")
    parser.add_argument("--disable-download", action="store_true", help="Não baixar True Color automaticamente")
    parser.add_argument("--wms-base-url", default="https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi", help="Endpoint WMS")
    parser.add_argument("--wms-layer", default="MODIS_Aqua_CorrectedReflectance_TrueColor", help="Camada WMS")
    parser.add_argument("--wms-size", type=int, default=1024, help="Largura/altura da requisição WMS")
    parser.add_argument("--download-timeout", type=float, default=60.0, help="Timeout por tentativa (s)")
    parser.add_argument("--download-retries", type=int, default=2, help="Tentativas extras se o download falhar")
    parser.add_argument("--columns", nargs="*", help="Lista explícita de colunas MOANA (default: todas)")
    parser.add_argument("--pctl", type=float, nargs=2, default=(5.0, 95.0), help="Percentis min/max para a escala")
    return parser


def resolve_features_file(features_dir: Path, features_file: Optional[str], date_str: Optional[str]) -> Path:
    if features_file:
        path = Path(features_file)
        if not path.is_absolute():
            path = features_dir / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if not date_str:
        raise ValueError("Use --features-file ou --date")
    token = date_str.replace("-", "")
    matches = sorted(features_dir.glob(f"{token}*_features.csv"))
    if not matches:
        raise FileNotFoundError(f"Nenhum CSV encontrado para {date_str}")
    return matches[0]


def compute_bbox(df: pd.DataFrame, padding: float = 0.3) -> list[float]:
    west = float(df["lon"].min()) - padding
    east = float(df["lon"].max()) + padding
    south = float(df["lat"].min()) - padding
    north = float(df["lat"].max()) + padding
    return [west, south, east, north]


def find_truecolor(date_iso: str, directory: Path) -> Optional[Path]:
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = directory / f"MODIS_truecolor_{date_iso}{ext}"
        if candidate.exists():
            return candidate
    matches = list(directory.glob(f"*{date_iso}*"))
    return matches[0] if matches else None


def download_truecolor(date_iso: str, directory: Path, bbox: Sequence[float], cfg: dict) -> Optional[Path]:
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
        "WIDTH": str(cfg["size"]),
        "HEIGHT": str(cfg["size"]),
        "SRS": "EPSG:4326",
        "TIME": date_iso,
    }
    for attempt in range(cfg["retries"] + 1):
        try:
            resp = requests.get(cfg["base_url"], params=params, timeout=cfg["timeout"])
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            img.save(out_path)
            print(f"[download] True Color salva em {out_path}")
            return out_path
        except Exception as exc:
            if attempt < cfg["retries"]:
                print(f"[warn] falha download True Color ({date_iso}) tent={attempt + 1}: {exc}")
            else:
                print(f"[erro] não foi possível baixar True Color ({date_iso}): {exc}")
    return None


def ensure_truecolor(date_iso: str, directory: Path, bbox: Sequence[float], cfg: dict) -> Optional[Path]:
    existing = find_truecolor(date_iso, directory)
    if existing:
        return existing
    if not cfg.get("enable", True):
        return None
    return download_truecolor(date_iso, directory, bbox, cfg)


def pivot_grid(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = df.pivot(index="lat", columns="lon", values=column)
    grid = grid.reindex(index=np.sort(grid.index.values), columns=np.sort(grid.columns.values))
    return grid.index.values, grid.columns.values, grid.to_numpy()


def pick_scale(grid: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, low))
    vmax = float(np.nanpercentile(finite, high))
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def discover_moana_columns(df: pd.DataFrame, explicit: Sequence[str] | None) -> list[str]:
    if explicit:
        cols = [col for col in explicit if col in df.columns]
        if not cols:
            raise ValueError("Nenhuma das colunas informadas está presente no CSV")
        return cols[:3]
    cols = [col for col in df.columns if col.startswith(MOANA_PREFIX)]
    if not cols:
        raise ValueError("Nenhuma coluna MOANA encontrada")
    return cols[:3]


def plot_panels(df: pd.DataFrame, date_iso: str, truecolor_path: Optional[Path], columns: Sequence[str], pctl: Sequence[float], out_path: Path) -> None:
    bbox = compute_bbox(df)
    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]

    subplot_kw = {"projection": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw=subplot_kw, squeeze=False)
    axes = axes.flatten()

    ax_tc = axes[0]
    if HAVE_CARTOPY:
        ax_tc.set_extent(extent, crs=ccrs.PlateCarree())
        try:
            ax_tc.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
            ax_tc.add_feature(cfeature.COASTLINE, linewidth=0.6)
        except Exception:
            pass
    if truecolor_path and truecolor_path.exists():
        img = mpimg.imread(truecolor_path)
        ax_tc.imshow(img, extent=extent, origin="upper")
        ax_tc.set_title(f"MODIS True Color — {date_iso}")
    else:
        ax_tc.text(0.5, 0.5, "True Color indisponível", ha="center", va="center")
        ax_tc.set_title("True Color indisponível")
    if HAVE_CARTOPY:
        gl = ax_tc.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax_tc.set_xlabel("Longitude")
        ax_tc.set_ylabel("Latitude")

    for idx, col in enumerate(columns, start=1):
        ax = axes[idx]
        subset = df[["lat", "lon", col]].dropna()
        if subset.empty:
            ax.text(0.5, 0.5, "Dados indisponíveis", ha="center", va="center")
            ax.set_title(col)
            continue
        lat_vals, lon_vals, grid = pivot_grid(subset, col)
        vmin, vmax = pick_scale(grid, pctl[0], pctl[1])
        if HAVE_CARTOPY:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            try:
                ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            except Exception:
                pass
            mesh = ax.pcolormesh(lon_vals, lat_vals, grid, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
        else:
            mesh = ax.pcolormesh(lon_vals, lat_vals, grid, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        ax.set_title(col.replace("moana_", "MOANA "))
        cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05)
        cbar.set_label("cells ml$^{-1}")

    for extra_ax in axes[len(columns) + 1:]:
        extra_ax.remove()

    fig.suptitle(f"True Color vs MOANA — {date_iso}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[ok] Painel salvo em {out_path}")


def main() -> None:
    args = build_argparser().parse_args()
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    truecolor_dir = Path(args.truecolor_dir)

    cfg = {
        "enable": not args.disable_download,
        "base_url": args.wms_base_url,
        "layer": args.wms_layer,
        "size": max(64, args.wms_size),
        "timeout": max(1e-3, args.download_timeout),
        "retries": max(0, args.download_retries),
    }

    if args.all or (not args.features_file and not args.date):
        features_list = sorted(features_dir.glob(args.pattern or DEFAULT_PATTERN))
        if not features_list:
            raise FileNotFoundError(f"Nenhum CSV encontrado em {features_dir} com padrão {args.pattern}")
    else:
        features_list = [resolve_features_file(features_dir, args.features_file, args.date)]

    for csv_path in features_list:
        df = pd.read_csv(csv_path)
        if df.empty or not {"lat", "lon"}.issubset(df.columns):
            print(f"[skip] {csv_path.name} sem dados ou sem lat/lon")
            continue
        stem = csv_path.stem.split("_")[0]
        if len(stem) >= 8:
            date_iso = f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"
        else:
            date_iso = args.date or stem
        try:
            columns = discover_moana_columns(df, args.columns)
        except ValueError as exc:
            print(f"[skip] {csv_path.name}: {exc}")
            continue
        bbox = compute_bbox(df)
        truecolor_path = ensure_truecolor(date_iso, truecolor_dir, bbox, cfg)
        out_path = out_dir / f"compare_truecolor_moana_{date_iso}.png"
        plot_panels(df, date_iso, truecolor_path, columns, args.pctl, out_path)


if __name__ == "__main__":
    main()
