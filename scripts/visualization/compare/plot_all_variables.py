#!/usr/bin/env python3
"""Gera painel comparativo com todas as variÃ¡veis principais.

PainÃ©is incluÃ­dos:
  - MODIS True Color (imagem real)
  - SST
  - Gradiente de SST
  - Chlorofila (PACE/MODIS coalescida)
  - SWOT SSH

Uso tÃ­pico:
  python scripts/visualization/compare/plot_all_variables.py --date 2025-09-27
"""

from __future__ import annotations

import argparse
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
HAVE_CARTOPY = True


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/features",
                        help="DiretÃ³rio com CSVs de features")
    parser.add_argument("--features-file", help="CSV especÃ­fico a usar")
    parser.add_argument("--date", help="Data (YYYY-MM-DD) para selecionar o CSV")
    parser.add_argument("--pattern", default="*_features.csv",
                        help="PadrÃ£o glob para --all (default: *_features.csv)")
    parser.add_argument("--all", action="store_true",
                        help="Gera painÃ©is para todos os CSVs que casam com --pattern")
    parser.add_argument("--truecolor-dir", default="data/compare",
                        help="DiretÃ³rio onde buscar/salvar imagens True Color")
    parser.add_argument("--out-dir", default="data/viz/compare",
                        help="DiretÃ³rio onde salvar o PNG gerado")
    parser.add_argument("--disable-download", action="store_true",
                        help="NÃ£o tentar baixar True Color automaticamente")
    parser.add_argument("--wms-base-url", default="https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
                        help="Endpoint WMS da NASA para True Color")
    parser.add_argument("--wms-layer", default="MODIS_Aqua_CorrectedReflectance_TrueColor",
                        help="Nome da camada WMS")
    parser.add_argument("--wms-size", type=int, default=1024,
                        help="Largura/altura (px) da requisiÃ§Ã£o WMS")
    parser.add_argument("--download-timeout", type=float, default=60.0,
                        help="Timeout por tentativa de download (s)")
    parser.add_argument("--download-retries", type=int, default=2,
                        help="NÃºmero de tentativas extra em caso de falha")
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
        raise ValueError("Informe --features-file ou --date")

    token = date_str.replace("-", "")
    matches = sorted(features_dir.glob(f"{token}*_features.csv"))
    if not matches:
        raise FileNotFoundError(f"Nenhum CSV encontrado para {date_str} em {features_dir}")
    return matches[0]


def coalesce_chlorophyll(df: pd.DataFrame) -> pd.Series:
    sources = []
    if "chlor_a" in df.columns:
        sources.append(df["chlor_a"])
    for col in ("chlor_a_pace", "chlor_a_modis"):
        if col in df.columns:
            sources.append(df[col])
    if not sources:
        return pd.Series(np.nan, index=df.index)
    stacked = pd.concat(sources, axis=1)
    return stacked.mean(axis=1, skipna=True)


def compute_moana_total(df: pd.DataFrame) -> Optional[pd.Series]:
    """Retorna biomassa total MOANA (cells ml^-1) ou None se disponivel."""
    if "moana_total_cells" in df.columns:
        return df["moana_total_cells"].astype(float)
    base_cols = [col for col in ("moana_prococcus_moana", "moana_syncoccus_moana", "moana_picoeuk_moana") if col in df.columns]
    if not base_cols:
        return None
    return df[base_cols].astype(float).sum(axis=1)


def pivot_grid(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = df.pivot(index="lat", columns="lon", values=column)
    grid = grid.reindex(index=np.sort(grid.index.values), columns=np.sort(grid.columns.values))
    return grid.index.values, grid.columns.values, grid.to_numpy()


def percentile_range(data: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin = np.nanpercentile(finite, low)
    vmax = np.nanpercentile(finite, high)
    if vmin == vmax:
        vmax = vmin + 1e-3
    return float(vmin), float(vmax)


def symmetric_range(data: np.ndarray, pct: float = 98.0) -> float:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    val = np.nanpercentile(np.abs(finite), pct)
    return float(val if val > 0 else 1.0)


def find_truecolor(date_iso: str, directory: Path) -> Optional[Path]:
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = directory / f"MODIS_truecolor_{date_iso}{ext}"
        if candidate.exists():
            return candidate
    matches = list(directory.glob(f"*{date_iso}*"))
    return matches[0] if matches else None


def download_truecolor(date_iso: str, directory: Path, bbox: list[float], cfg: dict) -> Optional[Path]:
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
            response = requests.get(cfg["base_url"], params=params, timeout=cfg["timeout"])
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            img.save(out_path)
            print(f"[download] True Color salva em {out_path}")
            return out_path
        except Exception as exc:
            if attempt < cfg["retries"]:
                print(f"[warn] falha ao baixar True Color ({date_iso}), tentativa {attempt + 1}: {exc}")
            else:
                print(f"[erro] nÃ£o foi possÃ­vel baixar True Color ({date_iso}): {exc}")
    return None


def ensure_truecolor(date_iso: str, directory: Path, bbox: list[float], cfg: dict) -> Optional[Path]:
    existing = find_truecolor(date_iso, directory)
    if existing:
        return existing
    if not cfg.get("enable", True):
        return None
    return download_truecolor(date_iso, directory, bbox, cfg)


def compute_bbox(df: pd.DataFrame, padding: float = 0.5) -> list[float]:
    west = float(df["lon"].min()) - padding
    east = float(df["lon"].max()) + padding
    south = float(df["lat"].min()) - padding
    north = float(df["lat"].max()) + padding
    return [west, south, east, north]


def plot_panels(df: pd.DataFrame, date_iso: str, truecolor_path: Optional[Path], out_path: Path) -> None:
    bbox = compute_bbox(df, padding=0.3)
    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]  # west, east, south, north

    # Grades
    lat_sst, lon_sst, grid_sst = pivot_grid(df, "sst") if "sst" in df.columns else (None, None, None)
    lat_grad, lon_grad, grid_grad = pivot_grid(df, "sst_gradient") if "sst_gradient" in df.columns else (None, None, None)

    chlor = coalesce_chlorophyll(df)
    df_chl = pd.concat([df[["lat", "lon"]], chlor.rename("chlor_a")], axis=1)
    lat_chl, lon_chl, grid_chl = pivot_grid(df_chl, "chlor_a")

    moana_total = compute_moana_total(df)
    if moana_total is not None:
        moana_series = moana_total.rename("moana_total_cells")
        moana_df = pd.concat([df[["lat", "lon"]], moana_series], axis=1)
        lat_moana, lon_moana, grid_moana = pivot_grid(moana_df, "moana_total_cells")
    else:
        lat_moana = lon_moana = grid_moana = None

    ssh = df.get("ssh_swot")
    swot_mask = df.get("swot_mask")
    if swot_mask is not None:
        scatter_mask = swot_mask.to_numpy(dtype=float) > 0.1
    else:
        scatter_mask = np.isfinite(ssh.to_numpy(dtype=float)) if ssh is not None else np.array([])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.ravel()

    for ax in axes:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

    # True Color
    ax_tc = axes[0]
    if truecolor_path and truecolor_path.exists():
        try:
            img = mpimg.imread(truecolor_path)
            ax_tc.imshow(img, origin="upper", extent=extent, transform=ccrs.PlateCarree())
            ax_tc.set_title(f"MODIS True Color â€” {date_iso}")
        except Exception as exc:
            ax_tc.text(0.5, 0.5, f"Erro imagem\n{exc}", ha="center", va="center")
            ax_tc.set_title("True Color indisponÃ­vel")
    else:
        ax_tc.text(0.5, 0.5, "True Color indisponÃ­vel", ha="center", va="center")
        ax_tc.set_title("True Color indisponÃ­vel")

    # SST
    ax_sst = axes[1]
    if grid_sst is not None:
        vmin_sst, vmax_sst = percentile_range(grid_sst)
        mesh = ax_sst.pcolormesh(lon_sst, lat_sst, grid_sst, cmap="turbo", shading="auto",
                                 vmin=vmin_sst, vmax=vmax_sst, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(mesh, ax=ax_sst, orientation="horizontal", pad=0.05)
        cbar.set_label("SST (Â°C)")
    ax_sst.set_title("SST")

    # Gradiente de SST
    ax_grad = axes[2]
    if grid_grad is not None:
        vmax_grad = symmetric_range(grid_grad)
        mesh = ax_grad.pcolormesh(lon_grad, lat_grad, grid_grad, cmap="RdBu_r", shading="auto",
                                  vmin=-vmax_grad, vmax=vmax_grad, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(mesh, ax=ax_grad, orientation="horizontal", pad=0.05)
        cbar.set_label("Gradiente SST (Â°C/Â°)")
    ax_grad.set_title("Gradiente SST")

    # Chlorofila
    ax_chl = axes[3]
    if grid_chl is not None:
        finite = grid_chl[np.isfinite(grid_chl) & (grid_chl > 0)]
        norm = None
        if finite.size:
            norm = mcolors.LogNorm(vmin=np.nanpercentile(finite, 5), vmax=np.nanpercentile(finite, 95))
        mesh = ax_chl.pcolormesh(lon_chl, lat_chl, grid_chl, cmap="viridis", shading="auto",
                                  norm=norm, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(mesh, ax=ax_chl, orientation="horizontal", pad=0.05)
        cbar.set_label("Chlor_a (mg mâ»Â³)")
    ax_chl.set_title("Chlorofila")

    # SWOT
    ax_swot = axes[4]
    if ssh is not None and scatter_mask.any():
        values = ssh.to_numpy(dtype=float)[scatter_mask]
        lon_swot = df["lon"].to_numpy(dtype=float)[scatter_mask]
        lat_swot = df["lat"].to_numpy(dtype=float)[scatter_mask]
        vmin_ssh, vmax_ssh = percentile_range(values)
        scatter = ax_swot.scatter(lon_swot, lat_swot, c=values, cmap="coolwarm",
                                  s=14, alpha=0.9, vmin=vmin_ssh, vmax=vmax_ssh,
                                  transform=ccrs.PlateCarree())
        cbar = plt.colorbar(scatter, ax=ax_swot, orientation="horizontal", pad=0.05)
        cbar.set_label("SSH SWOT (m)")
    ax_swot.set_title("SWOT SSH")

    # MOANA biomassa total
    ax_moana = axes[5]
    if grid_moana is not None:
        norm_moana = None
        finite_moana = grid_moana[np.isfinite(grid_moana) & (grid_moana > 0)]
        if finite_moana.size:
            norm_moana = mcolors.LogNorm(vmin=np.nanpercentile(finite_moana, 5),
                                         vmax=np.nanpercentile(finite_moana, 95))
        if HAVE_CARTOPY:
            ax_moana.set_extent(extent, crs=ccrs.PlateCarree())
            try:
                ax_moana.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
                ax_moana.add_feature(cfeature.COASTLINE, linewidth=0.5)
            except Exception:
                pass
            gl = ax_moana.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            mesh = ax_moana.pcolormesh(lon_moana, lat_moana, grid_moana, cmap="magma", shading="auto",
                                       norm=norm_moana, transform=ccrs.PlateCarree())
        else:
            mesh = ax_moana.pcolormesh(lon_moana, lat_moana, grid_moana, cmap="magma", shading="auto",
                                       norm=norm_moana)
            ax_moana.set_xlabel("Longitude")
            ax_moana.set_ylabel("Latitude")
        cbar = plt.colorbar(mesh, ax=ax_moana, orientation="horizontal", pad=0.05)
        cbar.set_label("MOANA biomassa (cells ml^-1)")
    else:
        if HAVE_CARTOPY:
            ax_moana.set_extent(extent, crs=ccrs.PlateCarree())
        ax_moana.text(0.5, 0.5, "MOANA indisponivel", ha="center", va="center")
    ax_moana.set_title("MOANA Biomassa Total")

    fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.07, wspace=0.15, hspace=0.18)
    fig.text(0.5, 0.04, "Notas: SST/Gradiente -> frentes; Chlorofila -> produtividade; SWOT -> estruturas; MOANA -> biomassa fitoplanctonica.", ha="center")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ok] Painel salvo em {out_path}")


def main() -> None:
    args = build_argparser().parse_args()
    features_dir = Path(args.features_dir)
    features_list: list[Path]
    if args.all or (not args.features_file and not args.date):
        pattern = args.pattern or "*_features.csv"
        features_list = sorted(features_dir.glob(pattern))
        if not features_list:
            raise FileNotFoundError(f"Nenhum CSV encontrado em {features_dir} com padrÃ£o {pattern}")
    else:
        features_list = [resolve_features_file(features_dir, args.features_file, args.date)]

    download_cfg = {
        "enable": not args.disable_download,
        "base_url": args.wms_base_url,
        "layer": args.wms_layer,
        "size": max(64, args.wms_size),
        "timeout": max(1e-3, args.download_timeout),
        "retries": max(0, args.download_retries),
    }

    for features_path in features_list:
        df = pd.read_csv(features_path)
        if df.empty:
            print(f"[skip] {features_path.name} vazio")
            continue
        if not {"lat", "lon"}.issubset(df.columns):
            print(f"[skip] {features_path.name} sem colunas lat/lon")
            continue

        stem_token = features_path.stem.split("_")[0]
        if len(stem_token) >= 8:
            date_iso = f"{stem_token[:4]}-{stem_token[4:6]}-{stem_token[6:8]}"
        else:
            date_iso = args.date or stem_token

        bbox = compute_bbox(df)
        truecolor_path = ensure_truecolor(date_iso, Path(args.truecolor_dir), bbox, download_cfg)
        out_path = Path(args.out_dir) / f"compare_all_variables_{date_iso}.png"
        plot_panels(df, date_iso, truecolor_path, out_path)


if __name__ == "__main__":
    main()
