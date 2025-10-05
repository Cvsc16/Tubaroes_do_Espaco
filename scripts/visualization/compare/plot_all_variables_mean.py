#!/usr/bin/env python3
"""Gera painel comparativo usando um CSV agregado (media/mediana) de features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Painel comparativo para CSVs agregados (media/mediana)")
    parser.add_argument("--features-file", required=True, help="Arquivo CSV agregado (ex.: AVG_*.csv)")
    parser.add_argument("--out", help="PNG de saida. Default: data/viz/compare/compare_all_variables_mean_<stem>.png")
    parser.add_argument("--title", help="Titulo do painel (override)")
    return parser


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
    if "moana_total_cells" in df.columns:
        return df["moana_total_cells"].astype(float)
    cols = [c for c in ("moana_prococcus_moana", "moana_syncoccus_moana", "moana_picoeuk_moana") if c in df.columns]
    if not cols:
        return None
    return df[cols].astype(float).sum(axis=1)


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
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def symmetric_range(data: np.ndarray, pct: float = 98.0) -> float:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    val = np.nanpercentile(np.abs(finite), pct)
    return float(val if val > 0 else 1.0)


def compute_bbox(df: pd.DataFrame, padding: float = 0.3) -> list[float]:
    west = float(df["lon"].min()) - padding
    east = float(df["lon"].max()) + padding
    south = float(df["lat"].min()) - padding
    north = float(df["lat"].max()) + padding
    return [west, south, east, north]


def plot_mean_panel(df: pd.DataFrame, label: str, out_path: Path) -> None:
    bbox = compute_bbox(df)
    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]

    lat_sst, lon_sst, grid_sst = pivot_grid(df, "sst") if "sst" in df.columns else (None, None, None)
    lat_grad, lon_grad, grid_grad = pivot_grid(df, "sst_gradient") if "sst_gradient" in df.columns else (None, None, None)

    chlor = coalesce_chlorophyll(df)
    df_chl = pd.concat([df[["lat", "lon"]], chlor.rename("chlor_a")], axis=1)
    lat_chl, lon_chl, grid_chl = pivot_grid(df_chl, "chlor_a")

    moana_total = compute_moana_total(df)
    if moana_total is not None:
        moana_df = pd.concat([df[["lat", "lon"]], moana_total.rename("moana_total_cells")], axis=1)
        lat_moana, lon_moana, grid_moana = pivot_grid(moana_df, "moana_total_cells")
    else:
        lat_moana = lon_moana = grid_moana = None

    ssh = df.get("ssh_swot")
    swot_mask = df.get("swot_mask")
    if swot_mask is not None:
        scatter_mask = swot_mask.to_numpy(dtype=float) > 0.1
    elif ssh is not None:
        scatter_mask = np.isfinite(ssh.to_numpy(dtype=float))
    else:
        scatter_mask = np.array([])

    subplot_kwargs = {"projection": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw=subplot_kwargs)
    axes = axes.ravel()

    for ax in axes:
        if HAVE_CARTOPY:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            try:
                ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
            except Exception:
                pass
            gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

    # Placeholder for True Color (in medias nao faz sentido)
    axes[0].text(0.5, 0.5, "True Color nao disponivel\n(media)", ha="center", va="center")
    axes[0].set_title("MODIS True Color (media)")

    # SST
    ax_sst = axes[1]
    if grid_sst is not None:
        vmin_sst, vmax_sst = percentile_range(grid_sst)
        mesh = ax_sst.pcolormesh(lon_sst, lat_sst, grid_sst, cmap="turbo", shading="auto", vmin=vmin_sst, vmax=vmax_sst)
        if HAVE_CARTOPY:
            mesh.set_transform(ccrs.PlateCarree())
        plt.colorbar(mesh, ax=ax_sst, orientation="horizontal", pad=0.05, label="SST (C)")
    ax_sst.set_title("SST media")

    # Gradiente
    ax_grad = axes[2]
    if grid_grad is not None:
        vmax_grad = symmetric_range(grid_grad)
        mesh = ax_grad.pcolormesh(lon_grad, lat_grad, grid_grad, cmap="RdBu_r", shading="auto", vmin=-vmax_grad, vmax=vmax_grad)
        if HAVE_CARTOPY:
            mesh.set_transform(ccrs.PlateCarree())
        plt.colorbar(mesh, ax=ax_grad, orientation="horizontal", pad=0.05, label="Gradiente SST (C/deg)")
    ax_grad.set_title("Gradiente SST medio")

    # Clorofila
    ax_chl = axes[3]
    finite = grid_chl[np.isfinite(grid_chl) & (grid_chl > 0)]
    norm = None
    if finite.size:
        norm = mcolors.LogNorm(vmin=np.nanpercentile(finite, 5), vmax=np.nanpercentile(finite, 95))
    mesh = ax_chl.pcolormesh(lon_chl, lat_chl, grid_chl, cmap="viridis", shading="auto", norm=norm)
    if HAVE_CARTOPY:
        mesh.set_transform(ccrs.PlateCarree())
    plt.colorbar(mesh, ax=ax_chl, orientation="horizontal", pad=0.05, label="Chlor_a (mg m^-3)")
    ax_chl.set_title("Clorofila media")

    # SWOT
    ax_swot = axes[4]
    if ssh is not None and scatter_mask.any():
        values = ssh.to_numpy(dtype=float)[scatter_mask]
        lon_swot = df["lon"].to_numpy(dtype=float)[scatter_mask]
        lat_swot = df["lat"].to_numpy(dtype=float)[scatter_mask]
        vmin_ssh, vmax_ssh = percentile_range(values)
        scatter = ax_swot.scatter(lon_swot, lat_swot, c=values, cmap="coolwarm", s=14, alpha=0.9, vmin=vmin_ssh, vmax=vmax_ssh, transform=ccrs.PlateCarree() if HAVE_CARTOPY else None)
        plt.colorbar(scatter, ax=ax_swot, orientation="horizontal", pad=0.05, label="SSH SWOT (m)")
    else:
        ax_swot.text(0.5, 0.5, "SWOT sem cobertura", ha="center", va="center")
    ax_swot.set_title("SWOT (media)")

    # MOANA
    ax_moana = axes[5]
    if grid_moana is not None:
        finite_moana = grid_moana[np.isfinite(grid_moana) & (grid_moana > 0)]
        norm_moana = None
        if finite_moana.size:
            norm_moana = mcolors.LogNorm(vmin=np.nanpercentile(finite_moana, 5), vmax=np.nanpercentile(finite_moana, 95))
        mesh = ax_moana.pcolormesh(lon_moana, lat_moana, grid_moana, cmap="magma", shading="auto", norm=norm_moana)
        if HAVE_CARTOPY:
            mesh.set_transform(ccrs.PlateCarree())
        plt.colorbar(mesh, ax=ax_moana, orientation="horizontal", pad=0.05, label="MOANA biomassa (cells ml^-1)")
    else:
        ax_moana.text(0.5, 0.5, "MOANA indisponivel", ha="center", va="center")
    ax_moana.set_title("MOANA Biomassa media")

    fig.suptitle(f"Variaveis combinadas - {label}", fontsize=16)
    notes = "Notas: SST/Gradiente -> frentes; Clorofila -> produtividade; SWOT -> estruturas; MOANA -> qualidade do cardapio."
    if "n_samples" in df.columns:
        notes += f" Baseado em ate {int(df['n_samples'].max())} observacoes por pixel."
    fig.text(0.5, 0.04, notes, ha="center", fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08, wspace=0.15, hspace=0.18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[ok] Painel salvo em {out_path}")


def main() -> None:
    args = build_parser().parse_args()
    features_path = Path(args.features_file)
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    df = pd.read_csv(features_path)
    if df.empty:
        raise ValueError("CSV agregado vazio")
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("CSV precisa conter as colunas lat/lon")

    if args.title:
        label = args.title
    elif {"date_start", "date_end"}.issubset(df.columns):
        label = f"{df['date_start'].iat[0]} a {df['date_end'].iat[0]} (media)"
    else:
        label = features_path.stem

    out_path = Path(args.out) if args.out else Path("data/viz/compare") / f"compare_all_variables_mean_{features_path.stem}.png"
    plot_mean_panel(df, label, out_path)


if __name__ == "__main__":
    main()
