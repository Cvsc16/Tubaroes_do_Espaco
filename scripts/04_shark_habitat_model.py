#!/usr/bin/env python3
"""Modelo heuristico de habitat incorporando variaveis MOANA.

Gera previsoes rapidas por especie a partir dos CSVs em data/features/. O
score pondera gradiente termico, temperatura, clorofila, estrutura SWOT e
biomassa/composicao de fitoplancton derivada do PACE-MOANA.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Localizacao do projeto
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
for parent in _THIS_FILE.parents:
    if parent.name == "scripts":
        PROJECT_ROOT = parent.parent
        break
else:
    PROJECT_ROOT = _THIS_FILE.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import load_config  # type: ignore

ROOT = PROJECT_ROOT
FEATURES_DIR = ROOT / "data" / "features"
OUTPUT_DIR = ROOT / "data" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CFG = load_config()
EPS = 1e-9

# ---------------------------------------------------------------------------
# Pesos das componentes do score
# ---------------------------------------------------------------------------
COMPONENT_WEIGHTS: Dict[str, float] = {
    "thermal_gradient": 0.23,
    "temperature": 0.18,
    "chlorophyll": 0.15,
    "moana_productivity": 0.18,
    "moana_diversity": 0.10,
    "moana_composition": 0.08,
    "swot_structure": 0.05,
    "swot_polarity": 0.03,
}

# ---------------------------------------------------------------------------
# Preferencias simplificadas por especie
# ---------------------------------------------------------------------------
SPECIES_PREFS: Dict[str, Dict[str, object]] = {
    "blue_shark": {
        "temperature_zones": [
            (None, 12.0, 0.1),
            (12.0, 15.0, 0.6),
            (15.0, 24.0, 1.0),
            (24.0, 28.0, 0.7),
            (28.0, None, 0.3),
        ],
        "ssh_preference": "warm",
        "pico_ratio_range": (0.12, 0.28, 0.5, 0.72),
    },
    "white_shark": {
        "temperature_zones": [
            (None, 10.0, 0.1),
            (10.0, 14.0, 0.6),
            (14.0, 20.0, 1.0),
            (20.0, 24.0, 0.8),
            (24.0, None, 0.4),
        ],
        "ssh_preference": "cold",
        "pico_ratio_range": (0.18, 0.36, 0.62, 0.82),
    },
    "tiger_shark": {
        "temperature_zones": [
            (None, 20.0, 0.2),
            (20.0, 23.0, 0.6),
            (23.0, 28.0, 1.0),
            (28.0, 30.0, 0.6),
            (30.0, None, 0.3),
        ],
        "ssh_preference": "warm",
        "pico_ratio_range": (0.24, 0.46, 0.78, 0.94),
    },
}

# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------

def _coalesce_chlorophyll(df: pd.DataFrame) -> pd.Series:
    sources: list[pd.Series] = []
    if "chlor_a" in df.columns:
        sources.append(df["chlor_a"])
    for col in ("chlor_a_pace", "chlor_a_modis"):
        if col in df.columns:
            sources.append(df[col])
    if not sources:
        return pd.Series(np.nan, index=df.index, dtype="float32")
    stacked = pd.concat(sources, axis=1)
    return stacked.mean(axis=1, skipna=True).astype("float32")


def _piecewise_ratio(values: np.ndarray, low: float, opt_low: float, opt_high: float, high: float) -> np.ndarray:
    score = np.zeros_like(values, dtype=float)
    with np.errstate(invalid="ignore"):
        score = np.where(values <= low, 0.0, score)
        mask = (values > low) & (values < opt_low)
        score = np.where(mask, (values - low) / (opt_low - low + EPS), score)
        mask = (values >= opt_low) & (values <= opt_high)
        score = np.where(mask, 1.0, score)
        mask = (values > opt_high) & (values < high)
        score = np.where(mask, 1.0 - (values - opt_high) / (high - opt_high + EPS), score)
        score = np.where(values >= high, 0.0, score)
    return np.clip(score, 0.0, 1.0)


def score_thermal_gradient(grad: pd.Series) -> pd.Series:
    arr = grad.to_numpy(dtype=float)
    score = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return pd.Series(score, index=grad.index)

    g = arr[mask]
    component = np.full_like(g, 0.3, dtype=float)
    component[g < 0.01] = 0.2
    component[(g >= 0.01) & (g < 0.02)] = 0.6
    component[(g >= 0.02) & (g <= 0.15)] = 1.0
    component[(g > 0.15) & (g <= 0.30)] = 0.6
    component[g > 0.30] = 0.2

    score[mask] = component
    return pd.Series(score, index=grad.index)


def score_chlorophyll(chl: pd.Series) -> pd.Series:
    arr = chl.to_numpy(dtype=float)
    score = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return pd.Series(score, index=chl.index)

    c = arr[mask]
    component = np.full_like(c, 0.5, dtype=float)
    component[c < 0.05] = 0.2
    component[(c >= 0.05) & (c < 0.1)] = 0.4
    component[(c >= 0.1) & (c <= 2.0)] = 1.0
    component[(c > 2.0) & (c <= 5.0)] = 0.6
    component[c > 5.0] = 0.3

    score[mask] = component
    return pd.Series(score, index=chl.index)


def score_temperature(sst: pd.Series, species: str) -> pd.Series:
    prefs = SPECIES_PREFS.get(species, SPECIES_PREFS["blue_shark"])
    arr = sst.to_numpy(dtype=float)
    score = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return pd.Series(score, index=sst.index)

    temp_vals = arr[mask]
    component = np.full_like(temp_vals, 0.3, dtype=float)
    for low, high, value in prefs["temperature_zones"]:
        zone_mask = np.ones_like(temp_vals, dtype=bool)
        if low is not None:
            zone_mask &= temp_vals >= low
        if high is not None:
            zone_mask &= temp_vals < high
        component[zone_mask] = value

    score[mask] = component
    return pd.Series(score, index=sst.index)


def score_swot_structure(df: pd.DataFrame) -> pd.Series:
    if "ssh_swot_gradient" in df.columns:
        grad_abs = df["ssh_swot_gradient"].astype(float).abs()
    elif "ssh_swot" in df.columns:
        grad_abs = df["ssh_swot"].astype(float).diff().abs()
    else:
        return pd.Series(np.nan, index=df.index, dtype=float)

    arr = grad_abs.to_numpy(dtype=float)
    score = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return pd.Series(score, index=df.index)

    valid = arr[mask]
    q_low, q_high = np.nanquantile(valid, [0.3, 0.85])
    if q_high - q_low < EPS:
        norm = np.zeros_like(valid)
    else:
        norm = (valid - q_low) / (q_high - q_low)
        norm = np.clip(norm, 0.0, 1.0)

    score[mask] = norm
    return pd.Series(score, index=df.index)


def score_swot_polarity(ssh: pd.Series | None, species: str) -> pd.Series:
    if ssh is None:
        return pd.Series(np.nan, index=None, dtype=float)

    arr = ssh.to_numpy(dtype=float)
    score = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return pd.Series(score, index=ssh.index)

    valid = arr[mask]
    q25, q75 = np.nanquantile(valid, [0.25, 0.75])
    if q75 - q25 < EPS:
        warm = cold = np.zeros_like(valid)
    else:
        warm = np.clip((valid - q25) / (q75 - q25), 0.0, 1.0)
        cold = np.clip((q75 - valid) / (q75 - q25), 0.0, 1.0)

    pref = SPECIES_PREFS.get(species, {}).get("ssh_preference", "neutral")
    if pref == "warm":
        component = warm
    elif pref == "cold":
        component = cold
    else:
        component = np.maximum(warm, cold)

    score[mask] = component
    return pd.Series(score, index=ssh.index)

# ---------------------------------------------------------------------------
# MOANA: biomassa e composicao
# ---------------------------------------------------------------------------


def compute_moana_metrics(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in df.columns if col.startswith("moana_")]
    if not cols:
        return pd.DataFrame(index=df.index)

    data = df[cols].to_numpy(dtype=float)
    data[~np.isfinite(data)] = np.nan
    total = np.nansum(data, axis=1)

    shares = np.zeros_like(data, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = data / (total[:, None] + EPS)

    pico_idx = cols.index("moana_picoeuk_moana") if "moana_picoeuk_moana" in cols else None
    proc_idx = cols.index("moana_prococcus_moana") if "moana_prococcus_moana" in cols else None
    syn_idx = cols.index("moana_syncoccus_moana") if "moana_syncoccus_moana" in cols else None

    pico_share = shares[:, pico_idx] if pico_idx is not None else np.full_like(total, np.nan)
    cyano_share = np.zeros_like(total, dtype=float)
    if proc_idx is not None:
        cyano_share += np.nan_to_num(shares[:, proc_idx], nan=0.0)
    if syn_idx is not None:
        cyano_share += np.nan_to_num(shares[:, syn_idx], nan=0.0)

    with np.errstate(invalid="ignore"):
        entropy = -np.nansum(shares * np.log(shares + EPS), axis=1)
    max_entropy = np.log(len(cols)) if cols else 1.0
    diversity = np.clip(entropy / (max_entropy + EPS), 0.0, 1.0)

    metrics = pd.DataFrame(
        {
            "moana_total_cells": total.astype("float32"),
            "moana_picoeuk_share": pico_share.astype("float32"),
            "moana_cyanobacteria_share": cyano_share.astype("float32"),
            "moana_diversity_index": diversity.astype("float32"),
        },
        index=df.index,
    )
    return metrics


def score_moana_productivity(total_cells: pd.Series) -> pd.Series:
    arr = total_cells.to_numpy(dtype=float)
    score = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr > 0)
    if not mask.any():
        return pd.Series(score, index=total_cells.index)

    log_vals = np.log10(arr[mask] + EPS)
    p40, p90 = np.nanpercentile(log_vals, [40.0, 90.0])
    if p90 - p40 < EPS:
        norm = np.zeros_like(log_vals)
    else:
        norm = (log_vals - p40) / (p90 - p40)
    norm = np.clip(norm, 0.0, 1.0)

    score[mask] = norm
    return pd.Series(score, index=total_cells.index)


def score_moana_diversity(diversity: pd.Series) -> pd.Series:
    arr = diversity.to_numpy(dtype=float)
    score = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return pd.Series(score, index=diversity.index)

    vals = arr[mask]
    vmin = np.nanpercentile(vals, 25.0)
    vmax = np.nanpercentile(vals, 90.0)
    if vmax - vmin < EPS:
        norm = np.zeros_like(vals)
    else:
        norm = (vals - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    score[mask] = norm
    return pd.Series(score, index=diversity.index)


def score_moana_composition(pico_share: pd.Series, species: str) -> pd.Series:
    prefs = SPECIES_PREFS.get(species, SPECIES_PREFS["blue_shark"])
    arr = pico_share.to_numpy(dtype=float)
    score = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return pd.Series(score, index=pico_share.index)

    low, opt_low, opt_high, high = prefs.get("pico_ratio_range", (0.2, 0.4, 0.6, 0.8))
    score[mask] = _piecewise_ratio(arr[mask], low, opt_low, opt_high, high)
    return pd.Series(score, index=pico_share.index)

# ---------------------------------------------------------------------------
# Score agregado
# ---------------------------------------------------------------------------

def compute_habitat_scores(df: pd.DataFrame, species: str, moana_metrics: pd.DataFrame | None) -> pd.Series:
    chlor = _coalesce_chlorophyll(df)
    components = {
        "thermal_gradient": score_thermal_gradient(df.get("sst_gradient", pd.Series(index=df.index))),
        "temperature": score_temperature(df.get("sst", pd.Series(index=df.index)), species),
        "chlorophyll": score_chlorophyll(chlor),
        "swot_structure": score_swot_structure(df),
        "swot_polarity": score_swot_polarity(df.get("ssh_swot"), species),
    }

    if moana_metrics is None or moana_metrics.empty:
        moana_metrics = pd.DataFrame(index=df.index)

    if "moana_total_cells" in moana_metrics:
        components["moana_productivity"] = score_moana_productivity(moana_metrics["moana_total_cells"])
    else:
        components["moana_productivity"] = pd.Series(np.nan, index=df.index)

    if "moana_diversity_index" in moana_metrics:
        components["moana_diversity"] = score_moana_diversity(moana_metrics["moana_diversity_index"])
    else:
        components["moana_diversity"] = pd.Series(np.nan, index=df.index)

    if "moana_picoeuk_share" in moana_metrics:
        components["moana_composition"] = score_moana_composition(moana_metrics["moana_picoeuk_share"], species)
    else:
        components["moana_composition"] = pd.Series(np.nan, index=df.index)

    weights_sum = np.zeros(len(df), dtype=float)
    total = np.zeros(len(df), dtype=float)

    for name, series in components.items():
        arr = series.to_numpy(dtype=float)
        weight = COMPONENT_WEIGHTS.get(name, 0.0)
        if weight <= 0.0:
            continue
        valid = np.isfinite(arr)
        if not valid.any():
            continue
        total[valid] += arr[valid] * weight
        weights_sum[valid] += weight

    scores = np.zeros(len(df), dtype=float)
    valid_rows = weights_sum > 0
    scores[valid_rows] = total[valid_rows] / weights_sum[valid_rows]
    scores = np.clip(scores, 0.0, 1.0)
    return pd.Series(scores, index=df.index)

# ---------------------------------------------------------------------------
# Processamento por arquivo / especie
# ---------------------------------------------------------------------------

def process_features_file(file_path: Path, species: str) -> pd.DataFrame:
    print(f"Processando {file_path.name} ({species})...")
    df = pd.read_csv(file_path)
    if df.empty:
        print(f"[warn] {file_path.name} vazio")
        return pd.DataFrame()

    moana_metrics = compute_moana_metrics(df)
    for col in moana_metrics.columns:
        df[col] = moana_metrics[col]

    df["habitat_score"] = compute_habitat_scores(df, species, moana_metrics)

    df["habitat_class"] = pd.cut(
        df["habitat_score"],
        bins=[0.0, 0.3, 0.6, 0.8, 1.0],
        labels=["poor", "moderate", "good", "excellent"],
        include_lowest=True,
    )

    n_hotspots = max(1, int(len(df) * 0.10))
    top_hotspots_idx = df["habitat_score"].nlargest(n_hotspots).index
    df["is_hotspot"] = 0
    df.loc[top_hotspots_idx, "is_hotspot"] = 1

    return df

# ---------------------------------------------------------------------------
# Visualizacoes
# ---------------------------------------------------------------------------

def create_habitat_map(df: pd.DataFrame, date_str: str, output_path: Path, title: str) -> None:
    if df.empty:
        print(f"[warn] Sem dados para {title} em {date_str}")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([
        df["lon"].min(),
        df["lon"].max(),
        df["lat"].min(),
        df["lat"].max(),
    ])
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    scatter = ax.scatter(
        df["lon"],
        df["lat"],
        c=df["habitat_score"],
        s=1,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        alpha=0.6,
        transform=ccrs.PlateCarree(),
    )

    hotspots = df[df["is_hotspot"] == 1]
    if not hotspots.empty:
        ax.scatter(
            hotspots["lon"],
            hotspots["lat"],
            s=6,
            c="#00ff66",
            marker="x",
            alpha=0.9,
            label="Hotspots",
            transform=ccrs.PlateCarree(),
        )
        ax.legend(loc="upper right")

    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical")
    cbar.set_label("Habitat Score (0-1)")
    ax.set_title(f"{title} - {date_str}")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_comparative_map(dfs: Dict[str, pd.DataFrame], date_str: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    titles = {
        "white_shark": "White Shark",
        "tiger_shark": "Tiger Shark",
        "blue_shark": "Blue Shark",
    }
    scatter = None

    for ax, (species, df) in zip(axes, dfs.items()):
        if df.empty:
            continue
        ax.set_extent([
            df["lon"].min(),
            df["lon"].max(),
            df["lat"].min(),
            df["lat"].max(),
        ])
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        scatter = ax.scatter(
            df["lon"],
            df["lat"],
            c=df["habitat_score"],
            s=1,
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
        )
        hotspots = df[df["is_hotspot"] == 1]
        if not hotspots.empty:
            ax.scatter(
                hotspots["lon"],
                hotspots["lat"],
                s=6,
                c="#00ff66",
                marker="x",
                alpha=0.9,
                label="Hotspots",
                transform=ccrs.PlateCarree(),
            )
            ax.legend(loc="upper right")
        ax.set_title(titles.get(species, species))

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
        cbar.set_label("Habitat Suitability Score (0-1)")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Mapa comparativo salvo em {output_path}")

# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def generate_summary_report(df: pd.DataFrame, date_str: str, species: str) -> Dict[str, object]:
    report = {
        "date": date_str,
        "species": species,
        "total_points": int(len(df)),
        "mean_score": float(df["habitat_score"].mean()),
        "n_hotspots": int(df["is_hotspot"].sum()),
        "habitat_distribution": df["habitat_class"].value_counts().to_dict(),
        "mean_moana_total": float(df.get("moana_total_cells", pd.Series(dtype=float)).mean()),
        "mean_moana_pico_share": float(df.get("moana_picoeuk_share", pd.Series(dtype=float)).mean()),
    }
    top_cols = ["lat", "lon", "habitat_score"]
    if "moana_total_cells" in df.columns:
        top_cols.append("moana_total_cells")
    report["top_hotspots"] = (
        df.nlargest(5, "habitat_score")[top_cols]
        .round({"habitat_score": 3})
        .to_dict(orient="records")
    )
    return report

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    files = sorted(FEATURES_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Nenhum arquivo em data/features/")

    all_reports: list[dict[str, object]] = []
    species_list = ["white_shark", "tiger_shark", "blue_shark"]

    for file_path in files:
        date_token = file_path.stem.split("_")[0]
        date_fmt = f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:8]}"
        dfs_species: Dict[str, pd.DataFrame] = {}

        print("\n" + "=" * 60)
        print(f"MODELO DE HABITAT -- {date_fmt}")
        print("=" * 60)

        for species in species_list:
            df = process_features_file(file_path, species)
            if df.empty:
                continue
            dfs_species[species] = df

            out_csv = OUTPUT_DIR / f"{date_token}_{species}_predictions.csv"
            df.to_csv(out_csv, index=False)

            out_map = OUTPUT_DIR / f"{date_token}_{species}_map.png"
            create_habitat_map(df, date_fmt, out_map, species.replace("_", " ").title())

            report = generate_summary_report(df, date_fmt, species)
            all_reports.append(report)

            print(f"\n-> {species.replace('_', ' ').title()}")
            print(f"  Media score: {report['mean_score']:.3f}")
            print(
                f"  Hotspots: {report['n_hotspots']} "
                f"({(report['n_hotspots'] / max(report['total_points'], 1)) * 100:.1f}%)"
            )
            if report.get("mean_moana_total") is not None:
                print(f"  MOANA total medio: {report['mean_moana_total']:.1f} cells/ml")

        if dfs_species:
            out_cmp = OUTPUT_DIR / f"{date_token}_comparative_map.png"
            create_comparative_map(dfs_species, date_fmt, out_cmp)

    with open(OUTPUT_DIR / "habitat_model_report.json", "w", encoding="utf-8") as handle:
        json.dump(all_reports, handle, indent=2)

    print("\n[ok] Processamento concluido!")
    print(f"Resultados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
