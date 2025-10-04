#!/usr/bin/env python3
"""Modelo heurístico de adequabilidade de habitat para tubarões.

O objetivo é gerar mapas rápidos, explicáveis e por espécie a partir das
features pré-processadas, agora utilizando também métricas SWOT.

Entrada: arquivos CSV em data/features/
Saídas: CSVs, mapas individuais/comparativos e relatório JSON.
"""

from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ---------------------------------------------------------------------------
# Ajuste de paths / config carregada do projeto
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


# ---------------------------------------------------------------------------
# Constantes e heurísticas
# ---------------------------------------------------------------------------
COMPONENT_WEIGHTS = {
    "thermal_gradient": 0.30,
    "chlorophyll": 0.25,
    "temperature": 0.25,
    "swot_structure": 0.12,
    "swot_polarity": 0.08,
}

SPECIES_PREFS = {
    "blue_shark": {
        "temperature_zones": [
            (None, 12, 0.1),
            (12, 15, 0.6),
            (15, 24, 1.0),
            (24, 28, 0.7),
            (28, None, 0.3),
        ],
        "ssh_preference": "warm",
    },
    "white_shark": {
        "temperature_zones": [
            (None, 10, 0.1),
            (10, 14, 0.6),
            (14, 20, 1.0),
            (20, 24, 0.8),
            (24, None, 0.4),
        ],
        "ssh_preference": "cold",
    },
    "tiger_shark": {
        "temperature_zones": [
            (None, 20, 0.2),
            (20, 23, 0.6),
            (23, 28, 1.0),
            (28, 30, 0.6),
            (30, None, 0.3),
        ],
        "ssh_preference": "warm",
    },
}

EPS = 1e-6


# ---------------------------------------------------------------------------
# Helpers de feature engineering
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
    combined = stacked.mean(axis=1, skipna=True)
    return combined.astype("float32")


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
    mask = np.isfinite(arr)
    score = np.full_like(arr, np.nan, dtype=float)
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
    if ssh is None or ssh.empty:
        return pd.Series(np.nan, index=ssh.index if ssh is not None else None, dtype=float)

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


def compute_habitat_scores(df: pd.DataFrame, species: str) -> pd.Series:
    chlor = _coalesce_chlorophyll(df)
    components = {
        "thermal_gradient": score_thermal_gradient(df.get("sst_gradient", pd.Series(index=df.index))),
        "chlorophyll": score_chlorophyll(chlor),
        "temperature": score_temperature(df.get("sst", pd.Series(index=df.index)), species),
        "swot_structure": score_swot_structure(df),
        "swot_polarity": score_swot_polarity(df.get("ssh_swot"), species),
    }

    weights_sum = np.zeros(len(df), dtype=float)
    total = np.zeros(len(df), dtype=float)

    for name, series in components.items():
        arr = series.to_numpy(dtype=float)
        weight = COMPONENT_WEIGHTS[name]
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
# Processamento por arquivo / espécie
# ---------------------------------------------------------------------------
def process_features_file(file_path: Path, species: str) -> pd.DataFrame:
    """Processa um CSV de features e calcula scores por espécie."""

    print(f"Processando {file_path.name} ({species})...")
    df = pd.read_csv(file_path)
    if df.empty:
        print(f"[warn] {file_path.name} está vazio")
        return pd.DataFrame()

    df["habitat_score"] = compute_habitat_scores(df, species)

    df["habitat_class"] = pd.cut(
        df["habitat_score"],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["poor", "moderate", "good", "excellent"],
        include_lowest=True,
    )

    n_hotspots = max(1, int(len(df) * 0.10))
    top_hotspots_idx = df["habitat_score"].nlargest(n_hotspots).index
    df["is_hotspot"] = 0
    df.loc[top_hotspots_idx, "is_hotspot"] = 1

    return df


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
        vmin=0,
        vmax=1,
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


def create_comparative_map(dfs: dict[str, pd.DataFrame], date_str: str, output_path: Path) -> None:
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
            vmin=0,
            vmax=1,
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


def generate_summary_report(df: pd.DataFrame, date_str: str, species: str) -> dict[str, object]:
    return {
        "date": date_str,
        "species": species,
        "total_points": int(len(df)),
        "mean_score": float(df["habitat_score"].mean()),
        "n_hotspots": int(df["is_hotspot"].sum()),
        "habitat_distribution": df["habitat_class"].value_counts().to_dict(),
        "top_hotspots": df.nlargest(5, "habitat_score")[["lat", "lon", "habitat_score"]]
        .round({"habitat_score": 3})
        .to_dict(orient="records"),
    }


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
        dfs_species: dict[str, pd.DataFrame] = {}

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
            print(f"  Média score: {report['mean_score']:.3f}")
            print(
                f"  Hotspots: {report['n_hotspots']} "
                f"({(report['n_hotspots'] / max(report['total_points'], 1)) * 100:.1f}%)"
            )
            for hab_class, count in report["habitat_distribution"].items():
                print(f"    {hab_class}: {count:,}")

        if dfs_species:
            out_cmp = OUTPUT_DIR / f"{date_token}_comparative_map.png"
            create_comparative_map(dfs_species, date_fmt, out_cmp)

    with open(OUTPUT_DIR / "habitat_model_report.json", "w", encoding="utf-8") as handle:
        json.dump(all_reports, handle, indent=2)

    print("\n✔ Processamento concluído!")
    print(f"Resultados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
