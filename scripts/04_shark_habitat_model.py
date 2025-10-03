#!/usr/bin/env python3
"""Modelo de adequabilidade de habitat para tubarões baseado em conhecimento ecológico.

Agora inclui 3 espécies comuns no Atlântico Norte:
- Tubarão-branco (Carcharodon carcharias)
- Tubarão-tigre (Galeocerdo cuvier)
- Tubarão-azul (Prionace glauca)

Outputs:
- CSV de predições por espécie
- Mapas individuais
- Relatórios JSON
- Figura comparativa lado a lado
"""

from __future__ import annotations
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Ajuste de path do projeto
_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent

if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))

try:
    from scripts.utils import load_config
except ModuleNotFoundError:
    from utils_config import load_config

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
OUTPUT_DIR = ROOT / "data" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CFG = load_config()

# ===============================
# SCORES POR ESPÉCIE
# ===============================
def shark_habitat_score(sst, sst_gradient, chl_modis, chl_pace, species="blue_shark"):
    """Score de habitat (0–1) para diferentes espécies de tubarão"""
    chlor_a = chl_pace if not pd.isna(chl_pace) else chl_modis
    score = 0.0

    # --- Gradiente térmico ---
    if pd.isna(sst_gradient):
        gradient_score = 0.0
    elif sst_gradient < 0.01:
        gradient_score = 0.2
    elif 0.02 <= sst_gradient <= 0.15:
        gradient_score = 1.0
    elif 0.15 < sst_gradient <= 0.30:
        gradient_score = 0.7
    else:
        gradient_score = 0.3
    score += 0.40 * gradient_score

    # --- Clorofila ---
    if pd.isna(chlor_a):
        chl_score = 0.5
    elif chlor_a < 0.05:
        chl_score = 0.3
    elif 0.1 <= chlor_a <= 2.0:
        chl_score = 1.0
    elif 2.0 < chlor_a <= 5.0:
        chl_score = 0.6
    else:
        chl_score = 0.2
    score += 0.30 * chl_score

    # --- Temperatura ótima por espécie ---
    if pd.isna(sst):
        temp_score = 0.0
    else:
        if species == "blue_shark":
            if sst < 10: temp_score = 0.2
            elif 10 <= sst < 15: temp_score = 0.6
            elif 15 <= sst <= 24: temp_score = 1.0
            elif 24 < sst <= 28: temp_score = 0.7
            else: temp_score = 0.3
        elif species == "white_shark":
            if sst < 12: temp_score = 0.2
            elif 12 <= sst < 16: temp_score = 0.7
            elif 16 <= sst <= 20: temp_score = 1.0
            elif 20 < sst <= 24: temp_score = 0.8
            else: temp_score = 0.4
        elif species == "tiger_shark":
            if sst < 18: temp_score = 0.2
            elif 18 <= sst < 22: temp_score = 0.7
            elif 22 <= sst <= 28: temp_score = 1.0
            elif 28 < sst <= 30: temp_score = 0.6
            else: temp_score = 0.3
        else:
            temp_score = 0.5
    score += 0.30 * temp_score

    return score

# ===============================
# PROCESSAMENTO
# ===============================
def process_features_file(file_path: Path, species: str) -> pd.DataFrame:
    """Processa CSV de features e calcula habitat score para a espécie"""
    print(f"Processando {file_path.name} ({species})...")
    df = pd.read_csv(file_path)

    if df.empty:
        print(f"⚠️ {file_path.name} está vazio")
        return pd.DataFrame()

    df["habitat_score"] = df.apply(
        lambda row: shark_habitat_score(
            row["sst"], row["sst_gradient"],
            row.get("chlor_a_modis", np.nan),
            row.get("chlor_a_pace", np.nan),
            species
        ), axis=1
    )

    df["habitat_class"] = pd.cut(
        df["habitat_score"],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["poor", "moderate", "good", "excellent"],
        include_lowest=True
    )

    # Hotspots fixos = top 10%
    n_hotspots = max(1, int(len(df) * 0.10))
    top_hotspots_idx = df["habitat_score"].nlargest(n_hotspots).index
    df["is_hotspot"] = 0
    df.loc[top_hotspots_idx, "is_hotspot"] = 1

    return df

def create_habitat_map(df: pd.DataFrame, date_str: str, output_path: Path, title: str):
    """Mapa individual por espécie"""
    if df.empty:
        print(f"⚠️ Sem dados para {title} em {date_str}")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([df["lon"].min(), df["lon"].max(),
                   df["lat"].min(), df["lat"].max()])
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    # Scatter do score
    scatter = ax.scatter(df["lon"], df["lat"], c=df["habitat_score"],
                         s=1, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.6)

    # Hotspots (top 10%)
    hotspots = df[df["is_hotspot"] == 1]
    ax.scatter(hotspots["lon"], hotspots["lat"],
               s=5, c="red", marker="x", alpha=0.8,
               label="Hotspots", transform=ccrs.PlateCarree())
    ax.legend(loc="upper right")
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical")
    cbar.set_label("Habitat Score (0–1)")
    ax.set_title(f"{title} - {date_str}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_comparative_map(dfs: dict, date_str: str, output_path: Path):
    """Mapa comparativo com 3 espécies lado a lado"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    titles = {"white_shark": "White Shark",
              "tiger_shark": "Tiger Shark",
              "blue_shark": "Blue Shark"}
    scatter = None
    for ax, (species, df) in zip(axes, dfs.items()):
        if df.empty: 
            continue
        ax.set_extent([df["lon"].min(), df["lon"].max(),
                       df["lat"].min(), df["lat"].max()])
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        scatter = ax.scatter(df["lon"], df["lat"], c=df["habitat_score"],
                             s=1, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.6)
        # Hotspots
        hotspots = df[df["is_hotspot"] == 1]
        ax.scatter(hotspots["lon"], hotspots["lat"],
                   s=5, c="red", marker="x", alpha=0.8,
                   label="Hotspots", transform=ccrs.PlateCarree())
        ax.legend(loc="upper right")
        ax.set_title(titles[species])
    if scatter:
        cbar = fig.colorbar(scatter, ax=axes, orientation="horizontal",
                            fraction=0.05, pad=0.1)
        cbar.set_label("Habitat Suitability Score (0–1)")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Mapa comparativo salvo em {output_path}")

def generate_summary_report(df: pd.DataFrame, date_str: str, species: str) -> dict:
    return {
        "date": date_str,
        "species": species,
        "total_points": len(df),
        "mean_score": float(df["habitat_score"].mean()),
        "n_hotspots": int(df["is_hotspot"].sum()),
        "habitat_distribution": df["habitat_class"].value_counts().to_dict(),
        "top_hotspots": df.nlargest(5, "habitat_score")[
            ["lat", "lon", "habitat_score"]].to_dict(orient="records")
    }

# ===============================
# MAIN
# ===============================
def main():
    files = sorted(FEATURES_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Nenhum arquivo em data/features/")

    all_reports = []
    species_list = ["white_shark", "tiger_shark", "blue_shark"]

    for file_path in files:
        date_str = file_path.stem.split("_")[0]
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        dfs_species = {}

        print(f"\n{'='*60}")
        print(f"🌊 MODELO DE HABITAT — {date_fmt}")
        print(f"{'='*60}")

        for species in species_list:
            df = process_features_file(file_path, species)
            if df.empty:
                continue
            dfs_species[species] = df

            # salvar CSV
            out_csv = OUTPUT_DIR / f"{date_str}_{species}_predictions.csv"
            df.to_csv(out_csv, index=False)

            # salvar mapa individual
            out_map = OUTPUT_DIR / f"{date_str}_{species}_map.png"
            create_habitat_map(df, date_fmt, out_map, species)

            # gerar relatório
            report = generate_summary_report(df, date_fmt, species)
            all_reports.append(report)

            # imprimir resumo no terminal
            print(f"\n🐋 {species.replace('_',' ').title()}")
            print(f"  Média score: {report['mean_score']:.3f}")
            print(f"  Hotspots: {report['n_hotspots']} ({(report['n_hotspots']/report['total_points']*100):.1f}%)")
            print("  Distribuição:")
            for hab_class, count in report['habitat_distribution'].items():
                print(f"    {hab_class}: {count:,}")

        # mapa comparativo
        if dfs_species:
            out_cmp = OUTPUT_DIR / f"{date_str}_comparative_map.png"
            create_comparative_map(dfs_species, date_fmt, out_cmp)

    # salvar relatório consolidado
    with open(OUTPUT_DIR / "habitat_model_report.json", "w") as f:
        json.dump(all_reports, f, indent=2)

    print(f"\n✅ Processamento concluído!")
    print(f"📂 Resultados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
