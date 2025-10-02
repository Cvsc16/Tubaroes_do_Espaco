#!/usr/bin/env python3
"""Modelo de adequabilidade de habitat para tubarões baseado em conhecimento ecológico.

Calcula um score de 0-1 para cada ponto oceânico baseado em:
- Gradientes de SST (frentes térmicas agregam presas)
- Clorofila (produtividade primária)
- Temperatura adequada para a espécie

Referências científicas:
- Tubarão azul (Prionace glauca): prefere SST 10-25°C, frentes térmicas
- Gradientes fortes (0.02-0.15 °C/km) indicam zonas de convergência
- Clorofila moderada (0.1-2.0 mg/m³) indica produtividade sem blooms excessivos
"""

from __future__ import annotations

from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent

if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    from scripts.utils import load_config
except ModuleNotFoundError:
    from utils_config import load_config

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
OUTPUT_DIR = ROOT / "data" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CFG = load_config()


def shark_habitat_score(sst, sst_gradient, chlor_a, species="blue_shark"):
    """
    Calcula score de adequabilidade de habitat (0-1).
    
    Parâmetros baseados em literatura para tubarão azul (Prionace glauca):
    - SST ótima: 15-24°C (zona temperada/subtropical)
    - Gradiente: 0.02-0.15°C/km (frentes térmicas fortes)
    - Clorofila: 0.1-2.0 mg/m³ (produtividade moderada)
    
    Args:
        sst: Temperatura superficial (°C)
        sst_gradient: Magnitude do gradiente (°C/km)
        chlor_a: Concentração de clorofila (mg/m³)
        species: Espécie alvo (future: diferentes parâmetros por espécie)
    
    Returns:
        float: Score de 0 (inadequado) a 1 (ótimo)
    """
    score = 0.0
    
    # Componente 1: Gradiente de temperatura (40% do score)
    # Frentes térmicas agregam presas (zooplâncton, peixes)
    if pd.isna(sst_gradient):
        gradient_score = 0.0
    elif sst_gradient < 0.01:
        gradient_score = 0.2  # Águas muito homogêneas = pouca agregação
    elif 0.02 <= sst_gradient <= 0.15:
        gradient_score = 1.0  # Zona ótima
    elif 0.15 < sst_gradient <= 0.30:
        gradient_score = 0.7  # Ainda bom, mas muito intenso
    else:
        gradient_score = 0.3  # Gradientes extremos podem ser instáveis
    
    score += 0.40 * gradient_score
    
    # Componente 2: Clorofila (30% do score)
    # Base da cadeia trófica, mas tubarões evitam blooms excessivos
    if pd.isna(chlor_a):
        chl_score = 0.5  # Penalidade menor que gradiente ausente
    elif chlor_a < 0.05:
        chl_score = 0.3  # Oligotrófico demais
    elif 0.1 <= chlor_a <= 2.0:
        chl_score = 1.0  # Zona ótima de produtividade
    elif 2.0 < chlor_a <= 5.0:
        chl_score = 0.6  # Produtivo mas começando a ficar turvo
    else:
        chl_score = 0.2  # Bloom excessivo = água turva
    
    score += 0.30 * chl_score
    
    # Componente 3: Temperatura adequada (30% do score)
    # Tubarão azul é mesopelágico, prefere temperaturas temperadas
    if pd.isna(sst):
        temp_score = 0.0
    elif sst < 10:
        temp_score = 0.2  # Muito frio
    elif 10 <= sst < 15:
        temp_score = 0.6  # Limite inferior aceitável
    elif 15 <= sst <= 24:
        temp_score = 1.0  # Zona ótima
    elif 24 < sst <= 28:
        temp_score = 0.7  # Limite superior aceitável
    else:
        temp_score = 0.3  # Tropical demais
    
    score += 0.30 * temp_score
    
    return score


def process_features_file(file_path: Path) -> pd.DataFrame:
    """Processa um arquivo de features e calcula habitat scores."""
    print(f"Processando {file_path.name}...")
    
    # Carregar apenas colunas necessárias
    df = pd.read_csv(
        file_path,
        usecols=["date", "lat", "lon", "sst", "sst_gradient", "chlor_a"],
        dtype={
            "lat": "float32",
            "lon": "float32",
            "sst": "float32",
            "sst_gradient": "float32",
            "chlor_a": "float32"
        }
    )
    
    # Calcular score de habitat
    df["habitat_score"] = df.apply(
        lambda row: shark_habitat_score(
            row["sst"],
            row["sst_gradient"],
            row["chlor_a"]
        ),
        axis=1
    )
    
    # Classificar em categorias
    df["habitat_class"] = pd.cut(
        df["habitat_score"],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["poor", "moderate", "good", "excellent"],
        include_lowest=True
    )
    
    # Identificar top 10% como hotspots
    threshold = df["habitat_score"].quantile(0.90)
    df["is_hotspot"] = (df["habitat_score"] >= threshold).astype(int)
    
    return df


def create_habitat_map(df: pd.DataFrame, date_str: str, output_path: Path):
    """Cria mapa de calor mostrando adequabilidade de habitat."""
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Configurar mapa
    ax.set_extent(
        [df["lon"].min(), df["lon"].max(), df["lat"].min(), df["lat"].max()],
        crs=ccrs.PlateCarree()
    )
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    
    # Scatter plot com cores baseadas no score
    scatter = ax.scatter(
        df["lon"],
        df["lat"],
        c=df["habitat_score"],
        s=1,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        alpha=0.6,
        transform=ccrs.PlateCarree()
    )
    
    # Marcar hotspots
    hotspots = df[df["is_hotspot"] == 1]
    ax.scatter(
        hotspots["lon"],
        hotspots["lat"],
        s=5,
        c='red',
        marker='x',
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        label='Hotspots (top 10%)'
    )
    
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.05, shrink=0.8)
    cbar.set_label("Shark Habitat Suitability Score", fontsize=12)
    
    ax.legend(loc='upper right')
    ax.set_title(
        f"Predicted Shark Foraging Habitat - {date_str}\n"
        f"Based on SST, Thermal Fronts, and Chlorophyll-a",
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Mapa salvo em {output_path}")


def generate_summary_report(df: pd.DataFrame, date_str: str) -> dict:
    """Gera estatísticas resumidas do modelo."""
    report = {
        "date": date_str,
        "total_points": len(df),
        "mean_score": float(df["habitat_score"].mean()),
        "median_score": float(df["habitat_score"].median()),
        "std_score": float(df["habitat_score"].std()),
        "n_hotspots": int(df["is_hotspot"].sum()),
        "hotspot_percentage": float((df["is_hotspot"].sum() / len(df)) * 100),
        "habitat_distribution": df["habitat_class"].value_counts().to_dict(),
        "top_hotspot_locations": df.nlargest(10, "habitat_score")[
            ["lat", "lon", "habitat_score", "sst", "sst_gradient", "chlor_a"]
        ].to_dict(orient="records")
    }
    return report


def main():
    files = sorted(FEATURES_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            "Nenhum arquivo de features encontrado em data/features/. "
            "Execute 03_feature_engineering.py primeiro."
        )
    
    print(f"\n{'='*60}")
    print(f"MODELO DE HABITAT DE TUBARÕES")
    print(f"Processando {len(files)} arquivo(s)...")
    print(f"{'='*60}\n")
    
    all_reports = []
    
    for file_path in files:
        try:
            # Processar features
            df = process_features_file(file_path)
            
            # Extrair data do nome do arquivo
            date_str = file_path.stem.split("_")[0]  # 20250926_features -> 20250926
            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            # Salvar predições
            output_csv = OUTPUT_DIR / f"{date_str}_habitat_predictions.csv"
            df.to_csv(output_csv, index=False)
            print(f"Predições salvas em {output_csv}")
            
            # Criar mapa
            map_path = OUTPUT_DIR / f"{date_str}_habitat_map.png"
            create_habitat_map(df, date_formatted, map_path)
            
            # Gerar relatório
            report = generate_summary_report(df, date_formatted)
            all_reports.append(report)
            
            # Imprimir estatísticas
            print(f"\nEstatísticas para {date_formatted}:")
            print(f"  Score médio: {report['mean_score']:.3f}")
            print(f"  Hotspots identificados: {report['n_hotspots']:,} ({report['hotspot_percentage']:.1f}%)")
            print(f"  Distribuição de habitat:")
            for hab_class, count in report['habitat_distribution'].items():
                print(f"    {hab_class}: {count:,}")
            print()
            
        except Exception as e:
            print(f"Erro ao processar {file_path.name}: {e}")
            continue
    
    # Salvar relatório consolidado
    import json
    report_path = OUTPUT_DIR / "habitat_model_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"CONCLUÍDO!")
    print(f"  Predições: {OUTPUT_DIR}")
    print(f"  Relatório: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()