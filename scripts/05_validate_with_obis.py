#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Valida o modelo de habitat com dados reais de ocorrência (OBIS; fallback GBIF).

Inclui:
- OBIS com paginação + retries; usa taxonid=10193 (Elasmobranchii) se não houver --species
- GBIF fallback com taxonKey=121 (Elasmobranchii)
- Janela temporal configurável (+/- days-pad) e sazonalidade (anos anteriores)
- Buffer no BBOX
- Match por raio em KM (haversine) e tolerância temporal ±1 dia (D, D-1, D+1)
- Métrica Hit@p (percentual de ocorrências dentro dos hotspots do seu CSV)
- Baseline calculado a partir do % real de hotspots nos CSVs de predição (para Lift)
- Mapa com downsample
- Flags: --species, --radius-km, --days-pad, --seasonal-years, --bbox-buffer-deg, --no-map, --heuristic-marine
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
from datetime import datetime, timedelta
import argparse
from typing import List, Tuple

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
import requests

# cartopy/matplotlib são opcionais (o script roda sem o mapa)
HAVE_CARTOPY = True
try:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    HAVE_CARTOPY = False

# Config helpers
try:
    from scripts.utils import load_config, get_bbox
except ModuleNotFoundError:
    from utils_config import load_config, get_bbox

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "data" / "predictions"
VALIDATION_DIR = ROOT / "data" / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

CFG = load_config()
DEFAULT_BBOX = get_bbox(CFG) or [-80.0, 25.0, -60.0, 40.0]  # [west, south, east, north]

# Palavras-chave para filtro heurístico de nomes (opcional)
MARINE_KEYWORDS = [
    "shark","elasmobranch","carcharodon","carcharhinus","sphyrna","prionace","isurus","lamna",
    "galeocerdo","alopias","mustelus","rhizoprionodon","ginglymostoma","mobula","manta",
    "dasyatis","rajidae","squalus","triaenodon","hexanchus","scyliorhinus","negaprion",
    "hemiscyllium","chlna","centrophorus","centroscymnus","etmopterus","dalatias","squatina"
]

# --------------------------- Utils ---------------------------

def expand_bbox(bbox: List[float], buffer_deg: float) -> List[float]:
    """Expande o BBOX em graus."""
    W, S, E, N = bbox
    return [W - buffer_deg, S - buffer_deg, E + buffer_deg, N + buffer_deg]


def build_windows(start_date: str, end_date: str, days_pad: int, seasonal_years: int) -> List[Tuple[datetime, datetime]]:
    """Cria lista de janelas temporais: base ±pad e, opcionalmente, mesmo período em anos anteriores."""
    base_start = datetime.strptime(start_date, "%Y-%m-%d")
    base_end = datetime.strptime(end_date, "%Y-%m-%d")
    pad = timedelta(days=days_pad)
    windows = [(base_start - pad, base_end + pad)]
    for k in range(1, seasonal_years + 1):
        try:
            win_start = (base_start - pad).replace(year=base_start.year - k)
            win_end = (base_end + pad).replace(year=base_end.year - k)
        except ValueError:
            # datas como 29/02 em anos não bissextos: ajusta 28/02
            win_start = (base_start - pad) - timedelta(days=1)
            win_start = win_start.replace(year=base_start.year - k)
            win_end = (base_end + pad) - timedelta(days=1)
            win_end = win_end.replace(year=base_end.year - k)
        windows.append((win_start, win_end))
    return windows


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distância Haversine em KM."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def _load_preds_for_date_or_near(pred_dir: Path, date_iso: str) -> Path | None:
    """Tenta D, depois D-1 e D+1."""
    d = pd.to_datetime(date_iso, errors="coerce")
    if pd.isna(d):
        return None
    for dd in [d, d - pd.Timedelta(days=1), d + pd.Timedelta(days=1)]:
        tag = dd.strftime("%Y%m%d")
        f = pred_dir / f"{tag}_habitat_predictions.csv"
        if f.exists():
            return f
    return None


def _looks_marine(name: str) -> bool:
    if not isinstance(name, str):
        return False
    n = name.lower()
    return any(k in n for k in MARINE_KEYWORDS)


# --------------------------- OBIS (pagina + retry) ---------------------------

def fetch_shark_occurrences(
    bbox: List[float],
    start_date: str,
    end_date: str,
    species: str | None = None
) -> pd.DataFrame:
    """
    Busca ocorrências no OBIS com paginação e retries.
    Retorna DataFrame: species, lat, lon, date, source, id
    """
    import time
    from requests.adapters import HTTPAdapter, Retry

    W, S, E, N = bbox
    url = "https://api.obis.org/occurrence"
    base_params = {
        "geometry": f"POLYGON(({W} {S},{E} {S},{E} {N},{W} {N},{W} {S}))",
        "startdate": start_date,
        "enddate": end_date,
        "fields": "decimalLongitude,decimalLatitude,eventDate,scientificName,occurrenceID",
        "size": 10000,
        "from": 0,
    }
    if species:
        # Quando o usuário especifica espécie/gênero, confie no nome científico
        base_params["scientificname"] = species
    else:
        # Sem espécie, restrinja pela Classe Elasmobranchii
        base_params["taxonid"] = 10193  # Elasmobranchii (classe)

    sess = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))

    all_rows: list[dict] = []
    total = 0

    print(f"\n{'='*60}")
    print("BUSCANDO DADOS DE TUBARÕES NO OBIS")
    print(f"{'='*60}")
    print(f"Espécie: {species or 'Elasmobranchii (classe)'}")
    print(f"Região: {W} {S}  ->  {E} {N}")
    print(f"Período: {start_date} a {end_date}\n")

    while True:
        params = dict(base_params)
        params["from"] = total
        r = sess.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            break

        for rec in results:
            lon = rec.get("decimalLongitude")
            lat = rec.get("decimalLatitude")
            if lon is None or lat is None:
                continue
            all_rows.append({
                "species": rec.get("scientificName", "Unknown"),
                "lon": float(lon),
                "lat": float(lat),
                "date": (rec.get("eventDate") or "")[:10] or None,
                "source": "OBIS",
                "id": rec.get("occurrenceID", "")
            })
        total += len(results)
        print(f"  +{len(results)} registros (total {total})")
        time.sleep(0.1)
        if len(results) < base_params["size"]:
            break

    if not all_rows:
        print("Nenhuma ocorrência encontrada nesta janela.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["lat", "lon", "date", "species"])
    print(f"\nApós filtros: {len(df)} ocorrências únicas")
    top = df["species"].value_counts().head(10)
    if len(top):
        print("Top espécies:")
        for sp, c in top.items():
            print(f"  {sp}: {c}")
    return df


# --------------------------- GBIF fallback ---------------------------

def fetch_gbif_occurrences(bbox: List[float], start_date: str, end_date: str, species: str | None = None) -> pd.DataFrame:
    """
    Fallback usando GBIF (quando OBIS não retorna). Retorna DataFrame: species, lat, lon, date, source, id
    """
    import time
    W, S, E, N = bbox
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "decimalLatitude": f"{S},{N}",
        "decimalLongitude": f"{W},{E}",
        "hasCoordinate": "true",
        "limit": 300,
        "offset": 0,
        "eventDate": f"{start_date},{end_date}",
        "taxonKey": 121,  # Elasmobranchii (classe)
    }
    if species:
        params["scientificName"] = species

    all_rows: list[dict] = []
    total = 0
    print("\n[Fallback] Consultando GBIF...")
    while True:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        res = j.get("results", [])
        if not res:
            break
        for rec in res:
            lat = rec.get("decimalLatitude"); lon = rec.get("decimalLongitude")
            if lat is None or lon is None:
                continue
            # data pode vir como ano, evento ISO, etc.
            date = rec.get("eventDate")
            if not date:
                year = rec.get("year")
                date = f"{year}-01-01" if year else None
            else:
                date = str(date)[:10]
            all_rows.append({
                "species": rec.get("scientificName", "Unknown"),
                "lon": float(lon), "lat": float(lat),
                "date": date, "source": "GBIF", "id": str(rec.get("key",""))
            })
        total += len(res)
        print(f"  +{len(res)} registros (total {total})")
        if params["offset"] + params["limit"] >= j.get("count", 0):
            break
        params["offset"] += params["limit"]
        time.sleep(0.1)
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["lat","lon","date","species"])
    print(f"[Fallback] GBIF retornou {len(df)} ocorrências únicas.")
    return df


# --------------------------- Match temporal/espacial ---------------------------

def match_occurrences_with_predictions(
    occurrences: pd.DataFrame,
    predictions_dir: Path,
    radius_km: float = 5.0
) -> pd.DataFrame:
    """Casa ocorrências com predições por raio em km, permitindo D, D-1 e D+1."""
    print(f"\n{'='*60}")
    print("FAZENDO MATCH COM PREDIÇÕES")
    print(f"{'='*60}\n")

    matches: list[dict] = []
    if occurrences.empty:
        return pd.DataFrame()

    for _, occ in occurrences.iterrows():
        date_iso = occ.get("date")
        if not date_iso:
            continue

        pred_file = _load_preds_for_date_or_near(predictions_dir, date_iso)
        if pred_file is None:
            continue

        # carregar colunas necessárias
        preds = pd.read_csv(
            pred_file,
            usecols=["lat", "lon", "habitat_score", "habitat_class", "is_hotspot"]
        )

        # filtro retangular rápido (~raio em graus)
        deg_lat = radius_km / 111.0
        deg_lon = radius_km / (111.0 * max(np.cos(np.radians(occ["lat"])), 1e-3))
        box = preds[
            preds["lat"].between(occ["lat"] - deg_lat, occ["lat"] + deg_lat)
            & preds["lon"].between(occ["lon"] - deg_lon, occ["lon"] + deg_lon)
        ]
        if box.empty:
            continue

        # haversine exata
        dist = _haversine_km(
            occ["lat"], occ["lon"],
            box["lat"].to_numpy(), box["lon"].to_numpy()
        )
        within = box.loc[dist <= radius_km]
        if within.empty:
            continue

        avg_score = float(within["habitat_score"].mean())
        max_score = float(within["habitat_score"].max())
        habitat_class = within["habitat_class"].mode().iat[0]
        any_hotspot = bool((within.get("is_hotspot", pd.Series([0]*len(within))) == 1).any())

        matches.append({
            **occ.to_dict(),
            "predicted_score": avg_score,
            "max_nearby_score": max_score,
            "habitat_class": habitat_class,
            "n_nearby_points": int(len(within)),
            "in_hotspot": int(any_hotspot),
            "pred_file": Path(pred_file).name
        })

    if not matches:
        print("Nenhum match encontrado.")
        return pd.DataFrame()

    dfm = pd.DataFrame(matches)
    print(f"Matches encontrados: {len(dfm)}")
    return dfm


# --------------------------- Relatórios e mapas ---------------------------

def compute_hotspot_baseline_from_preds(pred_dir: Path, matches_df: pd.DataFrame) -> float:
    """
    Calcula a % de hotspots (is_hotspot==1) no(s) CSV(s) de predição referenciados nos matches.
    Se houver múltiplos arquivos, calcula média ponderada pelo número de matches que referenciam cada arquivo.
    Retorna porcentagem (0-100).
    """
    if matches_df.empty or "pred_file" not in matches_df.columns:
        return float("nan")
    weights = matches_df["pred_file"].value_counts().to_dict()
    total_w = sum(weights.values()) or 1
    acc = 0.0
    for fname, w in weights.items():
        fpath = pred_dir / fname
        if not fpath.exists():
            continue
        try:
            preds = pd.read_csv(fpath, usecols=["is_hotspot"])
            if "is_hotspot" not in preds.columns or preds.empty:
                continue
            pct = float((preds["is_hotspot"] == 1).mean() * 100.0)
            acc += pct * (w / total_w)
        except Exception:
            continue
    return acc if acc > 0 else float("nan")


def generate_validation_report(matches: pd.DataFrame, baseline_pct: float | None = None) -> dict:
    """Relatório com métricas, incluindo Hit@p e Lift vs baseline."""
    if matches.empty:
        return {"error": "Sem dados para validar"}

    dist = matches["habitat_class"].value_counts().to_dict()
    hit_at_p = float(matches.get("in_hotspot", pd.Series(dtype=int)).mean() * 100.0)

    if baseline_pct is None or not np.isfinite(baseline_pct):
        baseline_pct = float("nan")
        lift = float("nan")
    else:
        lift = hit_at_p / baseline_pct if baseline_pct > 0 else float("nan")

    report = {
        "total_occurrences": int(len(matches)),
        "mean_score_at_occurrences": float(matches["predicted_score"].mean()),
        "median_score_at_occurrences": float(matches["predicted_score"].median()),
        "std_score": float(matches["predicted_score"].std()),
        "habitat_distribution": dist,
        "occurrences_in_excellent": int((matches["habitat_class"] == "excellent").sum()),
        "occurrences_in_good_or_better": int(matches["habitat_class"].isin(["good", "excellent"]).sum()),
        "percentage_in_good_or_better": float(matches["habitat_class"].isin(["good", "excellent"]).mean() * 100),
        "hit_at_hotspot_percent": hit_at_p,
        "baseline_hotspot_percent": baseline_pct,
        "lift_vs_random": lift,
        "species_summary": matches["species"].value_counts().head(5).to_dict(),
    }
    return report


def create_validation_map(
    matches: pd.DataFrame,
    predictions_file: Path,
    output_path: Path,
    downsample_target: int = 150_000
):
    """Mapa com fundo downsampleado para performance."""
    if not HAVE_CARTOPY:
        print("Cartopy/Matplotlib indisponíveis — pulando geração do mapa.")
        return

    preds = pd.read_csv(
        predictions_file,
        usecols=["lat", "lon", "habitat_score"]
    )

    # downsample para ~downsample_target pontos
    N = len(preds)
    step = max(1, N // downsample_target)
    preds_ds = preds.iloc[::step].reset_index(drop=True)

    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [preds["lon"].min(), preds["lon"].max(), preds["lat"].min(), preds["lat"].max()],
        crs=ccrs.PlateCarree()
    )
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    sc_bg = ax.scatter(
        preds_ds["lon"], preds_ds["lat"],
        c=preds_ds["habitat_score"], s=1,
        cmap="RdYlGn", vmin=0, vmax=1, alpha=0.35,
        transform=ccrs.PlateCarree(), label="Predicted Habitat Score"
    )

    sc = ax.scatter(
        matches["lon"], matches["lat"],
        c=matches["predicted_score"], s=100,
        cmap="RdYlGn", vmin=0, vmax=1,
        edgecolors="black", linewidths=1.5,
        marker="*", alpha=0.9,
        transform=ccrs.PlateCarree(),
        label="Real Shark Occurrences"
    )

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.05, shrink=0.8)
    cbar.set_label("Habitat Suitability Score", fontsize=12)

    ax.legend(loc="upper right", fontsize=10)

    date_guess = predictions_file.stem.split("_")[0]
    date_formatted = f"{date_guess[:4]}-{date_guess[4:6]}-{date_guess[6:8]}"

    ax.set_title(
        f"Model Validation — OBIS/GBIF vs Predicted Habitat\n"
        f"Pred file: {predictions_file.name} | Date ~ {date_formatted} | n={len(matches)} occurrences",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Mapa de validação salvo em {output_path}")


# --------------------------- CLI & main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Valida predições de habitat com OBIS (fallback GBIF).")
    parser.add_argument("--species", type=str, default=None, help="Nome científico (ex.: 'Prionace glauca' ou gênero 'Carcharhinus')")
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("W", "S", "E", "N"),
                        help="BBox [west south east north] (default do config)")
    parser.add_argument("--bbox-buffer-deg", type=float, default=1.0,
                        help="Buffer em graus para expandir o BBOX (default 1.0)")
    parser.add_argument("--radius-km", type=float, default=5.0, help="Raio (km) para match espacial (default 5km)")
    parser.add_argument("--days-pad", type=int, default=7, help="Dias extras em cada ponta (default 7)")
    parser.add_argument("--seasonal-years", type=int, default=0,
                        help="Se >0, busca também os mesmos dias de anos anteriores (ex.: 3)")
    parser.add_argument("--no-map", action="store_true", help="Não gerar mapa de validação")
    parser.add_argument("--heuristic-marine", action="store_true",
                        help="Aplica filtro heurístico por palavras-chave marinhas nos nomes científicos")
    args = parser.parse_args()

    # CSVs de predição disponíveis
    pred_files = sorted(PREDICTIONS_DIR.glob("*_habitat_predictions.csv"))
    if not pred_files:
        print("Nenhuma predição encontrada. Execute o modelo de habitat primeiro.")
        return

    # intervalo pelas predições
    dates_iso = []
    for f in pred_files:
        ds = f.stem.split("_")[0]
        if len(ds) >= 8:
            dates_iso.append(f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}")
    if not dates_iso:
        print("Não foi possível inferir datas a partir dos arquivos de predição.")
        return
    start_date = min(dates_iso)
    end_date = max(dates_iso)

    # janelas temporais
    windows = build_windows(start_date, end_date, args.days_pad, args.seasonal_years)

    # bbox expandido
    bbox_expanded = expand_bbox(args.bbox, args.bbox_buffer_deg)

    # OBIS (todas as janelas; concatena)
    occ_parts = []
    for (w0, w1) in windows:
        part = fetch_shark_occurrences(
            bbox=bbox_expanded,
            start_date=w0.strftime("%Y-%m-%d"),
            end_date=w1.strftime("%Y-%m-%d"),
            species=args.species
        )
        if not part.empty:
            occ_parts.append(part)

    # Fallback GBIF se nada vier do OBIS
    if not occ_parts:
        gbif_df = fetch_gbif_occurrences(
            bbox=bbox_expanded,
            start_date=windows[0][0].strftime("%Y-%m-%d"),
            end_date=windows[-1][1].strftime("%Y-%m-%d"),
            species=args.species
        )
        if gbif_df.empty:
            print("OBIS e GBIF sem ocorrências. Tente ampliar janela/área.")
            return
        occurrences = gbif_df
    else:
        occurrences = pd.concat(occ_parts, ignore_index=True).drop_duplicates()

    # Filtro heurístico (opcional) para garantir taxa marinha
    if args.heuristic_marine and "species" in occurrences.columns:
        before = len(occurrences)
        occurrences = occurrences[occurrences["species"].apply(_looks_marine)]
        print(f"Filtro marinho heurístico: {before} -> {len(occurrences)} registros")

    if occurrences.empty:
        print("Sem ocorrências após filtros.")
        return

    # Salvar ocorrências
    occ_path = VALIDATION_DIR / "shark_occurrences.csv"
    occurrences.to_csv(occ_path, index=False)
    print(f"\nOcorrências salvas em {occ_path}")

    # Match com predições
    matches = match_occurrences_with_predictions(
        occurrences, PREDICTIONS_DIR, radius_km=args.radius_km
    )

    if matches.empty:
        print("\nNenhum match encontrado (datas/área/raio podem estar muito restritos).")
        return

    # Salvar matches
    matches_path = VALIDATION_DIR / "validation_matches.csv"
    matches.to_csv(matches_path, index=False)
    print(f"Matches salvos em {matches_path}")

    # Baseline de hotspots (dos CSVs referenciados pelos matches)
    baseline_pct = compute_hotspot_baseline_from_preds(PREDICTIONS_DIR, matches)

    # Relatório
    report = generate_validation_report(matches, baseline_pct)
    report_path = VALIDATION_DIR / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("RELATÓRIO DE VALIDAÇÃO")
    print(f"{'='*60}")
    print(f"Total de ocorrências: {report['total_occurrences']}")
    print(f"Score médio nas ocorrências: {report['mean_score_at_occurrences']:.3f}")
    print(f"Score mediano: {report['median_score_at_occurrences']:.3f}")
    print("Distribuição de habitat nas ocorrências:")
    for hab, count in report["habitat_distribution"].items():
        pct = (count / report["total_occurrences"]) * 100.0
        print(f"  {hab}: {count} ({pct:.1f}%)")
    print(f"Em 'good' ou melhor: {report['percentage_in_good_or_better']:.1f}%")
    print(f"Hit@hotspot: {report['hit_at_hotspot_percent']:.1f}%")

    if np.isfinite(report.get("baseline_hotspot_percent", float('nan'))):
        print(f"Baseline (área hotspot nos CSVs): {report['baseline_hotspot_percent']:.1f}%")
    if np.isfinite(report.get("lift_vs_random", float('nan'))):
        print(f"Lift vs acaso: {report['lift_vs_random']:.2f}x")

    print(f"\nRelatório salvo em {report_path}")

    # Mapa (opcional): usar o pred_file mais frequente entre os matches
    if not args.no_map:
        if "pred_file" in matches.columns and matches["pred_file"].notna().any():
            top_pred = matches["pred_file"].mode().iat[0]
            pred_file = PREDICTIONS_DIR / top_pred
            if pred_file.exists():
                map_path = VALIDATION_DIR / f"validation_map_{top_pred.split('_')[0]}.png"
                create_validation_map(matches, pred_file, map_path)
            else:
                print("CSV de predição mais frequente não encontrado — mapa não gerado.")
        else:
            print("Coluna pred_file ausente nos matches — mapa não gerado.")

    print(f"\n{'='*60}")
    print("VALIDAÇÃO CONCLUÍDA!")
    print(f"Arquivos em: {VALIDATION_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
