#!/usr/bin/env python3
"""Busca e download de dados NASA via earthaccess, guiado por config.yaml."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import datetime as dt

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent

if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))

from typing import Any, Dict

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earthaccess import login, search_data
from scripts.utils import load_config, project_root


ROOT = project_root()
CFG: Dict[str, Any] = load_config()

OUT_RAW = ROOT / "data" / "raw"
OUT_RAW.mkdir(parents=True, exist_ok=True)

AOI = CFG.get("aoi", {}).get("bbox")
TIME_RANGE = (
    CFG.get("time", {}).get("start"),
    CFG.get("time", {}).get("end"),
)
MAX_GRANULES = CFG.get("processing", {}).get("max_granules_per_source", 10)
MODIS_SHORT = CFG.get("datasets", {}).get("modis_l3_chl_short_name")


def login_earthdata() -> None:
    """Efetua login via arquivo ~/.netrc."""
    login(strategy="netrc")


def filter_results_by_date(results, start: str, end: str):
    """Mantém apenas granules com data no range exato [start, end]."""
    kept = []
    start_date = dt.datetime.fromisoformat(start).date()
    end_date = dt.datetime.fromisoformat(end).date()

    for g in results:
        links = g.data_links() or []
        if not links:
            continue

        href = links[0]
        match = re.search(r"(20\d{6})", href)
        if not match:
            continue

        file_date = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
        if start_date <= file_date <= end_date:
            kept.append(g)

    return kept


def find_and_download(
    *,
    short_name: str | None = None,
    keywords: list[str] | None = None,
    collection: str | None = None,
) -> None:
    """Executa a busca e baixa os granules retornados para ``data/raw``."""

    query: dict[str, object] = {}
    if short_name:
        query["short_name"] = short_name
    if AOI:
        west, south, east, north = AOI
        query["bounding_box"] = (west, south, east, north)
    if TIME_RANGE[0] and TIME_RANGE[1]:
        query["temporal"] = TIME_RANGE
    if keywords:
        query["query"] = " ".join(keywords)
    if collection:
        query["collection_concept_id"] = collection

    print(f"[earthaccess] search: {query}")
    results = search_data(**query)
    if not results:
        print("Nenhum granule encontrado.")
        return

    # 🔎 Filtro de datas (aplica a todos os datasets)
    results = filter_results_by_date(results, TIME_RANGE[0], TIME_RANGE[1])
    preferred_res = CFG.get("datasets", {}).get("modis_resolution", "4km")

    # Caso MODIS, removemos duplicados e agregados (8D etc.)
    if short_name and short_name == MODIS_SHORT:
        filtered = []
        seen_dates = set()
        kept_dates = []
        for granule in results:
            links = granule.data_links() or []
            href = links[0] if links else ""
            if "L3m.8D" in href:  # descarta agregados 8D
                continue
            if preferred_res not in href:  # ✅ garante resolução desejada
                continue

            match = re.search(r"(20\d{6})", href)
            if not match:
                continue
            token = match.group(1)
            date_key = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
            if date_key in seen_dates:
                continue
            seen_dates.add(date_key)
            kept_dates.append(date_key)
            filtered.append(granule)

        print(f"[earthaccess] MODIS ({preferred_res}) datas mantidas: {sorted(kept_dates)}")
        results = filtered

    limited = results[:MAX_GRANULES]
    print(f"-> {len(limited)} granules filtrados; iniciando download...")

    import earthaccess

    try:
        earthaccess.download(limited, str(OUT_RAW))
    except Exception as exc:  # pragma: no cover - apenas logging
        print(f"Falha ao baixar resultados: {exc}")


def main() -> None:
    login_earthdata()
    datasets = CFG.get("datasets", {})

    # SST
    find_and_download(short_name=datasets.get("sst_short_name"))

    # MODIS
    chl_short = datasets.get("modis_l3_chl_short_name")
    if chl_short:
        find_and_download(short_name=chl_short)

    print("Concluido. Dados em data/raw/")


if __name__ == "__main__":
    main()
