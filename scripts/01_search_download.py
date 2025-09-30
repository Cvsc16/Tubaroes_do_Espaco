#!/usr/bin/env python3
"""Busca e download de dados NASA via earthaccess, guiado por config.yaml."""

from __future__ import annotations

import sys
from pathlib import Path
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


def login_earthdata() -> None:
    """Efetua login via arquivo ~/.netrc."""

    login(strategy="netrc")


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

    limited = results[:MAX_GRANULES]
    print(f"-> {len(limited)} granules encontrados; iniciando download...")

    import earthaccess

    try:
        earthaccess.download(limited, str(OUT_RAW))
    except Exception as exc:  # pragma: no cover - apenas logging
        print(f"Falha ao baixar resultados: {exc}")


def main() -> None:
    login_earthdata()
    datasets = CFG.get("datasets", {})

    find_and_download(short_name=datasets.get("sst_short_name"))

    # Exemplos adicionais (descomente conforme necessario)
    # find_and_download(short_name=datasets.get("modis_l3_chl_short_name"))
    # find_and_download(keywords=datasets.get("pace_keywords"))
    # find_and_download(short_name=datasets.get("ecco_short_name"))
    # find_and_download(short_name=datasets.get("swot_short_name"))

    print("Concluido. Dados em data/raw/")


if __name__ == "__main__":
    main()
