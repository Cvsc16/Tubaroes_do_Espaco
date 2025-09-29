#!/usr/bin/env python3
"""Busca e download de dados NASA via earthaccess, guiado por config.yaml."""

from pathlib import Path
import yaml
from earthaccess import login, search_data


ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "config" / "config.yaml"

CFG = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))

OUT_RAW = ROOT / "data" / "raw"
OUT_RAW.mkdir(parents=True, exist_ok=True)

AOI = CFG["aoi"]["bbox"]
TIME_RANGE = (CFG["time"]["start"], CFG["time"]["end"])
MAX_GRANULES = CFG["processing"]["max_granules_per_source"]


def login_earthdata() -> None:
    """Efetua login via arquivo ~/.netrc."""

    login(strategy="netrc")


def find_and_download(*, short_name: str | None = None, keywords: list[str] | None = None,
                      collection: str | None = None) -> None:
    """Executa a busca e baixa os granules retornados."""

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
    find_and_download(short_name=CFG["datasets"]["sst_short_name"])
    # Exemplos adicionais (descomentear conforme disponibilidade):
    # find_and_download(short_name=CFG["datasets"]["modis_l3_chl_short_name"])
    # find_and_download(keywords=CFG["datasets"]["pace_keywords"])
    print("Concluido. Dados em data/raw/")


if __name__ == "__main__":
    main()
