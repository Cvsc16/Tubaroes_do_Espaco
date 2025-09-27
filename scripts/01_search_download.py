
#!/usr/bin/env python3
# Busca e download de dados NASA via earthaccess, com base em config.yaml

import os
from pathlib import Path
import yaml
from earthaccess import login, search_data, DataGranule

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT/"config"/"config.yaml"))

OUT_RAW = ROOT/"data"/"raw"
OUT_RAW.mkdir(parents=True, exist_ok=True)

bbox = CFG["aoi"]["bbox"]
start = CFG["time"]["start"]
end   = CFG["time"]["end"]
maxn  = CFG["processing"]["max_granules_per_source"]

def login_earthdata():
    # Usa ~/.netrc; se não existir, pedirá credenciais
    login(strategy="netrc")

def find_and_download(short_name=None, keywords=None, collection=None):
    query = {}
    if short_name:
        query["short_name"] = short_name
    if bbox:
        west, south, east, north = bbox
        query["bounding_box"] = (west, south, east, north)
    if start and end:
        query["temporal"] = (start, end)
    if keywords:
        query["query"] = " ".join(keywords)
    if collection:
        query["collection_concept_id"] = collection

    print(f"[earthaccess] search: {query}")
    results = search_data(**query)
    if not results:
        print("Nenhum granule encontrado.")
        return []

    # Limite para protótipo
    results = results[:maxn]
    print(f"→ {len(results)} granules; baixando...")
    import earthaccess
    try:
        earthaccess.download(results, str(OUT_RAW))   # ✅ baixa todos de uma vez
    except Exception as e:
        print("Falha ao baixar resultados:", e)
    return results

if __name__ == "__main__":
    login_earthdata()

    # SST MUR
    find_and_download(short_name=CFG["datasets"]["sst_short_name"])

    # MODIS L3 CHL (NRT)
    # find_and_download(short_name=CFG["datasets"]["modis_l3_chl_short_name"])

    # # PACE OCI (busca por keywords)
    # find_and_download(keywords=CFG["datasets"]["pace_keywords"])

    # # ECCO u/v
    # try:
    #     find_and_download(short_name=CFG["datasets"]["ecco_short_name"])
    # except Exception as e:
    #     print("ECCO pode exigir subset via Harmony; prossiga com SST/Chl.", e)

    # # SWOT
    # try:
    #     find_and_download(short_name=CFG["datasets"]["swot_short_name"])
    # except Exception as e:
    #     print("SWOT pode exigir autenticação/paths específicos; prossiga sem SWOT inicialmente.", e)

    print("Concluído. Dados em data/raw/")
