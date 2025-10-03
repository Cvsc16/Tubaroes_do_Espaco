#!/usr/bin/env python3
"""Busca e download de dados NASA via earthaccess, guiado por config.yaml.

Baixa arquivos em subpastas dentro de data/raw/{sst,modis,pace,...}.
O processamento é feito pelo script 02_preprocess.py
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import datetime as dt
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------
# Bootstrap de paths/projeto
# ---------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent

if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))

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

# datasets
MODIS_SHORT = CFG.get("datasets", {}).get("modis_l3_chl_short_name")
MODIS_PREF_RES = CFG.get("datasets", {}).get("modis_resolution", "4km")

PACE_SHORT = CFG.get("datasets", {}).get("pace_l3_chl_short_name")
PACE_RES = CFG.get("datasets", {}).get("pace_resolution", "4km")
PACE_TYPE = CFG.get("datasets", {}).get("pace_product_type", "DAY")
PACE_REALTIME = CFG.get("datasets", {}).get("pace_realtime", "NRT")


# ---------------------------------------------------------------------
# Utilitários de data
# ---------------------------------------------------------------------
def iso_to_date(s: str) -> dt.date:
    return dt.datetime.fromisoformat(s).date()


def date_to_iso(d: dt.date) -> str:
    return d.isoformat()


def apply_offset_to_range(start_iso: str, end_iso: str, days_offset: int) -> Tuple[str, str]:
    s = iso_to_date(start_iso) + dt.timedelta(days=days_offset)
    e = iso_to_date(end_iso) + dt.timedelta(days=days_offset)
    return (date_to_iso(s), date_to_iso(e))

def expected_renamed_name(href: str, tag_hint: str) -> str | None:
    """
    Retorna o nome esperado após renomear, com base no href e no tipo de dataset.
    Aplica o MESMO offset/lógica do rename_downloaded_files.
    """
    import re
    fname = Path(href).name.lower()

    # SST (MUR) -> renomeia para (data_do_arquivo - 1 dia)
    if "sstfnd" in fname or "mur" in fname or "ghrsst" in fname or tag_hint == "sst":
        m = re.search(r"(20\d{6})", fname)
        if m:
            d = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
            logical = d + dt.timedelta(days=-1)  # <- MESMO offset do rename
            return f"{logical.strftime('%Y%m%d')}_SSTfnd-MUR.nc"

    # MODIS (sem offset)
    if "modis" in fname or "aqua" in fname or "chlor" in fname or tag_hint == "modis":
        m = re.search(r"(20\d{6})", fname)
        if m:
            return f"{m.group(1)}_CHL-MODIS.nc"

        # PACE
    if "pace" in fname or "oci" in fname or tag_hint == "pace":
        parts = fname.split(".")
        if len(parts) >= 2 and re.match(r"20\d{6}", parts[1]):
            date_token = parts[1]
            return f"{date_token}_CHL-PACE.nc"


    # SWOT (usa timestamp com hora, sem offset)
    if "swot" in fname or tag_hint == "swot":
        m = re.search(r"(20\d{6}T\d{6})", fname)
        if m:
            return f"{m.group(1)}_SSH-SWOT.nc"

    return None



# ---------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------
def login_earthdata() -> None:
    """Efetua login via arquivo ~/.netrc."""
    login(strategy="netrc")


# ---------------------------------------------------------------------
# Busca e filtros
# ---------------------------------------------------------------------
def filter_results_by_date(results, start: str, end: str):
    """Mantém apenas granules com data no range [start, end] (ISO yyyy-mm-dd)."""
    kept = []
    start_date = iso_to_date(start)
    end_date = iso_to_date(end)

    for g in results:
        links = g.data_links() or []
        if not links:
            continue

        href = links[0]
        # tenta achar AAAAMMDD no link
        match = re.search(r"(20\d{6})", href)
        if not match:
            continue

        file_date = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
        if start_date <= file_date <= end_date:
            kept.append(g)

    return kept


def rename_downloaded_files(downloaded_paths: List[Path], tag_hint: str = "") -> List[Path]:
    """Renomeia arquivos baixados com a 'data lógica'.
    
    - MUR: data lógica = dia anterior
    - MODIS: data lógica = mesmo dia
    - PACE: data lógica = mesmo dia
    
    Cria nomes padronizados: YYYYMMDD_TAG.nc
    """
    renamed: List[Path] = []

    for src in downloaded_paths:
        if src.suffix.lower() != ".nc":
            renamed.append(src)
            continue

        name_low = src.name.lower()

        # Detectar tipo de arquivo
        if "sstfnd" in name_low or "mur" in name_low or "ghrsst" in name_low or tag_hint == "sst":
            tag = "SSTfnd-MUR"
            days_offset = -1
        elif "pace" in name_low or "oci" in name_low or tag_hint == "pace":
            tag = "CHL-PACE"
            days_offset = 0
        elif "chlor" in name_low or "chl" in name_low or "modis" in name_low or "aqua" in name_low or tag_hint == "modis":
            tag = "CHL-MODIS"
            days_offset = 0
        elif "swot" in name_low or tag_hint == "swot":
            tag = "SSH-SWOT"
            days_offset = 0

            match = re.search(r"(20\d{6}T\d{6})", src.name)
            if match:
                date_token = match.group(1)
                new_name = f"{date_token}_{tag}.nc"
                new_path = src.parent / new_name
                
                if new_path.exists():
                    new_path.unlink()
                
                src.rename(new_path)
                renamed.append(new_path)
                
                print(f"[rename] {tag}: {date_token} -> {new_name}")
                continue
            else:
                print(f"[rename] Não encontrei data em {src.name}, mantendo original.")
                renamed.append(src)
                continue

        else:
            print(f"[rename] Tipo não reconhecido, mantendo nome original: {src.name}")
            renamed.append(src)
            continue

        # Extrair data
        try:
            if "aqua_modis" in name_low or tag == "CHL-MODIS":
                parts = src.name.split(".")
                date_token = parts[1] if len(parts) >= 2 else src.name[:8]
            elif "pace" in name_low or "oci" in name_low or tag == "CHL-PACE":
                parts = src.name.split(".")
                date_token = parts[1] if len(parts) >= 2 else src.name[:8]
            else:
                date_token = src.name[:8]

            file_date = dt.datetime.strptime(date_token, "%Y%m%d").date()
            logical_date = file_date + dt.timedelta(days=days_offset)

            new_name = f"{logical_date.strftime('%Y%m%d')}_{tag}.nc"
            new_path = src.parent / new_name

            if new_path.exists():
                new_path.unlink()
            src.rename(new_path)
            renamed.append(new_path)

            delta_days = (logical_date - file_date).days
            sign = "+" if delta_days >= 0 else ""
            print(
                f"[rename] {tag}: {file_date.isoformat()} -> {logical_date.isoformat()} "
                f"(Δ={sign}{delta_days}d)\n"
                f"         {src.name}\n"
                f"      -> {new_name}"
            )

        except Exception as e:
            print(f"[rename] Erro ao renomear {src.name}: {e}")
            renamed.append(src)
            continue

    return renamed


def find_and_download(
    *,
    short_name: str | None = None,
    keywords: list[str] | None = None,
    collection: str | None = None,
    days_offset_for_query: int = 0,
    subdir: str = "",
    tag_hint: str = "",
) -> int:
    """Executa a busca e download para data/raw/{subdir}."""

    out_dir = OUT_RAW / subdir if subdir else OUT_RAW
    out_dir.mkdir(parents=True, exist_ok=True)

    query: dict[str, object] = {}
    if short_name:
        query["short_name"] = short_name
    if AOI:
        west, south, east, north = AOI
        query["bounding_box"] = (west, south, east, north)

    time_range = TIME_RANGE
    if TIME_RANGE[0] and TIME_RANGE[1]:
        time_range = apply_offset_to_range(TIME_RANGE[0], TIME_RANGE[1], days_offset_for_query)
        query["temporal"] = time_range
    if keywords:
        query["query"] = " ".join(keywords)
    if collection:
        query["collection_concept_id"] = collection

    print(f"[earthaccess] search: {query}")
    results = search_data(**query)
    if not results:
        print("Nenhum granule encontrado.")
        return 0

    results = filter_results_by_date(results, time_range[0], time_range[1])

    # Filtro MODIS
    if short_name and short_name == MODIS_SHORT:
        filtered = []
        seen_dates = set()
        kept_dates = []
        for granule in results:
            links = granule.data_links() or []
            href = links[0] if links else ""
            if "L3m.8D" in href:
                continue
            if MODIS_PREF_RES not in href:
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

        print(f"[earthaccess] MODIS ({MODIS_PREF_RES}) datas mantidas: {sorted(kept_dates)}")
        results = filtered

    # Filtro PACE
    if short_name and short_name == PACE_SHORT:
        filtered = []
        kept_dates = []
        for granule in results:
            links = granule.data_links() or []
            href = links[0] if links else ""
            if PACE_RES not in href:
                continue
            if PACE_TYPE not in href:
                continue
            if PACE_REALTIME and PACE_REALTIME not in href:
                continue

            match = re.search(r"(20\d{6})", href)
            if not match:
                continue
            token = match.group(1)
            date_key = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
            kept_dates.append(date_key)
            filtered.append(granule)

        print(f"[earthaccess] PACE ({PACE_RES}, {PACE_TYPE}, {PACE_REALTIME}) datas mantidas: {sorted(set(kept_dates))}")
        results = filtered

    limited = results[:MAX_GRANULES]
    print(f"-> {len(limited)} granules encontrados; iniciando verificação...")

    # 🚫 filtro: pula arquivos já renomeados
    to_download = []
    for granule in limited:
        links = granule.data_links() or []
        if not links:
            continue
        href = links[0]

        existing = {p.name.lower() for p in out_dir.glob("*.nc")}

        expected = expected_renamed_name(href, tag_hint)
        if expected and expected.lower() in existing:
            print(f"[skip] já existe: {expected}")
            continue

        to_download.append(granule)

    if not to_download:
        print("[skip] Nenhum novo arquivo para baixar (todos já existem).")
        return 0

    import earthaccess
    downloaded_count = 0

    try:
        paths = earthaccess.download(to_download, str(out_dir))
        if paths:
            downloaded_paths = [Path(p) for p in paths]
            print(f"[download] {len(downloaded_paths)} arquivo(s) baixado(s)")

            renamed_paths = rename_downloaded_files(downloaded_paths, tag_hint=tag_hint)
            downloaded_count = len(renamed_paths)

            print(f"\n[download] Total processado: {downloaded_count} arquivo(s) em {out_dir}")
        else:
            print("[download] Nenhum arquivo foi baixado.")
    except Exception as exc:
        print(f"[download] Falha ao baixar: {exc}")
        return 0

    return downloaded_count


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    login_earthdata()
    datasets = CFG.get("datasets", {})

    total_downloaded = 0

    # SST (MUR)
    print("\n" + "="*60)
    print("BAIXANDO SST (MUR)...")
    print("="*60)
    count = find_and_download(
        short_name=datasets.get("sst_short_name"),
        days_offset_for_query=+1,
        subdir="sst",
        tag_hint="sst",
    )
    total_downloaded += count

    # MODIS
    print("\n" + "="*60)
    print("BAIXANDO CLOROFILA (MODIS)...")
    print("="*60)
    chl_short = datasets.get("modis_l3_chl_short_name")
    if chl_short:
        count = find_and_download(
            short_name=chl_short,
            days_offset_for_query=0,
            subdir="modis",
            tag_hint="modis",
        )
        total_downloaded += count

    # PACE
    print("\n" + "="*60)
    print("BAIXANDO CLOROFILA (PACE OCI)...")
    print("="*60)
    pace_short = datasets.get("pace_l3_chl_short_name")
    if pace_short:
        count = find_and_download(
            short_name=pace_short,
            days_offset_for_query=0,
            subdir="pace",
            tag_hint="pace",
        )
        total_downloaded += count

    # SWOT
    print("\n" + "="*60)
    print("BAIXANDO SSH (SWOT)...")
    print("="*60)
    swot_short = datasets.get("swot_short_name")
    if swot_short:
        count = find_and_download(
            short_name=swot_short,
            days_offset_for_query=0,
            subdir="swot",
            tag_hint="swot",
        )
        total_downloaded += count    

    print("\n" + "="*60)
    print(f"Download concluído! Total: {total_downloaded} arquivo(s) em {OUT_RAW}")
    print("Execute 'python scripts/02_preprocess.py' para processar os dados.")
    print("="*60)


if __name__ == "__main__":
    main()
