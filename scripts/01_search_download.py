#!/usr/bin/env python3
"""Busca e download de dados NASA via earthaccess, guiado por config.yaml.

Apenas baixa os arquivos para data/raw/. O processamento é feito pelo script 02_preprocess.py
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
MODIS_SHORT = CFG.get("datasets", {}).get("modis_l3_chl_short_name")
MODIS_PREF_RES = CFG.get("datasets", {}).get("modis_resolution", "4km")


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


def rename_downloaded_files(downloaded_paths: List[Path]) -> List[Path]:
    """Renomeia arquivos baixados com a 'data lógica' baseada apenas no nome do arquivo.
    
    - MUR: arquivo 20250927... vira 20250926... (dia anterior, pois MUR usa timestamp às 09:00 do dia seguinte)
    - MODIS: arquivo 20250926... mantém 20250926... (mesmo dia)
    
    NÃO abre os arquivos NetCDF, apenas renomeia baseado no padrão do nome.
    """
    renamed: List[Path] = []
    
    for src in downloaded_paths:
        if src.suffix.lower() != ".nc":
            renamed.append(src)
            continue
        
        name_low = src.name.lower()
        
        # Detectar tipo de arquivo
        if "sstfnd" in name_low or "mur" in name_low or "ghrsst" in name_low:
            tag = "SSTfnd-MUR"
            days_offset = -1  # MUR: dia anterior
        elif "chlor" in name_low or "chl" in name_low or "modis" in name_low or "aqua" in name_low:
            tag = "CHL-MODIS"
            days_offset = 0  # MODIS: mesmo dia
        else:
            print(f"[rename] Tipo não reconhecido, mantendo nome original: {src.name}")
            renamed.append(src)
            continue
        
        # Extrair data do nome do arquivo
        try:
            if "aqua_modis" in name_low:
                # Formato: AQUA_MODIS.20250926.L3m...
                parts = src.name.split(".")
                if len(parts) >= 2:
                    date_token = parts[1]
                else:
                    raise ValueError("Formato MODIS inválido")
            else:
                # Formato: 20250927090000-JPL-L4...
                date_token = src.name[:8]
            
            # Converter para date e aplicar offset
            file_date = dt.datetime.strptime(date_token, "%Y%m%d").date()
            logical_date = file_date + dt.timedelta(days=days_offset)
            
            # Novo nome: YYYYMMDD_TAG.nc
            new_name = f"{logical_date.strftime('%Y%m%d')}_{tag}.nc"
            new_path = src.parent / new_name
            
            # Renomear (se já existe, sobrescreve)
            if new_path.exists():
                print(f"[rename] Arquivo destino já existe, sobrescrevendo: {new_name}")
                new_path.unlink()
            
            src.rename(new_path)
            renamed.append(new_path)
            
            # Log
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
) -> int:
    """Executa a busca e download para data/raw.
       Retorna o número de arquivos baixados."""

    query: dict[str, object] = {}
    if short_name:
        query["short_name"] = short_name
    if AOI:
        west, south, east, north = AOI
        query["bounding_box"] = (west, south, east, north)

    # Aplica offset de consulta por dataset
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

    # Filtro por data (aplica sobre o range offsetado)
    results = filter_results_by_date(results, time_range[0], time_range[1])

    # Se MODIS, remove 8D e garante resolução desejada
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

    limited = results[:MAX_GRANULES]
    print(f"-> {len(limited)} granules encontrados; iniciando download...")

    # Download
    import earthaccess
    downloaded_count = 0
    
    try:
        paths = earthaccess.download(limited, str(OUT_RAW))
        if paths:
            downloaded_paths = [Path(p) for p in paths]
            print(f"[download] {len(downloaded_paths)} arquivo(s) baixado(s)")
            
            # Renomear com data lógica
            renamed_paths = rename_downloaded_files(downloaded_paths)
            downloaded_count = len(renamed_paths)
            
            print(f"\n[download] Total processado: {downloaded_count} arquivo(s) em {OUT_RAW}")
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

    # SST (MUR): consulta +1d (arquivos MUR usam timestamp do dia seguinte)
    print("\n" + "="*60)
    print("BAIXANDO SST (MUR)...")
    print("="*60)
    count = find_and_download(
        short_name=datasets.get("sst_short_name"),
        days_offset_for_query=+1,
    )
    total_downloaded += count

    # MODIS CHL: consulta na data normal
    print("\n" + "="*60)
    print("BAIXANDO CLOROFILA (MODIS)...")
    print("="*60)
    chl_short = datasets.get("modis_l3_chl_short_name")
    if chl_short:
        count = find_and_download(
            short_name=chl_short,
            days_offset_for_query=0,
        )
        total_downloaded += count

    print("\n" + "="*60)
    print(f"Download concluído! Total: {total_downloaded} arquivo(s) em {OUT_RAW}")
    print("Execute 'python scripts/02_preprocess.py' para processar os dados.")
    print("="*60)


if __name__ == "__main__":
    main()