#!/usr/bin/env python3
"""Gera um manifest JSON com os GeoTIFFs disponiveis em data/tiles."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[2]
TILES_DIR = ROOT / "data" / "tiles"
MANIFEST_PATH = TILES_DIR / "tiles_manifest.json"


def human_label(stem: str) -> str:
    """Converter nome do arquivo em rotulo legivel."""

    prefix = "hotspots_probability_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    try:
        timestamp = datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S")
        return timestamp.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return stem


def build_entries() -> List[dict[str, Any]]:
    if not TILES_DIR.exists():
        return []

    entries: List[dict[str, Any]] = []
    for tif_path in sorted(TILES_DIR.glob("*.tif")):
        entries.append(
            {
                "label": human_label(tif_path.stem),
                "url": f"../data/tiles/{tif_path.name}",
            }
        )
    return entries


def main() -> None:
    entries = build_entries()
    if not entries:
        print("Nenhum GeoTIFF encontrado em data/tiles. Manifesto nao gerado.")
        return

    MANIFEST_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Manifesto salvo em {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
