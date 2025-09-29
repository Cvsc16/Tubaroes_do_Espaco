#!/usr/bin/env python3
"""
Utilitários simples para carregar a configuração do projeto.
Coloque neste diretório para permitir `import utils_config` a partir de scripts.
"""

from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    """Carrega e retorna o dicionário de configuração do arquivo config/config.yaml."""
    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_bbox(cfg):
    """Extrai a bbox [west, south, east, north] do dicionário de config."""
    return cfg.get("aoi", {}).get("bbox")

