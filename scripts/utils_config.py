#!/usr/bin/env python3
"""
Shim de compatibilidade: mantém imports antigos
(`from utils_config import load_config, get_bbox`) ao redirecionar
para `scripts.utils.config`.
"""

from scripts.utils.config import load_config, get_bbox  # noqa: F401

