#!/usr/bin/env python3
"""Remove todos os conteúdos dentro de data/, mas mantém a estrutura de pastas."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ajusta se necessário
DATA_DIR = ROOT / "data"

def clear_data():
    if not DATA_DIR.exists():
        print(f"❌ Pasta {DATA_DIR} não existe.")
        return

    count = 0
    for sub in DATA_DIR.iterdir():
        if sub.is_dir():
            for item in sub.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        count += 1
                except Exception as e:
                    print(f"⚠️ Erro ao remover {item}: {e}")
        elif sub.is_file():
            try:
                sub.unlink()
                count += 1
            except Exception as e:
                print(f"⚠️ Erro ao remover {sub}: {e}")

    print(f"✅ Limpeza concluída! {count} itens removidos de {DATA_DIR}")

if __name__ == "__main__":
    clear_data()
