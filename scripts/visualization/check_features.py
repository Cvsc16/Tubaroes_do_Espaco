#!/usr/bin/env python3
"""
Inspeção rápida dos dados de features
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"

def main():
    # Procura arquivos .csv ou .parquet
    files = list(FEAT.glob("*.csv")) + list(FEAT.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("Nenhum arquivo de features encontrado em data/features/")

    print("📂 Arquivos encontrados:")
    for f in files:
        print(" -", f.name)

    # Carrega o primeiro arquivo
    file = files[0]
    if file.suffix == ".csv":
        df = pd.read_csv(file)
    else:
        df = pd.read_parquet(file)

    print("\n✅ Carregado:", file.name)
    print("🔎 Dimensões:", df.shape)
    print("\n📑 Colunas:", df.columns.tolist())

    print("\n📊 Estatísticas:")
    print(df.describe(include="all"))

    print("\n❓ Valores ausentes por coluna:")
    print(df.isna().sum())

if __name__ == "__main__":
    main()
