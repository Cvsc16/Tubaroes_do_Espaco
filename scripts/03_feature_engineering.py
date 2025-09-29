#!/usr/bin/env python3
"""
Feature Engineering: converte arquivos NetCDF processados em tabelas tabulares
com variáveis ambientais (SST, gradientes, etc.). Agora preserva a coluna de tempo
quando disponível no NetCDF processado.
"""

from pathlib import Path
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "features"
OUT.mkdir(parents=True, exist_ok=True)


def extract_features(nc_file: Path) -> pd.DataFrame:
    """Extrai variáveis ambientais de um NetCDF processado em DataFrame tabular."""
    print(f"📂 Lendo {nc_file} ...")
    ds = xr.open_dataset(nc_file)

    if not {"sst", "sst_gradient"}.issubset(ds.variables):
        raise KeyError("Dataset processado precisa conter 'sst' e 'sst_gradient'")

    # Converte para DataFrame preservando coordenadas (lat, lon e, se houver, time)
    df = ds[["sst", "sst_gradient"]].to_dataframe().reset_index()

    # Renomeia coluna 'time' para 'date' (data/hora) se existir
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"])  # mantém precisão temporal
        df = df.drop(columns=["time"])
    else:
        # Fallback: tenta extrair data do nome do arquivo (frágil, mas útil)
        date_str = nc_file.name.split("JPL")[0][:8]
        try:
            df["date"] = pd.to_datetime(date_str, format="%Y%m%d")
        except Exception:
            pass

    return df


if __name__ == "__main__":
    files = sorted(PROC.glob("*_proc.nc"))
    if not files:
        raise FileNotFoundError("Nenhum arquivo processado em data/processed/")

    for f in files:
        try:
            df = extract_features(f)
            out_csv = OUT / f"{f.stem.replace('_proc','')}_features.csv"
            df.to_csv(out_csv, index=False)
            print(f"✅ Features salvas em {out_csv} ({len(df)} linhas)")
        except Exception as e:
            print(f"⚠️ Falha ao processar {f}: {e}")
