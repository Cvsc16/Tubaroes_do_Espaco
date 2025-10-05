#!/usr/bin/env python3
"""Agrupa CSVs de features e calcula a media (ou mediana) por pixel lat/lon."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------- parser ----------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agrega features em janelas temporais.")
    parser.add_argument("--features-dir", default="data/features", help="Diretorio com os CSVs diarios")
    parser.add_argument("--pattern", default="*_features.csv", help="Glob para encontrar arquivos (default: *_features.csv)")
    parser.add_argument("--start", help="Data inicial (YYYY-MM-DD) do intervalo")
    parser.add_argument("--end", help="Data final (YYYY-MM-DD) do intervalo")
    parser.add_argument("--stat", choices=["mean", "median"], default="mean", help="Estatistica de agregacao")
    parser.add_argument("--out", help="Arquivo de saida. Default: data/features/AVG_<start>_<end>_<stat>.csv")
    return parser

# ---------------- helpers ----------------

def parse_date_from_stem(stem: str) -> pd.Timestamp | None:
    digits = "".join(ch for ch in stem if ch.isdigit())
    if len(digits) < 8:
        return None
    try:
        return pd.to_datetime(digits[:8], format="%Y%m%d")
    except ValueError:
        return None


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError(f"{path.name} nao possui colunas lat/lon")
    df = df.copy()
    df["source_file"] = path.name
    return df


def filter_files(files: Iterable[Path], start: pd.Timestamp | None, end: pd.Timestamp | None) -> list[tuple[pd.Timestamp, Path]]:
    selected: list[tuple[pd.Timestamp, Path]] = []
    for path in files:
        timestamp = parse_date_from_stem(path.stem)
        if timestamp is None:
            continue
        if start is not None and timestamp < start:
            continue
        if end is not None and timestamp > end:
            continue
        selected.append((timestamp, path))
    return sorted(selected, key=lambda x: x[0])


def aggregate(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "lat" not in numeric_cols:
        numeric_cols.append("lat")
    if "lon" not in numeric_cols:
        numeric_cols.append("lon")

    group = df.groupby(["lat", "lon"], as_index=False)
    if stat == "median":
        agg_df = group[numeric_cols].median()
    else:
        agg_df = group[numeric_cols].mean()

    counts = df.groupby(["lat", "lon"]).size().reset_index(name="n_samples")
    agg_df = agg_df.merge(counts, on=["lat", "lon"], how="left")
    return agg_df

# ---------------- main ----------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        raise FileNotFoundError(features_dir)

    files = sorted(features_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {features_dir} com padrao {args.pattern}")

    start = pd.to_datetime(args.start) if args.start else None
    end = pd.to_datetime(args.end) if args.end else None

    dated_files = filter_files(files, start, end)
    if not dated_files:
        raise FileNotFoundError("Nenhum CSV dentro do intervalo informado")

    dfs = []
    for timestamp, path in dated_files:
        df = load_csv(path)
        df["source_date"] = timestamp.strftime("%Y-%m-%d")
        dfs.append(df)
        print(f"[load] {path.name} ({timestamp.date()}) -> {len(df):,} linhas")

    merged = pd.concat(dfs, ignore_index=True)
    if "date" in merged.columns:
        merged = merged.drop(columns=["date"])

    agg_df = aggregate(merged, args.stat)

    agg_df["date_start"] = dated_files[0][0].strftime("%Y-%m-%d")
    agg_df["date_end"] = dated_files[-1][0].strftime("%Y-%m-%d")

    default_out = features_dir / f"AVG_{agg_df['date_start'].iat[0]}_{agg_df['date_end'].iat[0]}_{args.stat}.csv"
    out_path = Path(args.out) if args.out else default_out
    agg_df.to_csv(out_path, index=False)
    print(f"[ok] Arquivo agregado salvo em {out_path} ({len(agg_df):,} linhas, {len(dated_files)} dias, stat={args.stat})")


if __name__ == "__main__":
    main()
