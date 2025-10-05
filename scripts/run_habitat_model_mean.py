#!/usr/bin/env python3
"""Executa o modelo heuristico usando um CSV agregado (media/mediana)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"


def load_habitat_module():
    model_path = Path(__file__).resolve().with_name("04_shark_habitat_model.py")
    spec = importlib.util.spec_from_file_location("shark_habitat_model", model_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modelo heuristico aplicado a arquivo medio")
    parser.add_argument("--features-file", required=True, help="CSV agregado gerado por aggregate_features_mean.py")
    parser.add_argument("--label", help="Titulo customizado para os mapas")
    parser.add_argument("--out-prefix", help="Prefixo extra para nomes de saida")
    return parser


def derive_label(meta_row: pd.Series, fallback: str, override: str | None) -> str:
    if override:
        return override
    if {"date_start", "date_end"}.issubset(meta_row.index):
        return f"{meta_row['date_start']} a {meta_row['date_end']} (media)"
    return fallback


def main() -> None:
    args = build_parser().parse_args()

    features_path = Path(args.features_file)
    if not features_path.is_absolute():
        candidate = FEATURES_DIR / features_path
        if candidate.exists():
            features_path = candidate
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    habitat = load_habitat_module()

    meta_head = pd.read_csv(features_path, nrows=1)
    label = derive_label(meta_head.iloc[0], features_path.stem, args.label)

    df_reports = []
    dfs_species = {}

    for species in ["white_shark", "tiger_shark", "blue_shark"]:
        df = habitat.process_features_file(features_path, species)
        if df.empty:
            continue
        dfs_species[species] = df

        stem = features_path.stem
        prefix = f"{args.out_prefix}_" if args.out_prefix else ""
        base_name = f"{prefix}{stem}_{species}"

        out_csv = habitat.OUTPUT_DIR / f"{base_name}_predictions.csv"
        df.to_csv(out_csv, index=False)

        map_label = f"{species.replace('_', ' ').title()} - {label}"
        out_map = habitat.OUTPUT_DIR / f"{base_name}_map.png"
        habitat.create_habitat_map(df, label, out_map, species.replace('_', ' ').title())

        report = habitat.generate_summary_report(df, label, species)
        report["source"] = features_path.name
        df_reports.append(report)

        print(f"[ok] {species}: {out_csv.name} | {out_map.name}")

    if dfs_species:
        stem = features_path.stem
        prefix = f"{args.out_prefix}_" if args.out_prefix else ""
        comparative = habitat.OUTPUT_DIR / f"{prefix}{stem}_comparative_map.png"
        habitat.create_comparative_map(dfs_species, label, comparative)

    if df_reports:
        reports_path = habitat.OUTPUT_DIR / "habitat_model_report_mean.json"
        with reports_path.open("w", encoding="utf-8") as f:
            json.dump(df_reports, f, indent=2)
        print(f"[ok] Relatorio salvo em {reports_path}")


if __name__ == "__main__":
    main()
