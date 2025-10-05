#!/usr/bin/env python3
"""Treina um modelo binário simples a partir das features geradas."""

from __future__ import annotations

from pathlib import Path
import sys
import json
import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as SkGradientBoosting
    HAVE_XGB = False


_THIS_FILE = Path(__file__).resolve()
for parent in _THIS_FILE.parents:
    if parent.name == "scripts":
        PROJECT_ROOT = parent.parent
        break
else:
    PROJECT_ROOT = _THIS_FILE.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import load_config  # type: ignore


ROOT = PROJECT_ROOT
FEATURES_DIR = ROOT / "data" / "features"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CFG = load_config()
TOP_PERCENT = CFG.get("model", {}).get("top_percent_for_hit", 20)
BASE_FEATURES = ["sst", "sst_gradient", "chlor_a_combined"]
OPTIONAL_FEATURES = ["ssh_swot", "ssh_swot_gradient", "swot_mask"]


def load_features() -> pd.DataFrame:
    files = sorted(FEATURES_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            "Nenhum arquivo de features encontrado em data/features/. Rode 03_feature_engineering primeiro."
        )

    print("\n" + "=" * 60)
    print(f"Carregando {len(files)} arquivo(s) de features...")
    print("=" * 60 + "\n")

    frames: list[pd.DataFrame] = []
    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[erro] ao ler {csv_path.name}: {exc}")
            continue

        if df.empty:
            print(f"[warn] {csv_path.name} vazio, ignorando...")
            continue

        required = {"lat", "lon", "sst", "sst_gradient"}
        missing = required - set(df.columns)
        if missing:
            print(f"[warn] {csv_path.name} sem colunas obrigatórias {missing}, ignorando...")
            continue

        numeric_cols = [
            "lat",
            "lon",
            "sst",
            "sst_gradient",
            "chlor_a",
            "chlor_a_modis",
            "chlor_a_pace",
            "ssh_swot",
            "ssh_swot_gradient",
            "swot_mask",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype("float32")

        df["source_file"] = csv_path.name
        frames.append(df)
        print(f"[ok] {csv_path.name}: {len(df):,} linhas")

    if not frames:
        raise ValueError("Nenhum CSV válido encontrado para treinamento.")

    print(f"\nConcatenando {len(frames)} DataFrames...")
    data = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Combinação de clorofila (PACE/MODIS)
    chlor_sources = []
    for col in ("chlor_a", "chlor_a_pace", "chlor_a_modis"):
        if col in data.columns:
            chlor_sources.append(data[col])
    if chlor_sources:
        stacked = pd.concat(chlor_sources, axis=1)
        data["chlor_a_combined"] = stacked.mean(axis=1, skipna=True).astype("float32")
    else:
        data["chlor_a_combined"] = np.nan

    print(f"[ok] Total consolidado: {len(data):,} linhas")
    return data


def apply_labels(df: pd.DataFrame, top_percent: float) -> pd.DataFrame:
    if not 0 < top_percent < 100:
        top_percent = 20

    print(f"\nAplicando labels (top {top_percent}% de gradiente)...")

    def label_group(group: pd.DataFrame) -> pd.DataFrame:
        valid = group["sst_gradient"].dropna()
        if valid.empty:
            group["label"] = 0
            return group
        threshold = np.percentile(valid, 100 - top_percent)
        group["label"] = (group["sst_gradient"] >= threshold).astype(int)
        return group

    if "date" in df.columns:
        df = df.groupby(df["date"].fillna(pd.Timestamp("1970-01-01")), group_keys=False).apply(label_group)
    else:
        df = label_group(df)

    positives = int(df["label"].sum())
    negatives = len(df) - positives
    print(f"[ok] Labels aplicados: {positives:,} positivos | {negatives:,} negativos")
    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    features: list[str] = [col for col in BASE_FEATURES if col in df.columns]
    for opt in OPTIONAL_FEATURES:
        if opt in df.columns and df[opt].notna().any():
            features.append(opt)
    if not features:
        raise ValueError("Nenhuma feature disponível para o treinamento.")
    return features


def prepare_dataset() -> tuple[pd.DataFrame, list[str]]:
    data = load_features()
    feature_cols = select_feature_columns(data)
    print(f"Features selecionadas: {feature_cols}")

    total_rows = len(data)
    for col in feature_cols:
        nan_count = int(data[col].isna().sum())
        nan_pct = (nan_count / total_rows) * 100
        print(f"  {col}: {nan_count:,} NaN ({nan_pct:.1f}%)")

    print("\nRemovendo linhas com NaN nas features...")
    data = data.dropna(subset=feature_cols)
    if data.empty:
        raise ValueError("Sem dados válidos após remover NaNs das features.")
    print(f"[ok] Restaram {len(data):,} linhas ({(len(data)/total_rows)*100:.1f}% do total)")

    data = apply_labels(data, TOP_PERCENT)
    if data["label"].nunique() < 2:
        raise ValueError("Rotulagem gerou apenas uma classe. Ajuste model.top_percent_for_hit.")

    if "date" in data.columns:
        data["date"] = data["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    dataset_cols = [col for col in ["lat", "lon", "date", *feature_cols, "label", "source_file"] if col in data.columns]
    dataset_path = PROCESSED_DIR / "dataset.csv"
    print("\nSalvando dataset consolidado...")
    data[dataset_cols].to_csv(dataset_path, index=False)
    print(f"[ok] Dataset salvo em {dataset_path} ({len(data):,} linhas)")

    return data, feature_cols


def train_model(data: pd.DataFrame, feature_cols: list[str]) -> None:
    print("\n" + "=" * 60)
    print("Treinando modelo...")
    print("=" * 60 + "\n")

    X = data[feature_cols].values
    y = data["label"].astype(int).values

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class balance: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    if HAVE_XGB:
        print("Usando XGBoost...")
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
    else:
        print("[warn] xgboost não encontrado; usando GradientBoostingClassifier...")
        model = SkGradientBoosting(random_state=42, n_estimators=100)

    print("\nTreinando...")
    model.fit(X_train, y_train)
    print("[ok] Treinamento concluído!")

    print("\nAvaliando modelo...")
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    model_path = PROCESSED_DIR / "model_xgb.pkl"
    joblib.dump(model, model_path)
    print(f"[ok] Modelo salvo em {model_path}")

    metrics_path = PROCESSED_DIR / "metrics.json"
    metrics = {
        "AUC": float(auc),
        "AveragePrecision": float(ap),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "feature_columns": feature_cols,
        "top_percent": float(TOP_PERCENT),
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[ok] Métricas salvas em {metrics_path}")
    print("\n" + "=" * 60)
    print("RESULTADOS:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print("=" * 60 + "\n")


def main() -> None:
    try:
        data, feature_cols = prepare_dataset()
        train_model(data, feature_cols)
    except Exception as exc:
        print(f"\n[erro] {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

