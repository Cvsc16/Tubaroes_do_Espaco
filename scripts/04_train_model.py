#!/usr/bin/env python3
"""Treina um modelo binario simples a partir das features geradas.

- Consolida arquivos em data/features/ em um unico dataset tabular
- Rotula hotspots como o top-N por cento de gradiente por data (configuravel em config.yaml)
- Usa XGBoost (ou GradientBoosting se xgboost nao estiver instalado)
- Salva dataset agregado, modelo e metricas em data/processed/
"""

from __future__ import annotations

from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent

if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))

import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FALLBACK = THIS_DIR.parent
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))


import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:  # fallback leve caso xgboost nao esteja disponivel
    from sklearn.ensemble import GradientBoostingClassifier as SkGradientBoosting
    HAVE_XGB = False
except ImportError:  # fallback leve caso xgboost nao esteja disponivel
    from sklearn.ensemble import GradientBoostingClassifier as SkGradientBoosting
    HAVE_XGB = False

try:
    from scripts.utils import load_config
except ModuleNotFoundError:
    from utils_config import load_config

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CFG = load_config()
TOP_PERCENT = CFG.get("model", {}).get("top_percent_for_hit", 20)
FEATURE_COLS = ["sst", "sst_gradient", "chlor_a"]


def load_features() -> pd.DataFrame:
    files = sorted(FEATURES_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Nenhum arquivo de features encontrado em data/features/. Rode 03_feature_engineering primeiro.")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["source_file"] = f.name
        frames.append(df)

    if not frames:
        raise ValueError("Arquivos de features vazios. Verifique dados de entrada.")

    data = pd.concat(frames, ignore_index=True)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


def apply_labels(df: pd.DataFrame, top_percent: float) -> pd.DataFrame:
    if not 0 < top_percent < 100:
        top_percent = 20

    def label_group(group: pd.DataFrame) -> pd.DataFrame:
        valid = group["sst_gradient"].dropna()
        if valid.empty:
            group["label"] = 0
            return group
        threshold = np.percentile(valid, 100 - top_percent)
        group["label"] = (group["sst_gradient"] >= threshold).astype(int)
        return group

    if "date" in df.columns and df["date"].notna().any():
        df = (
            df.groupby(df["date"].fillna(pd.Timestamp("1970-01-01")), group_keys=False)
            .apply(label_group)
        )
    else:
        df = label_group(df)
    return df


def prepare_dataset() -> pd.DataFrame:
    data = load_features()
    if 'chlor_a' not in data.columns:
        raise ValueError("Coluna 'chlor_a' ausente. Rode 01/02/03 para MODIS L3 CHL antes do treino.")

    data = data.dropna(subset=FEATURE_COLS)
    if data.empty:
        raise ValueError("Sem dados validos (sst/sst_gradient/chlor_a) para treinar.")

    data = apply_labels(data, TOP_PERCENT)
    if data["label"].nunique() < 2:
        raise ValueError("A rotulagem gerou apenas uma classe. Ajuste model.top_percent_for_hit em config/config.yaml.")

    if "date" in data.columns:
        data["date"] = data["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    dataset_cols = [col for col in ["lat", "lon", "date", *FEATURE_COLS, "label", "source_file"] if col in data.columns]
    dataset_path = PROCESSED_DIR / "dataset.csv"
    data[dataset_cols].to_csv(dataset_path, index=False)
    print(f"Dataset consolidado salvo em {dataset_path} ({len(data)} linhas)")
    return data


def train_model(data: pd.DataFrame):
    X = data[FEATURE_COLS].values
    y = data["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    if HAVE_XGB:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
        )
    else:
        print("WARNING: xgboost nao encontrado; usando GradientBoostingClassifier como fallback.")
        model = SkGradientBoosting(random_state=42)

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    joblib.dump(model, PROCESSED_DIR / "model_xgb.pkl")
    metrics_path = PROCESSED_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"AUC": float(auc), "AveragePrecision": float(ap)}, f, indent=2)

    print(f"Modelo salvo em {PROCESSED_DIR / 'model_xgb.pkl'}")
    print(f"Metricas salvas em {metrics_path} - AUC={auc:.3f} | AP={ap:.3f}")


def main():
    data = prepare_dataset()
    train_model(data)


if __name__ == "__main__":
    main()


