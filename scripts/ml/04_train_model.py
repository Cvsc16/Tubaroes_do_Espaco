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
import json
import gc

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if _parent.name == "scripts":
        _PROJECT_ROOT_FALLBACK = _parent.parent
        break
else:
    _PROJECT_ROOT_FALLBACK = _THIS_FILE.parent

if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))

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
    """Carrega features de forma otimizada."""
    files = sorted(FEATURES_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            "Nenhum arquivo de features encontrado em data/features/. "
            "Rode 03_feature_engineering primeiro."
        )

    print(f"\n{'='*60}")
    print(f"Carregando {len(files)} arquivo(s) de features...")
    print(f"{'='*60}\n")

    frames = []
    for f in files:
        # OTIMIZAÇÃO: Carregar apenas colunas necessárias
        try:
            df = pd.read_csv(
                f,
                usecols=["date", "lat", "lon", "sst", "sst_gradient", "chlor_a"],
                dtype={
                    "lat": "float32",
                    "lon": "float32", 
                    "sst": "float32",
                    "sst_gradient": "float32",
                    "chlor_a": "float32"
                }
            )
            if df.empty:
                print(f"⚠️  {f.name} está vazio, ignorando...")
                continue
            
            df["source_file"] = f.name
            frames.append(df)
            print(f"✅ {f.name}: {len(df):,} linhas carregadas")
            
        except Exception as e:
            print(f"❌ Erro ao carregar {f.name}: {e}")
            continue

    if not frames:
        raise ValueError("Arquivos de features vazios. Verifique dados de entrada.")

    print(f"\nConcatenando {len(frames)} DataFrames...")
    data = pd.concat(frames, ignore_index=True)
    
    # Limpeza de memória
    del frames
    gc.collect()
    
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    
    print(f"✅ Total carregado: {len(data):,} linhas")
    return data


def apply_labels(df: pd.DataFrame, top_percent: float) -> pd.DataFrame:
    """Rotula hotspots baseado no top-N% de gradiente por data."""
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

    if "date" in df.columns and df["date"].notna().any():
        df = (
            df.groupby(df["date"].fillna(pd.Timestamp("1970-01-01")), group_keys=False)
            .apply(label_group, include_groups=False)
        )
    else:
        df = label_group(df)
    
    n_positives = df["label"].sum()
    n_negatives = len(df) - n_positives
    print(f"✅ Labels aplicados: {n_positives:,} positivos | {n_negatives:,} negativos")
    
    return df


def prepare_dataset() -> pd.DataFrame:
    """Prepara dataset consolidado para treinamento."""
    data = load_features()
    
    print(f"\n{'='*60}")
    print("Preparando dataset...")
    print(f"{'='*60}\n")
    
    # Verifica se chlor_a existe
    if 'chlor_a' not in data.columns:
        raise ValueError(
            "Coluna 'chlor_a' ausente. "
            "Rode 01/02/03 para gerar features com MODIS L3 CHL."
        )

    # Mostra estatísticas de NaN antes de remover
    total_rows = len(data)
    for col in FEATURE_COLS:
        nan_count = data[col].isna().sum()
        nan_pct = (nan_count / total_rows) * 100
        print(f"  {col}: {nan_count:,} NaN ({nan_pct:.1f}%)")

    # Remove linhas com NaN nas features
    print(f"\nRemovendo linhas com NaN...")
    data = data.dropna(subset=FEATURE_COLS)
    
    if data.empty:
        raise ValueError(
            "Sem dados válidos (sst/sst_gradient/chlor_a) para treinar. "
            "Rode 03_feature_engineering.py com --dropna para gerar dados limpos."
        )
    
    print(f"✅ Após limpeza: {len(data):,} linhas ({(len(data)/total_rows)*100:.1f}% dos dados)")

    # Aplica labels
    data = apply_labels(data, TOP_PERCENT)
    
    if data["label"].nunique() < 2:
        raise ValueError(
            "A rotulagem gerou apenas uma classe. "
            "Ajuste model.top_percent_for_hit em config/config.yaml."
        )

    # Salva dataset consolidado
    if "date" in data.columns:
        data["date"] = data["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    dataset_cols = [
        col for col in ["lat", "lon", "date", *FEATURE_COLS, "label", "source_file"] 
        if col in data.columns
    ]
    dataset_path = PROCESSED_DIR / "dataset.csv"
    
    print(f"\nSalvando dataset consolidado...")
    data[dataset_cols].to_csv(dataset_path, index=False)
    print(f"✅ Dataset salvo em {dataset_path} ({len(data):,} linhas)")
    
    return data


def train_model(data: pd.DataFrame):
    """Treina modelo de classificação binária."""
    print(f"\n{'='*60}")
    print("Treinando modelo...")
    print(f"{'='*60}\n")
    
    X = data[FEATURE_COLS].values
    y = data["label"].astype(int).values

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class balance: {np.bincount(y)}")

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    # Modelo
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
            n_jobs=-1,  # Usar todos os cores
        )
    else:
        print("⚠️  xgboost não encontrado; usando GradientBoostingClassifier...")
        model = SkGradientBoosting(random_state=42, n_estimators=100)

    print("\nTreinando...")
    model.fit(X_train, y_train)
    print("✅ Treinamento concluído!")

    # Avaliação
    print("\nAvaliando modelo...")
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    # Salva modelo
    model_path = PROCESSED_DIR / "model_xgb.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Modelo salvo em {model_path}")

    # Salva métricas
    metrics_path = PROCESSED_DIR / "metrics.json"
    metrics = {
        "AUC": float(auc),
        "AveragePrecision": float(ap),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "top_percent": float(TOP_PERCENT)
    }
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Métricas salvas em {metrics_path}")
    print(f"\n{'='*60}")
    print(f"RESULTADOS:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print(f"{'='*60}\n")


def main():
    try:
        data = prepare_dataset()
        train_model(data)
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()