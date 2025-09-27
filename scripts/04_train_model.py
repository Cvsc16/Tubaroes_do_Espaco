
#!/usr/bin/env python3
# Treina XGBoost simples e salva métricas e modelo

from pathlib import Path
import yaml, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import joblib

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT/"config"/"config.yaml"))
PRO = ROOT/"data"/"processed"

df = pd.read_csv(PRO/"dataset.csv")
X = df[["sst","chl"]].values
y = df["label"].values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42)
model.fit(Xtr, ytr)
probs = model.predict_proba(Xte)[:,1]
auc = roc_auc_score(yte, probs)
ap  = average_precision_score(yte, probs)

joblib.dump(model, PRO/"model_xgb.pkl")
with open(PRO/"metrics.json","w") as f:
    json.dump({"AUC": float(auc), "AveragePrecision": float(ap)}, f, indent=2)

print(f"AUC={auc:.3f} | AP={ap:.3f} — artefatos salvos em data/processed/")
