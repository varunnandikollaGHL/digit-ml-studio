"""Shared digit CSV validation and evaluation; no training on uploads."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "digit"
FEATURES = [f"pixel_{r}_{c}" for r in range(8) for c in range(8)]
LABELS = list(range(10))
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

def validate_test_csv(df):
    required = FEATURES + [TARGET]
    if df.empty or len(df) > 5000:
        raise ValueError("Test CSV must contain between 1 and 5,000 rows.")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing[:10]))
    frame = df[required].apply(pd.to_numeric, errors="raise")
    if frame.isna().any().any():
        raise ValueError("Missing values are not allowed.")
    pixels = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 16).any():
        raise ValueError("Pixel intensities must be finite numbers in [0, 16].")
    target = frame[TARGET].to_numpy()
    if not np.isin(target, LABELS).all() or not np.equal(target, np.floor(target)).all():
        raise ValueError("Digit labels must be integers in [0, 9].")
    frame[TARGET] = frame[TARGET].astype(int)
    return frame

def evaluate(model, frame):
    y = frame[TARGET].to_numpy()
    x = frame[FEATURES]
    pred = model.predict(x)
    prob = model.predict_proba(x)
    auc = None
    if set(np.unique(y)) == set(model.classes_):
        auc = float(roc_auc_score(y, prob, labels=model.classes_,
                                  multi_class="ovr", average="weighted"))
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "AUC": auc,
        "Precision": float(precision_score(y, pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y, pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred))
    }
