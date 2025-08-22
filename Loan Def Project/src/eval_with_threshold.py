from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score, recall_score
)
import yaml

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def read_csv_robust(path):
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")

    data_path = Path(cfg["data_path"])
    target    = cfg["target"]
    test_size = float(cfg.get("test_size", 0.2))
    seed      = int(cfg.get("random_state", 42))

    df = read_csv_robust(data_path)

    # Drop ID-like columns
    for col in ["Index", "id", "ID", "customer_id", "loan_id"]:
        if col in df.columns and col != target:
            df = df.drop(columns=[col])

    df = df.dropna(subset=[target]).copy()
    df[target] = df[target].astype(float)

    X = df.drop(columns=[target])
    y = df[target]

    strat = y if y.nunique(dropna=True) == 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )

    # Load model
    pipe = joblib.load(root / "models" / "baseline.joblib")

    # Load tuned threshold
    thresh_path = root / "models" / "decision_threshold.txt"
    t = 0.5
    if thresh_path.exists():
        t = float(thresh_path.read_text().strip())

    # Predictions
    y_prob = pipe.predict_proba(Xte)[:, 1]
    y_hat  = (y_prob >= t).astype(int)

    print(f"Using threshold: {t:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(yte, y_hat))
    print("\nClassification Report:\n", classification_report(yte, y_hat))
    print(f"F1: {f1_score(yte, y_hat):.4f}  Precision: {precision_score(yte, y_hat):.4f}  Recall: {recall_score(yte, y_hat):.4f}")

    flagged_rate = y_hat.mean()
    print(f"\nFlagged as 'default' rate: {flagged_rate:.3%}")
