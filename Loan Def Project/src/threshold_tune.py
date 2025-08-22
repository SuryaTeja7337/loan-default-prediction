import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, average_precision_score
import yaml

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def read_csv(path):
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
    target = cfg["target"]

    # Load data (same cleaning as training: drop obvious IDs)
    df = read_csv(data_path)
    for col in ["Index", "id", "ID", "customer_id", "loan_id"]:
        if col in df.columns and col != target:
            df = df.drop(columns=[col])

    y = df[target].astype(float)
    X = df.drop(columns=[target])

    # Load the trained pipeline
    pipe = joblib.load(root / "models" / "baseline.joblib")

    # Get probabilities on the full dataset
    # (If you later save a dedicated test set, prefer to use that instead.)
    y_prob = pipe.predict_proba(X)[:, 1]

    # Search thresholds for best F1
    thresholds = np.linspace(0.01, 0.99, 99)
    best = {"thresh": 0.5, "f1": -1, "precision": None, "recall": None}
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        f1 = f1_score(y, y_hat, zero_division=0)

        tp = ((y_hat==1) & (y==1)).sum()
        fp = ((y_hat==1) & (y==0)).sum()
        fn = ((y_hat==0) & (y==1)).sum()
        precision = tp / (tp + fp) if (tp+fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp+fn) > 0 else 0.0

        if f1 > best["f1"]:
            best = {"thresh": float(t), "f1": float(f1),
                    "precision": float(precision), "recall": float(recall)}

    ap = average_precision_score(y, y_prob)

    print("Best threshold by F1:", best)
    print(f"Average Precision (PR AUC): {ap:.4f}")

    # Save chosen threshold for reuse by predict script
    out = root / "models" / "decision_threshold.txt"
    out.write_text(str(best["thresh"]))
    print(f"Saved chosen threshold to: {out}")
