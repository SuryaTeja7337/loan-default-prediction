import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

# --------------------------
# Helpers
# --------------------------
def load_config(cfg_path: Path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def detect_feature_types(df: pd.DataFrame):
    """
    df should already exclude the target.
    Returns numeric and categorical column lists.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

def normalize_binary_target(series: pd.Series) -> pd.Series:
    if series.dtype == "O":
        m_yes = {"yes":1, "y":1, "true":1, "t":1, "default":1, "defaulter":1, "bad":1, "1":1}
        m_no  = {"no":0, "n":0, "false":0, "f":0, "non-default":0, "good":0, "0":0}
        series = (
            series.astype(str)
                  .str.strip()
                  .str.lower()
                  .map(lambda x: m_yes.get(x, m_no.get(x, x)))
        )
    try:
        series = series.astype(float)
    except Exception:
        pass
    return series

# --------------------------
# Main
# --------------------------
def main():
    root = Path(__file__).resolve().parents[1]
    cfg  = load_config(root / "config.yaml")

    data_path    = Path(cfg["data_path"])
    target       = cfg["target"]
    test_size    = float(cfg.get("test_size", 0.2))
    random_state = int(cfg.get("random_state", 42))

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    # Load data
    df = read_csv_robust(data_path)

    # Drop obvious ID-like columns if present (and not the target)
    for col in ["Index", "id", "ID", "customer_id", "loan_id"]:
        if col in df.columns and col != target:
            df = df.drop(columns=[col])

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in CSV columns: {df.columns.tolist()}")

    # Clean target
    df = df.dropna(subset=[target]).copy()
    df[target] = normalize_binary_target(df[target])

    # Split features/labels
    X = df.drop(columns=[target])
    y = df[target]

    strat = y if y.nunique(dropna=True) == 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    # Feature types & preprocessing
    num_cols, cat_cols = detect_feature_types(Xtr)  # <— NO target argument here
    pre = build_preprocessor(num_cols, cat_cols)

    # Baseline model
    clf = LogisticRegression(max_iter=200, class_weight="balanced")

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", clf)
    ])

    # Train
    pipe.fit(Xtr, ytr)

    # Evaluate
    ypred = pipe.predict(Xte)
    try:
        yprob = pipe.predict_proba(Xte)[:, 1]
    except Exception:
        yprob = None

    print("=== Test Metrics ===")
    if yprob is not None and yte.nunique() == 2:
        print(f"ROC AUC: {roc_auc_score(yte, yprob):.4f}")
        print(f"PR  AUC: {average_precision_score(yte, yprob):.4f}")
    print(f"F1 Score: {f1_score(yte, ypred, average='binary' if yte.nunique()==2 else 'macro'):.4f}")
    print("\nClassification Report:\n", classification_report(yte, ypred))
    print("Confusion Matrix:\n", confusion_matrix(yte, ypred))

    # Save pipeline
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "baseline.joblib"
    joblib.dump(pipe, out_path)
    print(f"\nSaved trained pipeline to: {out_path}")

if __name__ == "__main__":
    main()
