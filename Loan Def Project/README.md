# Loan Default Prediction (VS Code Starter)

## 1) Create & activate a virtual environment
```
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

## 2) Install requirements
```
pip install -r requirements.txt
```

## 3) Set the target in `config.yaml`
Detected target guess: **Defaulted?**
If this is wrong, edit `config.yaml` and set `target:` to the correct column.

## 4) Run the baseline training
```
python src/baseline_train.py
```

This will:
- Load /mnt/data/Default_Fin.csv
- Split train/test
- Build a preprocessing + Logistic Regression baseline
- Handle missing values and categorical encoding
- Print metrics (ROC AUC, PR AUC, F1, confusion matrix), and save the model to `models/baseline.joblib`
