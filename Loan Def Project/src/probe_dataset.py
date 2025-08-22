import pandas as pd
import numpy as np
import sys
import json

path = sys.argv[1] if len(sys.argv) > 1 else "Default_Fin.csv"

encodings = ["utf-8", "utf-8-sig", "latin1"]
df = None
for enc in encodings:
    try:
        df = pd.read_csv(path, encoding=enc)
        break
    except Exception:
        pass
if df is None:
    df = pd.read_csv(path)

info = {}
info["shape"] = df.shape
info["columns"] = df.columns.tolist()
info["dtypes"] = df.dtypes.astype(str).to_dict()

# candidates
default_like = [c for c in df.columns if "default" in c.lower() or "defa" in c.lower()]
small_card = {}
for c in df.columns:
    nun = df[c].nunique(dropna=True)
    if nun <= 5:
        vals = [str(x) for x in sorted(df[c].dropna().unique().tolist())][:10]
        small_card[c] = vals

info["default_like"] = default_like
info["small_card_cols"] = small_card

print(json.dumps(info, indent=2))
