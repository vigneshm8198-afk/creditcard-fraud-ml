"""Score new transactions with the saved model.

    python src/predict.py --input new_transactions.csv --output scored.csv

The CSV needs the columns Time, V1..V28 and Amount (a 'Class' column is optional and ignored).
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import ROOT, add_features

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", default="scored_transactions.csv")
a = ap.parse_args()
bundle = joblib.load(ROOT / "models" / "best_model.joblib")
df = pd.read_csv(a.input)
prob = bundle["pipeline"].predict_proba(add_features(df))[:, 1]
df["fraud_probability"] = prob.round(4)
df["flagged_as_fraud"] = (prob >= bundle["threshold"]).astype(int)
df.to_csv(a.output, index=False)
print(f"Scored {len(df):,} transactions with {bundle['model']}; flagged {int(df.flagged_as_fraud.sum()):,}. Saved to {a.output}")
