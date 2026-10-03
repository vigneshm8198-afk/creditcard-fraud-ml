# Data

## Option A: Demo (automatic)
With no real file present, `python src/train.py` generates a **synthetic** 40,000-row dataset with the same columns as the
Kaggle file. It exists only so the project runs out of the box; its results are not real.

## Option B: Real data (recommended)
Dataset: **Credit Card Fraud Detection** (ULB / Worldline, on Kaggle). 284,807 transactions by European cardholders over two days;
492 are fraud (0.172%). Features V1 to V28 are anonymised PCA components, plus `Time`, `Amount` and `Class` (1 = fraud).

1. Download `creditcard.csv` from Kaggle (search "Credit Card Fraud Detection").
2. Save it in this folder as `data/creditcard.csv` (about 150 MB, so it is not stored in the repo).
3. Run `python src/train.py`. Random Forest on the full file takes a few minutes; lower it with `--rf-trees 60` if needed.
