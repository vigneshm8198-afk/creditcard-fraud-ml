"""SYNTHETIC data with the same columns as the Kaggle 'creditcard.csv' (Time, V1..V28, Amount, Class).
Used only so the project runs out of the box. Replace with the real file for real results."""
import numpy as np
import pandas as pd


def make(n=40_000, seed=5, fraud_rate=0.005):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    v = rng.normal(size=(n, 28))
    shifts = {4: 2.5, 11: 2.0, 2: 1.5, 3: -2.5, 10: -2.5, 12: -3.0, 14: -3.5, 17: -2.5, 7: -1.5}
    fi = np.where(y == 1)[0]
    typical = rng.random(len(fi)) < 0.7                      # 30% of frauds look almost normal (hard cases)
    for k, s in shifts.items():
        v[fi, k - 1] += 0.55 * s * np.where(typical, 1.0, 0.3)
        hour_w = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 3, 2, 1], float)
    t_legit = rng.choice(np.arange(48), n, p=np.tile(hour_w, 2) / np.tile(hour_w, 2).sum()) * 3600 + rng.integers(0, 3600, n)
    time = np.where(y == 1, rng.integers(0, 172_792, n), t_legit)
    amount = np.where(y == 1, rng.lognormal(3.0, 1.8, n).clip(0, 2500), rng.lognormal(3.3, 1.4, n).clip(0, 5000)).round(2)
    df = pd.DataFrame(v, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time.astype(float))
    df["Amount"], df["Class"] = amount, y
    return df


if __name__ == "__main__":
    d = make()
    d.to_csv("data/sample_creditcard.csv", index=False)
    print(f"Wrote {len(d):,} synthetic rows (fraud {d.Class.mean():.2%})")
