"""Train and evaluate fraud-detection models on highly imbalanced data.

    python src/train.py                  # uses data/creditcard.csv if present, else a synthetic demo
    python src/train.py --rf-trees 300

Outputs: outputs/metrics.csv, outputs/findings.md, outputs/charts/*.png, models/best_model.joblib
"""
import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "creditcard.csv"
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Amount", "hour"] + V_COLS
BLUE, ORANGE, GREY = "#2E75B6", "#C55A11", "#7F7F7F"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def add_features(df):
    """Time (seconds since first transaction) -> hour of day. Returns model-ready columns."""
    d = df.copy()
    d["hour"] = (d["Time"] // 3600) % 24
    return d[FEATURES]


def build(model):
    pre = ColumnTransformer([("scale", RobustScaler(), ["Amount", "hour"])], remainder="passthrough")
    return Pipeline([("pre", pre), ("clf", model)])


def pick_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def evaluate(name, y, p, thr, amount):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fraud_value = amount[y == 1].sum()
    return {"model": name, "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "threshold": thr, "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fraud_value_caught_pct": 100 * amount[(y == 1) & (pred == 1)].sum() / fraud_value if fraud_value else 0,
            "false_alerts_per_1000": 1000 * fp / len(y)}


def save(fig, name, label):
    fig.text(0.99, 0.01, label, ha="right", va="bottom", fontsize=7, color=GREY, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ROOT / "outputs" / "charts" / name, dpi=130)
    plt.close(fig)


def main(rf_trees):
    if REAL.exists():
        df, label = pd.read_csv(REAL), "Kaggle creditcard.csv (real data)"
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from make_sample_data import make
        df, label = make(), "SYNTHETIC DEMO (not real results)"
        print("NOTE: data/creditcard.csv not found. Using the synthetic demo data.")
    X, y, amount = add_features(df), df["Class"].values, df["Amount"].values
    print(f"[{label}] {len(df):,} rows, {y.sum():,} frauds ({y.mean():.3%})")

    idx = np.arange(len(df))
    tr_val, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    tr, va = train_test_split(tr_val, test_size=0.25, stratify=y[tr_val], random_state=42)   # 60 / 20 / 20
    models = {
        "Baseline: never flags fraud": (DummyClassifier(strategy="most_frequent"), False),
        "Logistic Regression": (LogisticRegression(max_iter=2000), True),
        "Random Forest": (RandomForestClassifier(n_estimators=rf_trees, min_samples_leaf=2, n_jobs=-1, random_state=42), True),
        "Gradient Boosting": (HistGradientBoostingClassifier(max_iter=200, random_state=42), True),
    }
    sw = compute_sample_weight("balanced", y[tr])          # fraud rows weigh much more in training
    rows, fitted, scores, val_ap = [], {}, {}, {}
    for name, (model, tune) in models.items():
        pipe = build(model)
        pipe.fit(X.iloc[tr], y[tr], **({"clf__sample_weight": sw} if tune else {}))
        pv, pt = pipe.predict_proba(X.iloc[va])[:, 1], pipe.predict_proba(X.iloc[te])[:, 1]
        thr = pick_threshold(y[va], pv) if tune else 0.5     # threshold chosen on validation, never on test
        rows.append(evaluate(name, y[te], pt, thr, amount[te]))
        fitted[name], scores[name], val_ap[name] = pipe, pt, average_precision_score(y[va], pv)
        r = rows[-1]
        print(f"  {name:30s} PR-AUC {r['pr_auc']:.3f}  precision {r['precision']:.3f}  recall {r['recall']:.3f}")
    m = pd.DataFrame(rows)
    m.round(4).to_csv(ROOT / "outputs" / "metrics.csv", index=False)
    best = max((n for n in models if not n.startswith("Baseline")), key=lambda n: val_ap[n])
    best_row = m[m.model == best].iloc[0]
    joblib.dump({"pipeline": fitted[best], "threshold": float(best_row.threshold), "model": best,
                 "features": FEATURES}, ROOT / "models" / "best_model.joblib")

    # ---- charts
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in models:
        if n.startswith("Baseline"):
            continue
        p_, r_, _ = precision_recall_curve(y[te], scores[n])
        ax.plot(r_, p_, label=f"{n} (PR-AUC {m[m.model == n].pr_auc.iloc[0]:.2f})")
    ax.axhline(y[te].mean(), color=GREY, ls="--", label="No-skill (fraud rate)")
    ax.set(title="Precision-recall curves (test set)", xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    save(fig, "1_precision_recall_curves.png", label)

    pred = (scores[best] >= best_row.threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y[te], pred), display_labels=["Legit", "Fraud"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix: {best}\n(threshold {best_row.threshold:.2f})")
    save(fig, "2_confusion_matrix.png", label)

    p_, r_, t_ = precision_recall_curve(y[te], scores[best])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_, p_[:-1], color=BLUE, label="Precision")
    ax.plot(t_, r_[:-1], color=ORANGE, label="Recall")
    ax.axvline(best_row.threshold, color=GREY, ls="--", label="Chosen threshold")
    ax.set(title=f"Precision vs recall by threshold: {best}", xlabel="Decision threshold", ylabel="Score")
    ax.legend()
    save(fig, "3_threshold_tradeoff.png", label)

    rf = fitted["Random Forest"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=["Amount", "hour"] + V_COLS).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set(title="Top 12 features (Random Forest importance)", xlabel="Importance")
    save(fig, "4_feature_importance.png", label)

    # ---- findings
    b0 = m.iloc[0]
    lines = ["# Auto-generated results", "", f"_Data: {label}_", "",
             f"- {len(df):,} transactions, {int(y.sum()):,} fraud ({y.mean():.3%}). Split 60/20/20 (train/validation/test), stratified.",
             f"- The baseline that never flags fraud scores **{b0.accuracy:.2%} accuracy** but catches **{b0.recall:.0%}** of fraud, so accuracy is misleading here.",
             f"- Best model (by validation PR-AUC): **{best}**. On the test set it reaches precision {best_row.precision:.1%} and "
             f"recall {best_row.recall:.1%}, catches {best_row.fraud_value_caught_pct:.1f}% of fraud value, with "
             f"{best_row.false_alerts_per_1000:.2f} false alerts per 1,000 transactions.", "",
             "| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Fraud value caught | False alerts / 1,000 |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in m.iterrows():
        lines.append(f"| {r.model} | {r.pr_auc:.3f} | {r.roc_auc:.3f} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | "
                     f"{r.fraud_value_caught_pct:.1f}% | {r.false_alerts_per_1000:.2f} |")
    (ROOT / "outputs" / "findings.md").write_text("\n".join(lines) + "\n")
    print(f"\nBest model: {best}. Results in outputs/, model saved to models/best_model.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-trees", type=int, default=150)
    main(ap.parse_args().rf_trees)
