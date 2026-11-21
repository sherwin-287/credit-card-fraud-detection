"""Credit card fraud detection: time-based evaluation of three models, with and without class weights,
cost-based threshold tuning, permutation importance and a per-fraud-type breakdown.

Run:  python3 src/fraud_pipeline.py
Out:  results/metrics.csv, results/summary.json, figures/*.png
Uses scikit-learn only (no extra installs). XGBoost, SMOTE and SHAP are in the Colab notebook.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "credit_card_fraud_detection_2026.csv"
SEED = 42
REVIEW_COST = 5.0          # assumed cost (USD) of manually reviewing one flagged transaction

# Columns that must NOT be features
IDS = ["transaction_id", "customer_id", "card_id", "merchant_id", "transaction_datetime"]
LEAK = ["fraud_type", "fraud_amount_lost"]       # only known after fraud is confirmed
TARGET = "is_fraud"


def load():
    df = pd.read_csv(DATA, parse_dates=["transaction_datetime"]).sort_values("transaction_datetime")
    df["browser"] = df["browser"].fillna("None")   # empty for in-store/ATM: no browser used
    return df.reset_index(drop=True)


def time_split(df):
    """60% train / 20% validation (threshold choice) / 20% test, in time order."""
    n = len(df)
    return df.iloc[: int(.6 * n)], df.iloc[int(.6 * n): int(.8 * n)], df.iloc[int(.8 * n):]


def build_preprocessor(X):
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ]), num, cat


def make_models(weighted):
    cw = "balanced" if weighted else None
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=cw),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                class_weight="balanced_subsample" if weighted else None,
                                                random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(class_weight=cw, random_state=SEED),
    }


def threshold_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return dict(precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
                flagged=int(pred.sum()))


def total_cost(y, loss, p, thr):
    """Money lost to missed fraud + review cost of every flagged transaction."""
    flagged = p >= thr
    return float(loss[(~flagged) & (y == 1)].sum() + REVIEW_COST * flagged.sum())


def best_threshold(y, loss, p):
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.9995, 400)))
    costs = [total_cost(y, loss, p, t) for t in grid]
    return float(grid[int(np.argmin(costs))]), grid, costs


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    df = load()
    tr, va, te = time_split(df)
    feats = [c for c in df.columns if c not in IDS + LEAK + [TARGET]]
    print(f"rows {len(df)} | fraud {df[TARGET].mean():.2%} | features {len(feats)}")
    for name, part in [("train", tr), ("validation", va), ("test", te)]:
        print(f"  {name:10s} {part.transaction_datetime.min():%Y-%m-%d} to {part.transaction_datetime.max():%Y-%m-%d}"
              f"  n={len(part)}  fraud={part[TARGET].mean():.2%}")
    pre, num, cat = build_preprocessor(df[feats])

    rows, fitted, test_scores, val_scores = [], {}, {}, {}
    for weighted in (False, True):
        for name, clf in make_models(weighted).items():
            pipe = Pipeline([("prep", pre), ("clf", clf)]).fit(tr[feats], tr[TARGET])
            pv, pt = pipe.predict_proba(va[feats])[:, 1], pipe.predict_proba(te[feats])[:, 1]
            key = (name, weighted)
            fitted[key], val_scores[key], test_scores[key] = pipe, pv, pt
            m = threshold_metrics(te[TARGET], pt, 0.5)
            rows.append(dict(model=name, class_weights=weighted,
                             roc_auc=roc_auc_score(te[TARGET], pt),
                             pr_auc=average_precision_score(te[TARGET], pt),
                             precision_at_0_5=m["precision"], recall_at_0_5=m["recall"]))
            print(f"{name:20s} weights={weighted!s:5} ROC-AUC {rows[-1]['roc_auc']:.3f}  PR-AUC {rows[-1]['pr_auc']:.3f}"
                  f"  P@0.5 {m['precision']:.3f}  R@0.5 {m['recall']:.3f}")
    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(ROOT / "results" / "metrics.csv", index=False)

    # best model by validation PR-AUC (selection is on validation, never on test)
    val_pr = {k: average_precision_score(va[TARGET], v) for k, v in val_scores.items()}
    best = max(val_pr, key=val_pr.get)
    pipe, pv, pt = fitted[best], val_scores[best], test_scores[best]
    print("selected on validation PR-AUC:", best, round(val_pr[best], 4))

    # cost-based threshold chosen on validation, applied once to test
    thr, grid, costs = best_threshold(va[TARGET].values, va["fraud_amount_lost"].values, pv)
    y_te, loss_te = te[TARGET].values, te["fraud_amount_lost"].values
    at05, attuned = threshold_metrics(y_te, pt, 0.5), threshold_metrics(y_te, pt, thr)
    cost05, costtuned = total_cost(y_te, loss_te, pt, 0.5), total_cost(y_te, loss_te, pt, thr)
    cost_none = float(loss_te[y_te == 1].sum())
    print(f"threshold {thr:.4f}: precision {attuned['precision']:.3f} recall {attuned['recall']:.3f} flagged {attuned['flagged']}")

    # per fraud type recall at the tuned threshold (test)
    te_ = te.assign(score=pt, caught=(pt >= thr).astype(int))
    by_type = (te_[te_[TARGET] == 1].groupby("fraud_type").agg(n=("caught", "size"), recall=("caught", "mean"),
                                                                avg_loss=("fraud_amount_lost", "mean")).round(3))
    by_type.to_csv(ROOT / "results" / "recall_by_fraud_type.csv")
    print(by_type)

    # permutation importance on a test sample, scored by PR-AUC
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(te), size=min(10000, len(te)), replace=False)
    pi = permutation_importance(pipe, te[feats].iloc[idx], te[TARGET].iloc[idx], scoring="average_precision",
                                n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    imp.round(5).to_csv(ROOT / "results" / "permutation_importance.csv", header=["pr_auc_drop"])
    print(imp.head(8).round(4))

    # ---------------- figures ----------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # 1 model comparison
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, col, ttl in zip(ax, ["pr_auc", "recall_at_0_5"], ["PR-AUC (test)", "Recall at threshold 0.5 (test)"]):
        piv = metrics.pivot(index="model", columns="class_weights", values=col)
        piv.columns = ["No class weights", "Class weights"]
        piv.plot.bar(ax=a, color=["#9aa5b1", "#1f5fa8"], rot=15, width=.75)
        a.set_title(ttl); a.set_xlabel(""); a.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(ROOT / "figures" / "fig1_model_comparison.png", dpi=160); plt.close()
    # 2 PR curve for selected model vs baseline
    fig, a = plt.subplots(figsize=(5.2, 4))
    pr, rc, _ = precision_recall_curve(y_te, pt)
    a.plot(rc, pr, color="#1f5fa8", lw=2, label=f"{best[0]} (AP {average_precision_score(y_te, pt):.2f})")
    a.axhline(y_te.mean(), color="#9aa5b1", ls="--", label=f"Chance ({y_te.mean():.3f})")
    a.scatter([attuned["recall"]], [attuned["precision"]], color="#d1495b", zorder=5, label="Tuned threshold")
    a.set_xlabel("Recall"); a.set_ylabel("Precision"); a.set_title("Precision-recall curve (test)"); a.legend(frameon=False)
    plt.tight_layout(); plt.savefig(ROOT / "figures" / "fig2_pr_curve.png", dpi=160); plt.close()
    # 3 cost vs threshold (validation)
    fig, a = plt.subplots(figsize=(5.2, 4))
    a.plot(grid, costs, color="#1f5fa8"); a.axvline(thr, color="#d1495b", ls="--", label=f"Chosen: {thr:.3f}")
    a.set_xlabel("Score threshold"); a.set_ylabel("Total cost on validation (USD)")
    a.set_title("Cost-based threshold choice"); a.legend(frameon=False)
    plt.tight_layout(); plt.savefig(ROOT / "figures" / "fig3_cost_threshold.png", dpi=160); plt.close()
    # 4 permutation importance
    fig, a = plt.subplots(figsize=(6, 4.2))
    imp.head(10)[::-1].plot.barh(ax=a, color="#1f5fa8"); a.set_xlabel("Drop in PR-AUC when feature is shuffled")
    a.set_title("Permutation importance (test sample)")
    plt.tight_layout(); plt.savefig(ROOT / "figures" / "fig4_permutation_importance.png", dpi=160); plt.close()
    # 5 recall by fraud type
    fig, a = plt.subplots(figsize=(5.6, 3.8))
    by_type["recall"].plot.bar(ax=a, color="#1f5fa8", rot=15); a.set_ylabel("Recall"); a.set_xlabel("")
    a.set_ylim(0, 1); a.set_title("Share of each fraud type caught (test)")
    plt.tight_layout(); plt.savefig(ROOT / "figures" / "fig5_recall_by_fraud_type.png", dpi=160); plt.close()

    summary = dict(
        rows=len(df), fraud_rate=float(df[TARGET].mean()), n_features=len(feats),
        split={k: dict(n=len(p), fraud=int(p[TARGET].sum())) for k, p in [("train", tr), ("validation", va), ("test", te)]},
        selected_model=best[0], selected_class_weights=best[1], validation_pr_auc=val_pr[best],
        test_roc_auc=float(roc_auc_score(y_te, pt)), test_pr_auc=float(average_precision_score(y_te, pt)),
        chance_pr_auc=float(y_te.mean()), threshold=thr,
        test_at_0_5=dict(at05, cost=cost05), test_at_tuned=dict(attuned, cost=costtuned),
        test_cost_if_no_model=cost_none, review_cost=REVIEW_COST, test_fraud_count=int(y_te.sum()),
    )
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
