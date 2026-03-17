"""
Benchmark BorutaPy on real-world datasets (Iris, Wine).
"""
import csv, time, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    feat_cols = [k for k in rows[0] if k != "label"]
    X = np.array([[float(r[c]) for c in feat_cols] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    return X, y, feat_cols

for dataset in ["iris", "wine"]:
    path = f"benchmark/{dataset}.csv"
    X, y, feat_cols = load_csv(path)
    n_classes = len(set(y))
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}  ({X.shape[0]} obs × {X.shape[1]} features, {n_classes} classes)")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector = BorutaPy(estimator=rf, n_estimators="auto", max_iter=100,
                        alpha=0.01, random_state=42, verbose=0)

    t0 = time.perf_counter()
    selector.fit(X, y)
    elapsed = time.perf_counter() - t0

    n_iter = getattr(selector, "n_iter_", "?")
    confirmed = [feat_cols[i] for i in range(len(feat_cols)) if selector.support_[i]]
    tentative = [feat_cols[i] for i in range(len(feat_cols)) if selector.support_weak_[i]]
    rejected  = [feat_cols[i] for i in range(len(feat_cols))
                 if not selector.support_[i] and not selector.support_weak_[i]]

    print(f"=== BorutaPy ({n_iter} iterations, {elapsed:.3f}s) ===")
    print(f"  Confirmed  ({len(confirmed):2d}): {confirmed}")
    print(f"  Rejected   ({len(rejected):2d}): {rejected}")
    print(f"  Tentative  ({len(tentative):2d}): {tentative}")
    print(f"  Elapsed: {elapsed:.3f}s")
