"""
Benchmark BorutaPy (scikit-learn-contrib) on the shared dataset.
"""
import csv, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# ── Load dataset ──────────────────────────────────────────────────────────────
with open("benchmark/dataset.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

feature_cols = [k for k in rows[0] if k != "label"]
X = np.array([[float(r[c]) for c in feature_cols] for r in rows])
y = np.array([int(r["label"]) for r in rows])

print(f"Dataset: {X.shape[0]} obs × {X.shape[1]} features")
print(f"Class balance: {y.mean():.2%} positive\n")

# ── Run BorutaPy ──────────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
selector = BorutaPy(
    estimator=rf,
    n_estimators="auto",
    max_iter=100,
    alpha=0.01,
    random_state=42,
    verbose=0,
)

t0 = time.perf_counter()
selector.fit(X, y)
elapsed = time.perf_counter() - t0

# ── Results ───────────────────────────────────────────────────────────────────
n_iter = getattr(selector, "n_iter_", getattr(selector, "n_estimators", "?"))
print(f"=== BorutaPy results ({n_iter} iterations, {elapsed:.2f}s) ===")
confirmed  = [feature_cols[i] for i in range(len(feature_cols)) if selector.support_[i]]
tentative  = [feature_cols[i] for i in range(len(feature_cols)) if selector.support_weak_[i]]
rejected   = [feature_cols[i] for i in range(len(feature_cols))
              if not selector.support_[i] and not selector.support_weak_[i]]

print(f"  Confirmed  ({len(confirmed):2d}): {confirmed}")
print(f"  Rejected   ({len(rejected):2d}): {rejected}")
print(f"  Tentative  ({len(tentative):2d}): {tentative}")
print(f"\n  Elapsed: {elapsed:.3f}s")
