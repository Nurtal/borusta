"""Quick smoke test for the boruta_rs Python bindings."""
import numpy as np
from boruta_rs import Boruta

rng = np.random.default_rng(42)
n_obs, n_informative, n_noise = 300, 4, 4
X_inf = (rng.random((n_obs, n_informative)) - 0.5) * 4.0
X_noise = (rng.random((n_obs, n_noise)) - 0.5) * 4.0
X = np.hstack([X_inf, X_noise])
y = (X_inf.sum(axis=1) > 0).astype(np.uint32)

print(f"Dataset: {X.shape[0]} obs × {X.shape[1]} features")
print(f"Class balance: {y.mean():.1%} positive\n")

# Classification
result = Boruta(max_iter=100, p_value=0.01, n_estimators=100, random_seed=42).fit(X, y)
print(repr(result))
print(f"  confirmed : {result.confirmed}")
print(f"  rejected  : {result.rejected}")
print(f"  tentative : {result.tentative}")

# Rough fix (should already be empty here, but API test)
result.rough_fix()
print(f"  after rough_fix — tentative: {result.tentative}")

# CSV export
csv = result.importance_history_csv()
lines = csv.splitlines()
print(f"\nimportance_history_csv: {len(lines)} lines, header = '{lines[0][:60]}...'")

# Regression
y_reg = X_inf.sum(axis=1) + rng.normal(0, 0.5, n_obs)
result_reg = Boruta(max_iter=100, n_estimators=100, random_seed=42).fit_regression(X, y_reg)
print(f"\nRegression: {repr(result_reg)}")
print(f"  confirmed : {result_reg.confirmed}")
print(f"  rejected  : {result_reg.rejected}")

print("\n✓ All bindings work correctly")
