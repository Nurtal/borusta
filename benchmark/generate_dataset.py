"""
Generates the benchmark dataset and saves it as CSV.
Same logic as the Rust integration test: label = sign(sum of informative features).
"""
import numpy as np
import csv, sys

def make_classification(n_obs=500, n_informative=5, n_noise=5, seed=42):
    rng = np.random.default_rng(seed)
    n_features = n_informative + n_noise
    X = (rng.random((n_obs, n_features)) - 0.5) * 4.0
    y = (X[:, :n_informative].sum(axis=1) > 0).astype(int)
    return X, y

if __name__ == "__main__":
    X, y = make_classification()
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmark/dataset.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = [f"f{i}" for i in range(X.shape[1])] + ["label"]
        w.writerow(header)
        for row, label in zip(X, y):
            w.writerow(list(row) + [label])
    print(f"Dataset written to {path}  ({X.shape[0]} obs, {X.shape[1]} features)")
