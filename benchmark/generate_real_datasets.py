"""
Generates CSV files for real-world benchmark datasets (Iris, Wine).
Requires scikit-learn.
"""
import csv
from sklearn.datasets import load_iris, load_wine

DATASETS = {
    "iris": (load_iris, ["sepal_length", "sepal_width", "petal_length", "petal_width"]),
    "wine": (load_wine, None),  # None → auto-named by sklearn
}

for name, (loader, feat_names) in DATASETS.items():
    ds = loader()
    names = feat_names or [n.replace(" ", "_").replace("(", "").replace(")", "") for n in ds.feature_names]
    path = f"benchmark/{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names + ["label"])
        for row, label in zip(ds.data, ds.target):
            w.writerow(list(row) + [int(label)])
    print(f"{name}.csv: {ds.data.shape[0]} obs × {ds.data.shape[1]} features, {len(set(ds.target))} classes → {path}")
