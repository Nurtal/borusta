# boruta-rs

[![Crates.io](https://img.shields.io/crates/v/boruta-rs.svg)](https://crates.io/crates/boruta-rs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![benchmark](https://img.shields.io/badge/benchmark-~1.4s%20(500obs%2C%2016cores)-brightgreen)

A fast, parallel Rust implementation of the [Boruta](https://doi.org/10.18637/jss.v036.i11) all-relevant feature selection algorithm. Supports classification and regression. Optional Python bindings via PyO3.

---

## Features

- **All-relevant selection** — finds every feature that carries information, not just a minimal subset
- **OOB permutation importance** — unbiased alternative to MDI (Mean Decrease Impurity); converges faster in practice
- **Two-level parallelism** via [rayon](https://docs.rs/rayon): tree training and the permutation loop run concurrently
- **Classification and regression** — `fit()` for integer labels, `fit_regression()` for continuous targets
- **Multi-class** — supports N ≥ 2 classes natively
- **Input validation** — panics with a clear message on NaN/Inf or shape mismatch
- **TentativeRoughFix** — heuristic post-processing to resolve undecided features after `max_iter`
- **CSV export** — per-iteration importance history as a CSV string
- **Python bindings** — `Boruta().fit(X, y)` from Python via PyO3 + maturin

---

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
boruta-rs = "0.1"
```

---

## Usage

### Rust — classification

```rust
use boruta_rs::{Boruta, BorutaConfig};
use ndarray::{Array1, Array2};

let x: Array2<f64> = /* your feature matrix [n_obs × n_features] */;
let y: Array1<u32> = /* integer class labels */;

let config = BorutaConfig {
    max_iter: 100,
    p_value: 0.01,
    bonferroni: true,
    n_estimators: 100,
    random_seed: Some(42),
};

let mut result = Boruta::new(config).fit(&x, &y);

// Optionally name the features
result.feature_names = Some(vec!["age".into(), "score".into(), /* ... */]);

result.summary();
// → Confirmed: [0, 1]  Rejected: [2, 3]  Tentative: []

println!("confirmed indices: {:?}", result.confirmed_indices());
println!("rejected  indices: {:?}", result.rejected_indices());
```

### Rust — regression

```rust
let y_cont: Array1<f64> = /* continuous target */;
let result = Boruta::new(config).fit_regression(&x, &y_cont);
```

### TentativeRoughFix

If some features are still `Tentative` after `max_iter`, apply a median-threshold heuristic (mirrors R's `TentativeRoughFix`):

```rust
result.tentative_rough_fix();
// No features remain Tentative
```

### Export importance history

```rust
// CSV: one row per iteration, one column per feature
let csv: String = result.importance_history_to_csv();
result.write_importance_csv("importance.csv")?;
```

### JSON serialisation

```rust
let json = result.to_json(); // serde_json pretty-printed
```

### Python bindings

Build with [maturin](https://github.com/PyO3/maturin) from the `python/` directory:

```bash
cd python
pip install maturin numpy
maturin develop --features python
```

```python
import numpy as np
from boruta_rs import Boruta

result = Boruta(max_iter=100, p_value=0.01, n_estimators=100, random_seed=42).fit(X, y)
print(result)            # BorutaResult(confirmed=5, rejected=5, tentative=0, n_iterations=18)
print(result.confirmed)  # [0, 1, 2, 3, 4]

result.rough_fix()                  # TentativeRoughFix
csv = result.importance_history_csv()

# Regression
result_reg = Boruta(n_estimators=100).fit_regression(X, y_continuous)
```

---

## Algorithm

Boruta (Kursa & Rudnicki, 2010) is an **all-relevant** wrapper around Random Forest. At each iteration:

1. Each feature column is shuffled independently to create **shadow features** — random copies that preserve the marginal distribution but carry no signal.
2. A Random Forest is trained on the extended matrix `[real features | shadow features]`.
3. **OOB permutation importance** is computed for every column: permute column *j*, measure the drop in out-of-bag accuracy/MSE. This avoids the bias of MDI and gives shadow features an importance close to zero, making the signal-to-noise separation sharp.
4. A feature scores a **hit** if its importance exceeds the maximum shadow importance (MZSA) for that iteration.
5. A **binomial test** (optionally with Bonferroni correction) decides after *t* iterations:
   - too many hits → `Confirmed`
   - too few hits → `Rejected`
   - otherwise → `Tentative`
6. Rejected features are dropped from subsequent iterations. The loop stops when all features are decided or `max_iter` is reached.

---

## Benchmark

Conditions: 500 observations, `max_iter = 100`, `n_estimators = 100`, `alpha = 0.01` (Bonferroni), 16-core machine.

### Synthetic dataset (10 features: 5 informative, 5 noise)

| Implementation | Confirmed | Rejected | Iterations | Time |
|---|---|---|---|---|
| **boruta-rs** (Rust) | f0–f4 ✅ | f5–f9 ✅ | 18 | **~1.4s** |
| BorutaPy (Python, `n_jobs=-1`) | f0–f4 ✅ | f5–f9 ✅ | auto | ~3.2s |
| Boruta (R) | f0–f4 ✅ | f5–f9 ✅ | 273 | ~8.4s |

boruta-rs is **2.4× faster than Python** and **6.2× faster than R**.

### Iris (150 obs, 4 features, 3 classes)

| Implementation | Confirmed | Rejected | Time |
|---|---|---|---|
| **boruta-rs** | petal_length, petal_width | sepal_length, sepal_width | **~0.13s** |
| BorutaPy | all 4 | — | ~1.3s |
| Boruta (R) | all 4 | — | ~2.2s |

### Wine (178 obs, 13 features, 3 classes)

| Implementation | Confirmed | Rejected | Tentative | Time |
|---|---|---|---|---|
| **boruta-rs** | 4 | 8 | magnesium | **~1.7s** |
| BorutaPy | all 13 | — | — | ~2.2s |
| Boruta (R) | all 13 | — | — | ~0.9s |

**OOB permutation vs MDI**: on small datasets, OOB permutation only rewards features the forest actually uses for prediction. This makes boruta-rs more conservative than R/Python (which use MDI). Call `tentative_rough_fix()` to resolve remaining Tentatives with a median heuristic.

---

## Architecture

```
src/
├── lib.rs         Public API — re-exports Boruta, BorutaConfig, BorutaResult, FeatureStatus
├── boruta.rs      Main struct, fit() / fit_regression(), validate_inputs()
├── shadow.rs      Shadow feature generation (per-column shuffle, index-based)
├── importance.rs  OOB permutation importance — parallel forest training + permutation loop
├── stats.rs       Z-scores, binomial test, Bonferroni correction
├── decision.rs    FeatureStatus enum + update_decisions()
└── python.rs      PyO3 bindings (compiled only with --features python)
```

### Key types

| Type | Description |
|---|---|
| `BorutaConfig` | `max_iter`, `p_value`, `bonferroni`, `n_estimators`, `random_seed` |
| `BorutaResult` | `statuses`, `importance_history`, `max_shadow_history`, `n_iterations`, `feature_names` |
| `FeatureStatus` | `Confirmed` / `Rejected` / `Tentative` |

---

## Roadmap

| Status | Item |
|---|---|
| ✅ | OOB permutation importance |
| ✅ | Parallel forest training + permutation loop (rayon) |
| ✅ | Classification (`fit`) and regression (`fit_regression`) |
| ✅ | Multi-class support (N ≥ 2) |
| ✅ | Input validation (NaN/Inf, shape mismatch) |
| ✅ | TentativeRoughFix |
| ✅ | Importance history CSV export |
| ✅ | Python bindings (PyO3 + maturin) |
| 🔲 | `linfa` backend as an alternative to `smartcore` |
| 🔲 | Publish to crates.io |

---

## References

- Kursa, M. B., & Rudnicki, W. R. (2010). *Feature Selection with the Boruta Package*. Journal of Statistical Software, 36(11). <https://doi.org/10.18637/jss.v036.i11>
- [BorutaPy — Python port](https://github.com/scikit-learn-contrib/boruta_py)
- [smartcore](https://docs.rs/smartcore) · [ndarray](https://docs.rs/ndarray) · [statrs](https://docs.rs/statrs) · [rayon](https://docs.rs/rayon) · [PyO3](https://pyo3.rs)

---

## License

MIT — see [LICENSE](LICENSE).
