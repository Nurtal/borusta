# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**borusta-rs** — A Rust implementation of the Boruta algorithm, an "all-relevant" feature selection method based on Random Forest. The README (892 lines, in French) contains the full algorithm specification; no source code exists yet.

## Commands

```bash
cargo build                        # Build the project
cargo test                         # Run all tests
cargo test stats                   # Run a single test module
RUST_LOG=info cargo run --example iris  # Run an example with logging
cargo clippy                       # Lint
cargo fmt                          # Format
```

## Architecture

```
src/
├── lib.rs          # Public API: re-exports Boruta, BorutaConfig, BorutaResult, FeatureStatus
├── boruta.rs       # Main struct + core algorithm loop
├── shadow.rs       # Shadow feature generation (shuffled copies of real features)
├── importance.rs   # Random Forest feature importance extraction
├── stats.rs        # Z-scores, binomial tests, Bonferroni correction
└── decision.rs     # FeatureStatus enum (Confirmed/Rejected/Tentative) + decision logic
```

## Key dependencies

| Crate | Purpose |
|-------|---------|
| `ndarray` + `rayon` feature | Matrix manipulation (numpy equivalent) |
| `smartcore` (features: `serde`, `ndarray-bindings`) | Random Forest implementation — no `randomforest` feature exists |
| `serde` / `serde_json` | Serialization |
| `rand` / `rand_distr` | RNG and shuffling |
| `statrs` | Statistical distributions and tests |
| `rayon` | Parallelization |
| `log` / `env_logger` | Logging |
| `approx` (dev) | Floating-point comparisons in tests |

## Algorithm summary

Each iteration:
1. Shuffle each real feature column → shadow features
2. Train RF on `[real features | shadow features]` with `keep_samples=true`
3. Compute **OOB permutation importance** for every column — permute column j, measure OOB accuracy drop (avoids overfitting bias of train-set importance)
4. Count "hits": a feature scores a hit if its importance > max shadow importance this iteration
5. Apply binomial test (with optional Bonferroni correction) to confirm or reject
6. Remove rejected features; repeat until convergence or `max_iter`

Key types: `BorutaConfig` (max_iter, p_value, bonferroni, n_estimators, random_seed), `BorutaResult` (statuses, importance_history, n_iterations).

## Implementation notes

- **No `feature_importances()` in smartcore 0.3**: the method doesn't exist, so `importance.rs` implements OOB permutation importance from scratch.
- **`n_trees` is `u16`** in `RandomForestClassifierParameters`, not `usize`.
- **Column shuffling** in `shadow.rs` uses index-based assignment (not `as_slice_mut()`) because ndarray arrays are row-major and column views are non-contiguous.
- **Test data**: the integration test labels observations by the sign of the sum of informative features (`label = sign(Σ x_j)`), so each feature carries only partial information and permuting one has a measurable impact.
