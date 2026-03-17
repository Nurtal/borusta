use ndarray::{Array1, Array2};
use rand::Rng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier, DecisionTreeClassifierParameters,
};
use smartcore::tree::decision_tree_regressor::{
    DecisionTreeRegressor, DecisionTreeRegressorParameters,
};

// Concrete tree type used throughout — avoids repeating the four type params.
type Tree = DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>>;

// ── Parallel forest training ──────────────────────────────────────────────────

/// Trains `n_estimators` decision trees **in parallel** (rayon), each on a
/// bootstrap sample of the dataset.
///
/// Returns every tree together with a boolean OOB mask (true = sample was
/// NOT included in that tree's bootstrap).
fn train_forest_parallel(
    rows: &[Vec<f64>],
    y_vec: &[u32],
    n_estimators: usize,
    seed: u64,
) -> Vec<(Tree, Vec<bool>)> {
    let n_obs = rows.len();

    (0..n_estimators)
        .into_par_iter()
        .map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(i as u64));

            // Bootstrap: draw n_obs indices with replacement
            let indices: Vec<usize> =
                (0..n_obs).map(|_| rng.gen_range(0..n_obs)).collect();

            let mut in_bag = vec![false; n_obs];
            for &idx in &indices {
                in_bag[idx] = true;
            }
            let oob_mask: Vec<bool> = in_bag.into_iter().map(|b| !b).collect();

            // Build bootstrap matrices
            let boot_rows: Vec<Vec<f64>> =
                indices.iter().map(|&j| rows[j].clone()).collect();
            let y_boot: Vec<u32> = indices.iter().map(|&j| y_vec[j]).collect();
            let x_boot = DenseMatrix::from_2d_vec(&boot_rows);

            let params = DecisionTreeClassifierParameters {
                seed: Some(seed.wrapping_add(1_000_000).wrapping_add(i as u64)),
                ..Default::default()
            };
            let tree = DecisionTreeClassifier::fit(&x_boot, &y_boot, params)
                .expect("decision tree training failed");

            (tree, oob_mask)
        })
        .collect()
}

// ── OOB accuracy ─────────────────────────────────────────────────────────────

/// Computes OOB accuracy for the given row set.
///
/// For each sample, only trees whose bootstrap did NOT include that sample
/// vote on its class. Samples covered by no OOB tree are skipped.
fn oob_accuracy(
    forest: &[(Tree, Vec<bool>)],
    rows: &[Vec<f64>],
    y_vec: &[u32],
    n_classes: usize,
) -> f64 {
    let n_obs = rows.len();
    let x_sm = DenseMatrix::from_2d_vec(&rows.to_vec());

    // votes[sample][class] — accumulated across all OOB trees
    let mut votes: Vec<Vec<u32>> = vec![vec![0u32; n_classes]; n_obs];

    for (tree, oob_mask) in forest {
        let preds: Vec<u32> = tree.predict(&x_sm).expect("predict failed");
        for (i, (&pred, &is_oob)) in preds.iter().zip(oob_mask.iter()).enumerate() {
            if is_oob {
                let cls = pred as usize;
                if cls < n_classes {
                    votes[i][cls] += 1;
                }
            }
        }
    }

    let mut correct = 0usize;
    let mut n_valid = 0usize;
    for (vote_row, &true_label) in votes.iter().zip(y_vec.iter()) {
        if vote_row.iter().sum::<u32>() == 0 {
            continue; // sample has no OOB tree — skip
        }
        n_valid += 1;
        let predicted = vote_row
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(c, _)| c as u32)
            .unwrap_or(0);
        if predicted == true_label {
            correct += 1;
        }
    }

    if n_valid == 0 {
        0.0
    } else {
        correct as f64 / n_valid as f64
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Trains a Random Forest **in parallel** and returns OOB permutation importance
/// for every column of `x_extended` (length = `x_extended.ncols()`).
///
/// Two levels of parallelism via rayon:
/// 1. Tree training   — each of the `n_estimators` trees trains on its own
///    bootstrap sample independently.
/// 2. Permutation loop — each of the `n_cols` columns is permuted and evaluated
///    independently.
pub fn compute_importances(
    x_extended: &Array2<f64>,
    y: &Array1<u32>,
    n_estimators: usize,
    seed: u64,
) -> Vec<f64> {
    let n_cols = x_extended.ncols();
    let rows: Vec<Vec<f64>> = x_extended
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect();
    let y_vec: Vec<u32> = y.to_vec();
    let n_classes = (*y_vec.iter().max().unwrap_or(&0) as usize) + 1;

    // Step 1 — parallel tree training
    let forest = train_forest_parallel(&rows, &y_vec, n_estimators, seed);

    // Step 2 — baseline OOB accuracy
    let baseline_acc = oob_accuracy(&forest, &rows, &y_vec, n_classes);

    // Step 3 — parallel permutation importance
    (0..n_cols)
        .into_par_iter()
        .map(|j| {
            let mut rng = ChaCha8Rng::seed_from_u64(
                seed.wrapping_add(2_000_000).wrapping_add(j as u64),
            );
            let mut col_perm: Vec<f64> = x_extended.column(j).to_vec();
            col_perm.shuffle(&mut rng);

            let rows_perm: Vec<Vec<f64>> = rows
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    let mut r = row.clone();
                    r[j] = col_perm[i];
                    r
                })
                .collect();

            let perm_acc = oob_accuracy(&forest, &rows_perm, &y_vec, n_classes);
            (baseline_acc - perm_acc).max(0.0)
        })
        .collect()
}

/// Splits a flat importance vector into `(original_features, shadow_features)`.
pub fn split_importances(importances: &[f64], n_features: usize) -> (&[f64], &[f64]) {
    (&importances[..n_features], &importances[n_features..])
}

// ── Regression variant ────────────────────────────────────────────────────────

type TreeReg = DecisionTreeRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>;

fn train_forest_regression_parallel(
    rows: &[Vec<f64>],
    y_vec: &[f64],
    n_estimators: usize,
    seed: u64,
) -> Vec<(TreeReg, Vec<bool>)> {
    let n_obs = rows.len();

    (0..n_estimators)
        .into_par_iter()
        .map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(i as u64));

            let indices: Vec<usize> =
                (0..n_obs).map(|_| rng.gen_range(0..n_obs)).collect();

            let mut in_bag = vec![false; n_obs];
            for &idx in &indices {
                in_bag[idx] = true;
            }
            let oob_mask: Vec<bool> = in_bag.into_iter().map(|b| !b).collect();

            let boot_rows: Vec<Vec<f64>> =
                indices.iter().map(|&j| rows[j].clone()).collect();
            let y_boot: Vec<f64> = indices.iter().map(|&j| y_vec[j]).collect();
            let x_boot = DenseMatrix::from_2d_vec(&boot_rows);

            let params = DecisionTreeRegressorParameters {
                seed: Some(seed.wrapping_add(1_000_000).wrapping_add(i as u64)),
                ..Default::default()
            };
            let tree = DecisionTreeRegressor::fit(&x_boot, &y_boot, params)
                .expect("decision tree regressor training failed");

            (tree, oob_mask)
        })
        .collect()
}

/// Computes OOB Mean Squared Error for the regression forest.
fn oob_mse(forest: &[(TreeReg, Vec<bool>)], rows: &[Vec<f64>], y: &[f64]) -> f64 {
    let n_obs = rows.len();
    let x_sm = DenseMatrix::from_2d_vec(&rows.to_vec());

    // sum_pred[i] and count[i] accumulate predictions from OOB trees
    let mut sum_pred = vec![0.0f64; n_obs];
    let mut count = vec![0u32; n_obs];

    for (tree, oob_mask) in forest {
        let preds: Vec<f64> = tree.predict(&x_sm).expect("regressor predict failed");
        for (i, (&pred, &is_oob)) in preds.iter().zip(oob_mask.iter()).enumerate() {
            if is_oob {
                sum_pred[i] += pred;
                count[i] += 1;
            }
        }
    }

    let mut sq_err = 0.0f64;
    let mut n_valid = 0usize;
    for i in 0..n_obs {
        if count[i] == 0 {
            continue;
        }
        n_valid += 1;
        let pred = sum_pred[i] / count[i] as f64;
        sq_err += (pred - y[i]).powi(2);
    }

    if n_valid == 0 {
        0.0
    } else {
        sq_err / n_valid as f64
    }
}

/// Trains a regression Random Forest **in parallel** and returns OOB permutation
/// importance for every column of `x_extended`.
///
/// `importance[j] = (perm_mse_j - baseline_mse).max(0.0)` — a higher MSE after
/// permuting column j means that column carried useful information.
pub fn compute_importances_regression(
    x_extended: &Array2<f64>,
    y: &Array1<f64>,
    n_estimators: usize,
    seed: u64,
) -> Vec<f64> {
    let n_cols = x_extended.ncols();
    let rows: Vec<Vec<f64>> = x_extended
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect();
    let y_vec: Vec<f64> = y.to_vec();

    let forest = train_forest_regression_parallel(&rows, &y_vec, n_estimators, seed);
    let baseline_mse = oob_mse(&forest, &rows, &y_vec);

    (0..n_cols)
        .into_par_iter()
        .map(|j| {
            let mut rng = ChaCha8Rng::seed_from_u64(
                seed.wrapping_add(2_000_000).wrapping_add(j as u64),
            );
            let mut col_perm: Vec<f64> = x_extended.column(j).to_vec();
            col_perm.shuffle(&mut rng);

            let rows_perm: Vec<Vec<f64>> = rows
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    let mut r = row.clone();
                    r[j] = col_perm[i];
                    r
                })
                .collect();

            let perm_mse = oob_mse(&forest, &rows_perm, &y_vec);
            (perm_mse - baseline_mse).max(0.0)
        })
        .collect()
}
