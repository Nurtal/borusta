use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;

/// Trains a Random Forest on `(x_extended, y)` and returns OOB permutation importance
/// for every column (length = `x_extended.ncols()`).
///
/// Importance[j] = drop in OOB accuracy when column j is randomly permuted.
/// Using OOB (out-of-bag) samples avoids the overfitting bias of train-set importance.
/// Values are clamped to ≥ 0 so that noise features don't produce negative importance.
pub fn compute_importances(
    x_extended: &Array2<f64>,
    y: &Array1<u32>,
    n_estimators: usize,
    seed: u64,
) -> Vec<f64> {
    let n_obs = x_extended.nrows();
    let n_cols = x_extended.ncols();

    // Convert ndarray → DenseMatrix (row-major Vec<Vec<f64>>)
    let rows: Vec<Vec<f64>> = x_extended
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();
    let x_sm = DenseMatrix::from_2d_vec(&rows);
    let y_vec: Vec<u32> = y.to_vec();

    // Train with keep_samples=true so OOB prediction is available
    let params = RandomForestClassifierParameters::default()
        .with_n_trees(n_estimators as u16)
        .with_seed(seed)
        .with_keep_samples(true);

    let rf = RandomForestClassifier::fit(&x_sm, &y_vec, params)
        .expect("Random Forest training failed");

    // Baseline OOB accuracy (each sample predicted only by trees that did not see it)
    let baseline_preds: Vec<u32> = rf.predict_oob(&x_sm).expect("OOB predict failed");
    let baseline_acc: f64 = baseline_preds
        .iter()
        .zip(y_vec.iter())
        .filter(|(p, t)| p == t)
        .count() as f64
        / n_obs as f64;

    // OOB permutation importance: for each column, shuffle it and measure OOB accuracy drop
    let mut importances = vec![0.0f64; n_cols];
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1_000_000));

    for j in 0..n_cols {
        let mut col_perm: Vec<f64> = x_extended.column(j).to_vec();
        col_perm.shuffle(&mut rng);

        // Build a new row set with column j replaced by the permuted values
        let rows_perm: Vec<Vec<f64>> = rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut r = row.clone();
                r[j] = col_perm[i];
                r
            })
            .collect();
        let x_perm_sm = DenseMatrix::from_2d_vec(&rows_perm);

        // predict_oob checks that the row count matches training data — it does (same n_obs)
        let perm_preds: Vec<u32> = rf.predict_oob(&x_perm_sm).expect("OOB predict failed");
        let perm_acc: f64 = perm_preds
            .iter()
            .zip(y_vec.iter())
            .filter(|(p, t)| p == t)
            .count() as f64
            / n_obs as f64;

        importances[j] = (baseline_acc - perm_acc).max(0.0);
    }

    importances
}

/// Splits a flat importance vector into `(original_features, shadow_features)`.
pub fn split_importances(importances: &[f64], n_features: usize) -> (&[f64], &[f64]) {
    (&importances[..n_features], &importances[n_features..])
}
