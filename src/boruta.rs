use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::decision::update_decisions;
use crate::importance::{compute_importances, compute_importances_regression, split_importances};
use crate::shadow::create_shadow_matrix;

/// Status of a feature after selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeatureStatus {
    Confirmed,
    Rejected,
    Tentative,
}

/// Final result of the Boruta algorithm.
#[derive(Debug, Serialize, Deserialize)]
pub struct BorutaResult {
    pub statuses: Vec<FeatureStatus>,
    pub feature_names: Option<Vec<String>>,
    pub n_iterations: usize,
    /// importance_history[feature_index][iteration]
    pub importance_history: Vec<Vec<f64>>,
    /// max_shadow_history[iteration] = max shadow importance at that iteration.
    /// Used by `tentative_rough_fix` to resolve undecided features.
    #[serde(default)]
    pub max_shadow_history: Vec<f64>,
}

/// Configuration for the Boruta algorithm.
pub struct BorutaConfig {
    /// Maximum number of iterations (default: 100).
    pub max_iter: usize,
    /// Significance threshold for the binomial test (default: 0.01).
    pub p_value: f64,
    /// Whether to apply Bonferroni correction (default: true).
    pub bonferroni: bool,
    /// Number of trees in the Random Forest (default: 100).
    pub n_estimators: usize,
    pub random_seed: Option<u64>,
}

impl Default for BorutaConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            p_value: 0.01,
            bonferroni: true,
            n_estimators: 100,
            random_seed: None,
        }
    }
}

pub struct Boruta {
    config: BorutaConfig,
}

impl Boruta {
    pub fn new(config: BorutaConfig) -> Self {
        Self { config }
    }

    /// Runs the Boruta algorithm.
    ///
    /// # Arguments
    /// * `x` - Feature matrix [n_obs × n_features], f64
    /// * `y` - Target vector [n_obs], integer class labels
    ///
    /// # Returns
    /// A [`BorutaResult`] with the status of each feature.
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<u32>) -> BorutaResult {
        validate_inputs(x, y.len());
        let n_features = x.ncols();
        let seed_base = self.config.random_seed.unwrap_or(42);

        let mut statuses = vec![FeatureStatus::Tentative; n_features];
        // hits[i] = number of iterations where feature i beat the max shadow importance
        let mut hits = vec![0u64; n_features];
        // importance_history[feature][iteration]
        let mut importance_history: Vec<Vec<f64>> = vec![Vec::new(); n_features];
        let mut max_shadow_history: Vec<f64> = Vec::new();

        let mut n_iter = 0;

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            // Check convergence: all features decided?
            let all_decided = statuses.iter().all(|s| *s != FeatureStatus::Tentative);
            if all_decided {
                log::info!("Convergence reached at iteration {}", n_iter);
                break;
            }

            // Build active mask: include Tentative and Confirmed, exclude Rejected
            let active_mask: Vec<bool> = statuses
                .iter()
                .map(|s| *s != FeatureStatus::Rejected)
                .collect();

            let x_active = filter_columns(x, &active_mask);
            let n_active = x_active.ncols();

            // Step 1-2: extend with shadow features and train RF
            let x_extended = create_shadow_matrix(&x_active, seed_base + iter as u64);
            let importances = compute_importances(
                &x_extended,
                y,
                self.config.n_estimators,
                seed_base + iter as u64,
            );

            let (orig_imp, shadow_imp) = split_importances(&importances, n_active);

            // Step 3: MZSA — max shadow importance this iteration
            let max_shadow = shadow_imp
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            max_shadow_history.push(max_shadow);

            // Step 4: record importance and count hits (map active→original indices)
            let mut active_idx = 0;
            for orig_idx in 0..n_features {
                if statuses[orig_idx] == FeatureStatus::Rejected {
                    importance_history[orig_idx].push(0.0);
                } else {
                    let imp = orig_imp[active_idx];
                    importance_history[orig_idx].push(imp);
                    if imp > max_shadow {
                        hits[orig_idx] += 1;
                    }
                    active_idx += 1;
                }
            }

            // Step 5: update decisions via binomial test
            update_decisions(
                &hits,
                n_iter,
                self.config.p_value,
                self.config.bonferroni,
                &mut statuses,
            );

            log::debug!(
                "Iter {:3} | Confirmed: {} | Rejected: {} | Tentative: {}",
                n_iter,
                statuses
                    .iter()
                    .filter(|s| **s == FeatureStatus::Confirmed)
                    .count(),
                statuses
                    .iter()
                    .filter(|s| **s == FeatureStatus::Rejected)
                    .count(),
                statuses
                    .iter()
                    .filter(|s| **s == FeatureStatus::Tentative)
                    .count(),
            );
        }

        BorutaResult {
            statuses,
            feature_names: None,
            n_iterations: n_iter,
            importance_history,
            max_shadow_history,
        }
    }

    /// Runs the Boruta algorithm for **regression** targets (continuous `y: f64`).
    ///
    /// Feature importance is measured by OOB MSE drop after permutation.
    /// Everything else (shadow features, binomial test, convergence) is identical
    /// to the classification variant.
    pub fn fit_regression(&self, x: &Array2<f64>, y: &Array1<f64>) -> BorutaResult {
        validate_inputs(x, y.len());
        let n_features = x.ncols();
        let seed_base = self.config.random_seed.unwrap_or(42);

        let mut statuses = vec![FeatureStatus::Tentative; n_features];
        let mut hits = vec![0u64; n_features];
        let mut importance_history: Vec<Vec<f64>> = vec![Vec::new(); n_features];
        let mut max_shadow_history: Vec<f64> = Vec::new();
        let mut n_iter = 0;

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            let all_decided = statuses.iter().all(|s| *s != FeatureStatus::Tentative);
            if all_decided {
                log::info!("Convergence reached at iteration {}", n_iter);
                break;
            }

            let active_mask: Vec<bool> = statuses
                .iter()
                .map(|s| *s != FeatureStatus::Rejected)
                .collect();

            let x_active = filter_columns(x, &active_mask);
            let n_active = x_active.ncols();

            let x_extended = create_shadow_matrix(&x_active, seed_base + iter as u64);
            let importances = compute_importances_regression(
                &x_extended,
                y,
                self.config.n_estimators,
                seed_base + iter as u64,
            );

            let (orig_imp, shadow_imp) = split_importances(&importances, n_active);

            let max_shadow = shadow_imp
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            max_shadow_history.push(max_shadow);

            let mut active_idx = 0;
            for orig_idx in 0..n_features {
                if statuses[orig_idx] == FeatureStatus::Rejected {
                    importance_history[orig_idx].push(0.0);
                } else {
                    let imp = orig_imp[active_idx];
                    importance_history[orig_idx].push(imp);
                    if imp > max_shadow {
                        hits[orig_idx] += 1;
                    }
                    active_idx += 1;
                }
            }

            update_decisions(
                &hits,
                n_iter,
                self.config.p_value,
                self.config.bonferroni,
                &mut statuses,
            );

            log::debug!(
                "Iter {:3} | Confirmed: {} | Rejected: {} | Tentative: {}",
                n_iter,
                statuses.iter().filter(|s| **s == FeatureStatus::Confirmed).count(),
                statuses.iter().filter(|s| **s == FeatureStatus::Rejected).count(),
                statuses.iter().filter(|s| **s == FeatureStatus::Tentative).count(),
            );
        }

        BorutaResult {
            statuses,
            feature_names: None,
            n_iterations: n_iter,
            importance_history,
            max_shadow_history,
        }
    }
} // impl Boruta

/// Validates that `x` contains no NaN/Inf and that `y` has the same length.
/// Panics with a clear message on failure.
fn validate_inputs(x: &Array2<f64>, n_labels: usize) {
    assert_eq!(
        x.nrows(),
        n_labels,
        "x and y must have the same number of rows (x: {}, y: {})",
        x.nrows(),
        n_labels
    );
    assert!(
        x.iter().all(|v| v.is_finite()),
        "x contains NaN or Inf — impute or drop them before running Boruta"
    );
}

/// Selects columns of `x` according to a boolean mask.
fn filter_columns(x: &Array2<f64>, mask: &[bool]) -> Array2<f64> {
    let selected: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
        .collect();

    let n_rows = x.nrows();
    let n_cols = selected.len();

    if n_cols == 0 {
        return Array2::zeros((n_rows, 0));
    }

    let mut result = Array2::zeros((n_rows, n_cols));
    for (new_col, &old_col) in selected.iter().enumerate() {
        result.column_mut(new_col).assign(&x.column(old_col));
    }
    result
}
