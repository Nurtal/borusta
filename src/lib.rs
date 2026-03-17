pub mod boruta;
pub mod decision;
pub mod importance;
pub mod shadow;
pub mod stats;

pub use boruta::{Boruta, BorutaConfig, BorutaResult, FeatureStatus};

fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

impl BorutaResult {
    /// Indices of Confirmed features.
    pub fn confirmed_indices(&self) -> Vec<usize> {
        self.statuses
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if *s == FeatureStatus::Confirmed {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Indices of Rejected features.
    pub fn rejected_indices(&self) -> Vec<usize> {
        self.statuses
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if *s == FeatureStatus::Rejected {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Indices of Tentative features (undecided after `max_iter`).
    pub fn tentative_indices(&self) -> Vec<usize> {
        self.statuses
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if *s == FeatureStatus::Tentative {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Serialises the result to a pretty-printed JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialisation error")
    }

    /// Resolves remaining Tentative features using a median threshold (heuristic).
    ///
    /// Mirrors the R Boruta package's `TentativeRoughFix`:
    /// - Compute the median of `max_shadow_history` (per-iteration max shadow importance).
    /// - For each Tentative feature, if its median importance > that threshold → Confirmed,
    ///   otherwise → Rejected.
    ///
    /// This is a **heuristic post-processing step**, not a statistical test. Call it only
    /// after `fit()` / `fit_regression()` when some features remain Tentative.
    pub fn tentative_rough_fix(&mut self) {
        let tentative = self.tentative_indices();
        if tentative.is_empty() || self.max_shadow_history.is_empty() {
            return;
        }

        let median_shadow = {
            let mut v = self.max_shadow_history.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            median_sorted(&v)
        };

        for i in tentative {
            let median_feat = {
                let mut v = self.importance_history[i].clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                median_sorted(&v)
            };
            self.statuses[i] = if median_feat > median_shadow {
                FeatureStatus::Confirmed
            } else {
                FeatureStatus::Rejected
            };
        }
    }

    /// Returns the importance history as a CSV string.
    ///
    /// Format: one row per iteration, one column per feature.
    /// Header: `iteration,f0,f1,...` (or named features if `feature_names` is set).
    pub fn importance_history_to_csv(&self) -> String {
        let n_features = self.importance_history.len();
        if n_features == 0 {
            return String::new();
        }
        let n_iter = self.importance_history[0].len();

        let names: Vec<String> = match &self.feature_names {
            Some(names) => names.clone(),
            None => (0..n_features).map(|i| format!("f{}", i)).collect(),
        };

        let mut out = String::from("iteration,");
        out.push_str(&names.join(","));
        out.push('\n');

        for iter in 0..n_iter {
            out.push_str(&(iter + 1).to_string());
            for feat in 0..n_features {
                out.push(',');
                let val = self.importance_history[feat].get(iter).copied().unwrap_or(0.0);
                out.push_str(&format!("{:.6}", val));
            }
            out.push('\n');
        }
        out
    }

    /// Writes the importance history CSV to `path`.
    pub fn write_importance_csv(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.importance_history_to_csv())
    }

    /// Prints a human-readable summary to stdout.
    pub fn summary(&self) {
        println!("=== Boruta result ({} iterations) ===", self.n_iterations);
        println!("  Confirmed : {:?}", self.confirmed_indices());
        println!("  Rejected  : {:?}", self.rejected_indices());
        println!("  Tentative : {:?}", self.tentative_indices());
        if let Some(names) = &self.feature_names {
            println!("\nConfirmed features:");
            for i in self.confirmed_indices() {
                println!("  - {}", names[i]);
            }
        }
    }
}
