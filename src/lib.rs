pub mod boruta;
pub mod decision;
pub mod importance;
pub mod shadow;
pub mod stats;

pub use boruta::{Boruta, BorutaConfig, BorutaResult, FeatureStatus};

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
