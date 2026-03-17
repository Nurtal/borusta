use crate::boruta::FeatureStatus;
use crate::stats::{bonferroni_threshold, p_value_lower, p_value_upper};

/// Updates the status of each Tentative feature via a binomial test.
///
/// # Arguments
/// * `hits`      - `hits[i]` = number of iterations where feature i beat the max shadow importance
/// * `n_iter`    - Total iterations completed so far
/// * `alpha`     - Significance threshold (e.g. 0.01)
/// * `bonferroni`- Whether to apply Bonferroni correction
/// * `statuses`  - Mutable slice of feature statuses (only Tentative entries are updated)
pub fn update_decisions(
    hits: &[u64],
    n_iter: usize,
    alpha: f64,
    bonferroni: bool,
    statuses: &mut Vec<FeatureStatus>,
) {
    let n_undecided = statuses
        .iter()
        .filter(|s| **s == FeatureStatus::Tentative)
        .count();

    let threshold = if bonferroni {
        bonferroni_threshold(alpha, n_undecided)
    } else {
        alpha
    };

    for (i, status) in statuses.iter_mut().enumerate() {
        if *status != FeatureStatus::Tentative {
            continue;
        }

        let p_up = p_value_upper(hits[i], n_iter as u64);
        let p_lo = p_value_lower(hits[i], n_iter as u64);

        if p_up < threshold {
            *status = FeatureStatus::Confirmed;
            log::info!("Feature {} → Confirmed (p_up={:.4})", i, p_up);
        } else if p_lo < threshold {
            *status = FeatureStatus::Rejected;
            log::info!("Feature {} → Rejected  (p_lo={:.4})", i, p_lo);
        }
        // Otherwise: stays Tentative
    }
}
