use statrs::distribution::{Binomial, DiscreteCDF};

// ── Z-scores ──────────────────────────────────────────────────────────────────

/// Computes the Z-score of a cumulative importance history.
///
/// Z = mean(history) / std(history, ddof=1).
/// Returns 0.0 when the series is constant or has fewer than 2 points.
pub fn z_score(history: &[f64]) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let n = history.len() as f64;
    let mean = history.iter().sum::<f64>() / n;
    let variance = history.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    if std_dev < f64::EPSILON {
        0.0
    } else {
        mean / std_dev
    }
}

/// Computes Z-scores for a collection of importance histories.
/// `history[i]` contains the importance values of feature i across all iterations.
pub fn compute_z_scores(history: &[Vec<f64>]) -> Vec<f64> {
    history.iter().map(|h| z_score(h)).collect()
}

/// Returns the Maximum Z-score Among Shadow Attributes (MZSA).
pub fn mzsa(shadow_z_scores: &[f64]) -> f64 {
    shadow_z_scores
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
}

// ── Binomial tests ────────────────────────────────────────────────────────────

/// One-sided upper p-value: P(X ≥ hits) under H₀: p = 0.5.
/// Used to confirm a feature (it beats the shadow more often than chance).
pub fn p_value_upper(hits: u64, n_trials: u64) -> f64 {
    if n_trials == 0 {
        return 1.0;
    }
    let binom = Binomial::new(0.5, n_trials).expect("invalid binomial parameters");
    if hits == 0 {
        1.0
    } else {
        1.0 - binom.cdf(hits - 1)
    }
}

/// One-sided lower p-value: P(X ≤ hits) under H₀: p = 0.5.
/// Used to reject a feature (it rarely beats the shadow).
pub fn p_value_lower(hits: u64, n_trials: u64) -> f64 {
    if n_trials == 0 {
        return 1.0;
    }
    let binom = Binomial::new(0.5, n_trials).expect("invalid binomial parameters");
    binom.cdf(hits)
}

/// Adjusts the significance threshold via Bonferroni correction.
pub fn bonferroni_threshold(alpha: f64, n_undecided: usize) -> f64 {
    if n_undecided == 0 {
        alpha
    } else {
        alpha / n_undecided as f64
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_z_score_basic() {
        let h = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = z_score(&h);
        // mean = 3, std = sqrt(2.5) ≈ 1.5811 → z ≈ 1.897
        assert_relative_eq!(z, 1.897, epsilon = 0.01);
    }

    #[test]
    fn test_z_score_constant() {
        let h = vec![2.0, 2.0, 2.0];
        assert_eq!(z_score(&h), 0.0);
    }

    #[test]
    fn test_z_score_single() {
        assert_eq!(z_score(&[5.0]), 0.0);
    }

    #[test]
    fn test_p_value_upper_certain() {
        // A feature that beats the shadow every time should have a tiny upper p-value.
        let p = p_value_upper(100, 100);
        assert!(p < 1e-10);
    }

    #[test]
    fn test_p_value_upper_never() {
        // A feature with 0 hits has upper p-value = 1.
        let p = p_value_upper(0, 100);
        assert_relative_eq!(p, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_p_value_lower_never() {
        // A feature that never beats the shadow should have a tiny lower p-value.
        let p = p_value_lower(0, 100);
        assert!(p < 1e-10);
    }

    #[test]
    fn test_p_value_lower_always() {
        // A feature with hits = n_trials has lower p-value ≈ 1.
        let p = p_value_lower(100, 100);
        assert_relative_eq!(p, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_bonferroni() {
        let threshold = bonferroni_threshold(0.05, 10);
        assert_relative_eq!(threshold, 0.005, epsilon = 1e-9);
    }

    #[test]
    fn test_bonferroni_zero_undecided() {
        assert_relative_eq!(bonferroni_threshold(0.05, 0), 0.05, epsilon = 1e-9);
    }
}
