use boruta_rs::{Boruta, BorutaConfig, FeatureStatus};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generates a synthetic classification dataset where:
/// - `n_informative` features together determine the label (label = 1 if sum > 0)
/// - `n_noise` features are pure Uniform(0, 1) noise
///
/// Because the label depends on the SUM of informative features (not any single one),
/// each informative feature individually carries partial information. Permuting any
/// one of them changes the sum and therefore flips some labels → real permutation drop.
fn make_classification(
    n_obs: usize,
    n_informative: usize,
    n_noise: usize,
    seed: u64,
) -> (Array2<f64>, Array1<u32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_features = n_informative + n_noise;
    let mut data = vec![0.0f64; n_obs * n_features];
    let mut labels = Vec::with_capacity(n_obs);

    for i in 0..n_obs {
        // Each informative feature is N(class_mean, 1.0): class_mean ∈ {-0.5, +0.5}
        // Label is determined by the majority vote of features (sum > 0)
        // We generate features first without knowing the label, then assign label
        let mut sum = 0.0f64;
        for j in 0..n_informative {
            // Each feature is N(0, 1) + a tiny bias; label = sign(sum)
            let val: f64 = (rng.gen::<f64>() - 0.5) * 4.0; // Uniform(-2, 2)
            data[i * n_features + j] = val;
            sum += val;
        }
        // Label is the sign of the sum — each feature contributes equally
        let label: u32 = if sum > 0.0 { 1 } else { 0 };
        labels.push(label);

        // Noise features: independent Uniform(-2, 2)
        for j in n_informative..n_features {
            data[i * n_features + j] = (rng.gen::<f64>() - 0.5) * 4.0;
        }
    }

    let x = Array2::from_shape_vec((n_obs, n_features), data).unwrap();
    let y = Array1::from_vec(labels);
    (x, y)
}

#[test]
fn test_boruta_detects_informative_features() {
    let (x, y) = make_classification(500, 5, 5, 42);

    let config = BorutaConfig {
        max_iter: 100,
        p_value: 0.01,
        bonferroni: true,
        n_estimators: 100,
        random_seed: Some(42),
    };

    let result = Boruta::new(config).fit(&x, &y);

    println!("Statuses after {} iterations:", result.n_iterations);
    for (i, s) in result.statuses.iter().enumerate() {
        let kind = if i < 5 { "INFO" } else { "NOISE" };
        println!("  feature {:2} ({}): {:?}", i, kind, s);
    }

    // Informative features must be Confirmed
    for i in 0..5 {
        assert_eq!(
            result.statuses[i],
            FeatureStatus::Confirmed,
            "informative feature {} should be Confirmed, got {:?}",
            i,
            result.statuses[i]
        );
    }

    // Noise features must not be Confirmed
    for i in 5..10 {
        assert_ne!(
            result.statuses[i],
            FeatureStatus::Confirmed,
            "noise feature {} should not be Confirmed",
            i
        );
    }
}

#[test]
fn test_result_helpers() {
    let (x, y) = make_classification(200, 3, 3, 7);

    let config = BorutaConfig {
        max_iter: 30,
        p_value: 0.01,
        bonferroni: true,
        n_estimators: 30,
        random_seed: Some(7),
    };

    let result = Boruta::new(config).fit(&x, &y);

    // All indices should cover 0..6 exactly once
    let mut all_indices: Vec<usize> = result
        .confirmed_indices()
        .into_iter()
        .chain(result.rejected_indices())
        .chain(result.tentative_indices())
        .collect();
    all_indices.sort_unstable();
    assert_eq!(all_indices, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn test_json_serialisation() {
    let (x, y) = make_classification(100, 2, 2, 1);

    let config = BorutaConfig {
        max_iter: 10,
        p_value: 0.01,
        bonferroni: false,
        n_estimators: 20,
        random_seed: Some(1),
    };

    let result = Boruta::new(config).fit(&x, &y);
    let json = result.to_json();

    assert!(json.contains("statuses"));
    assert!(json.contains("n_iterations"));
    assert!(json.contains("importance_history"));
}
