use ndarray::{Array2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Concatenates `x` with independently shuffled copies of each column (shadow features).
///
/// Returns a matrix of shape [n_obs, 2 * n_features]:
/// - columns [0..n_features)          → original features
/// - columns [n_features..2*n_features) → shadow features
pub fn create_shadow_matrix(x: &Array2<f64>, seed: u64) -> Array2<f64> {
    let (_, n_features) = x.dim();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut shadow = x.clone();

    // Shuffle each column independently.
    // ndarray arrays are row-major, so columns are not contiguous; use index-based swap.
    for j in 0..n_features {
        let mut col: Vec<f64> = shadow.column(j).to_vec();
        col.shuffle(&mut rng);
        for (i, val) in col.into_iter().enumerate() {
            shadow[[i, j]] = val;
        }
    }

    ndarray::concatenate(Axis(1), &[x.view(), shadow.view()])
        .expect("both matrices must have the same number of rows")
}

/// Returns the range of column indices belonging to shadow features in the extended matrix.
pub fn shadow_indices(n_features: usize) -> std::ops::Range<usize> {
    n_features..(2 * n_features)
}
