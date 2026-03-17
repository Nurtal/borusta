/// Rust benchmark — same dataset as Python/R benchmarks.
/// Run: cargo run --release --bin bench
use boruta_rs::{Boruta, BorutaConfig, FeatureStatus};
use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn load_csv(path: &str) -> (Array2<f64>, Array1<u32>) {
    let file = File::open(path).expect("cannot open dataset.csv");
    let mut lines = BufReader::new(file).lines();
    // skip header
    let header = lines.next().unwrap().unwrap();
    let n_features = header.split(',').count() - 1;

    let mut data: Vec<f64> = Vec::new();
    let mut labels: Vec<u32> = Vec::new();

    for line in lines {
        let line = line.unwrap();
        let vals: Vec<f64> = line.split(',').map(|v| v.parse().unwrap()).collect();
        labels.push(vals[n_features] as u32);
        data.extend_from_slice(&vals[..n_features]);
    }

    let n_obs = labels.len();
    let x = Array2::from_shape_vec((n_obs, n_features), data).unwrap();
    let y = Array1::from_vec(labels);
    (x, y)
}

fn main() {
    let (x, y) = load_csv("benchmark/dataset.csv");
    println!("Dataset: {} obs × {} features", x.nrows(), x.ncols());
    let pos = y.iter().filter(|&&v| v == 1).count();
    println!("Class balance: {:.2}% positive\n", pos as f64 / y.len() as f64 * 100.0);

    let config = BorutaConfig {
        max_iter: 100,
        p_value: 0.01,
        bonferroni: true,
        n_estimators: 100,
        random_seed: Some(42),
    };

    let t0 = Instant::now();
    let mut result = Boruta::new(config).fit(&x, &y);
    let elapsed = t0.elapsed();

    let feature_names: Vec<String> = (0..x.ncols()).map(|i| format!("f{}", i)).collect();
    result.feature_names = Some(feature_names.clone());

    println!("=== boruta-rs results ({} iterations, {:.2}s) ===",
        result.n_iterations, elapsed.as_secs_f64());

    let confirmed: Vec<&str> = result.confirmed_indices().iter().map(|&i| feature_names[i].as_str()).collect();
    let rejected:  Vec<&str> = result.rejected_indices().iter().map(|&i| feature_names[i].as_str()).collect();
    let tentative: Vec<&str> = result.tentative_indices().iter().map(|&i| feature_names[i].as_str()).collect();

    println!("  Confirmed  ({:2}): {:?}", confirmed.len(), confirmed);
    println!("  Rejected   ({:2}): {:?}", rejected.len(), rejected);
    println!("  Tentative  ({:2}): {:?}", tentative.len(), tentative);
    println!("\n  Elapsed: {:.3}s", elapsed.as_secs_f64());
}
