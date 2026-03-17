/// Rust benchmark on the Iris dataset.
/// Run: cargo run --release --bin bench_iris
use boruta_rs::{Boruta, BorutaConfig};
use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn load_csv(path: &str) -> (Array2<f64>, Array1<u32>, Vec<String>) {
    let file = File::open(path).expect("cannot open CSV");
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().unwrap().unwrap();
    let cols: Vec<&str> = header.split(',').collect();
    let n_features = cols.len() - 1;
    let feat_names: Vec<String> = cols[..n_features].iter().map(|s| s.to_string()).collect();

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
    (x, y, feat_names)
}

fn main() {
    let (x, y, feat_names) = load_csv("benchmark/iris.csv");
    let n_classes = (*y.iter().max().unwrap() + 1) as usize;
    println!("Dataset: {} obs × {} features, {} classes\n", x.nrows(), x.ncols(), n_classes);

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

    result.feature_names = Some(feat_names.clone());

    println!("=== boruta-rs (Iris, {} iterations, {:.3}s) ===",
        result.n_iterations, elapsed.as_secs_f64());
    let confirmed: Vec<&str> = result.confirmed_indices().iter().map(|&i| feat_names[i].as_str()).collect();
    let rejected:  Vec<&str> = result.rejected_indices().iter().map(|&i| feat_names[i].as_str()).collect();
    let tentative: Vec<&str> = result.tentative_indices().iter().map(|&i| feat_names[i].as_str()).collect();
    println!("  Confirmed  ({:2}): {:?}", confirmed.len(), confirmed);
    println!("  Rejected   ({:2}): {:?}", rejected.len(), rejected);
    println!("  Tentative  ({:2}): {:?}", tentative.len(), tentative);
    println!("\n  Elapsed: {:.3}s", elapsed.as_secs_f64());
}
