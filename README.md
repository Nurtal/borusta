# boruta-rs

> Implémentation de l'algorithme Boruta en Rust — sélection de features "all-relevant" basée sur Random Forest.

---

## Table des matières

1. [Contexte et principe de l'algorithme](#1-contexte-et-principe-de-lalgorithme)
2. [Architecture du projet](#2-architecture-du-projet)
3. [Dépendances (Cargo.toml)](#3-dépendances-cargotoml)
4. [Étape 1 — Représentation des données](#4-étape-1--représentation-des-données)
5. [Étape 2 — Création des shadow features](#5-étape-2--création-des-shadow-features)
6. [Étape 3 — Entraînement Random Forest et importance des features](#6-étape-3--entraînement-random-forest-et-importance-des-features)
7. [Étape 4 — Calcul des Z-scores](#7-étape-4--calcul-des-z-scores)
8. [Étape 5 — Test statistique et prise de décision](#8-étape-5--test-statistique-et-prise-de-décision)
9. [Étape 6 — Boucle principale et convergence](#9-étape-6--boucle-principale-et-convergence)
10. [Étape 7 — API publique et résultat](#10-étape-7--api-publique-et-résultat)
11. [Exemple d'utilisation complet](#11-exemple-dutilisation-complet)
12. [Tests](#12-tests)
13. [Feuille de route](#13-feuille-de-route)

---

## 1. Contexte et principe de l'algorithme

Boruta est un algorithme de **sélection de features "all-relevant"**, publié par Kursa & Rudnicki (2010). Contrairement aux méthodes "minimal-optimal" (comme la RFE), Boruta cherche *toutes* les features qui portent une information utile, pas seulement un sous-ensemble minimal.

### Fonctionnement en résumé

```
Pour chaque itération jusqu'à max_iter ou convergence :
  1. Créer des copies shufflées de chaque feature (shadow features)
  2. Entraîner une Random Forest sur [features originales + shadows]
  3. Récupérer l'importance de chaque feature
  4. Calculer le Z-score de chaque feature sur l'historique
  5. Comparer le Z-score de chaque feature au max des Z-scores shadow (MZSA)
  6. Test binomial :
     - Z-score >> MZSA  → feature confirmée (Confirmed)
     - Z-score << MZSA  → feature rejetée (Rejected)
     - Sinon            → feature indéterminée (Tentative)
  7. Retirer les features rejetées du dataset
  8. Répéter jusqu'à ce que toutes les features soient tranchées
```

---

## 2. Architecture du projet

```
boruta-rs/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs           # Point d'entrée public
    ├── boruta.rs        # Struct principale + boucle algorithmique
    ├── shadow.rs        # Génération des shadow features
    ├── importance.rs    # Extraction de l'importance depuis la RF
    ├── stats.rs         # Z-scores, test binomial, correction Bonferroni
    └── decision.rs      # Enum FeatureStatus + logique de décision
```

---

## 3. Dépendances (Cargo.toml)

```toml
[package]
name    = "boruta-rs"
version = "0.1.0"
edition = "2021"

[dependencies]

# Algèbre linéaire et manipulation de matrices (équivalent numpy)
ndarray = { version = "0.16", features = ["rayon"] }

# Sérialisation/désérialisation (utile pour sauvegarder les résultats)
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"

# Random Forest — le crate le plus complet de l'écosystème Rust ML
# Fournit DecisionTree, RandomForest, feature importances (MDI & permutation)
smartcore = { version = "0.3", features = ["randomforest"] }

# Alternative possible : linfa + linfa-trees
# linfa       = "0.7"
# linfa-trees = "0.7"

# Génération de nombres aléatoires (shuffle des shadow features)
rand       = "0.8"
rand_distr = "0.4"   # distributions (Binomial, Normal, etc.)

# Statistiques : test binomial, correction Bonferroni
# statrs fournit les distributions de probabilité nécessaires
statrs = "0.16"

# Parallélisation des itérations de la Random Forest
rayon = "1.10"

# Logging
log     = "0.4"
env_logger = "0.11"

[dev-dependencies]
# Jeux de données synthétiques pour les tests
approx = "0.5"   # Comparaisons flottantes dans les tests
```

### Rôle de chaque crate

| Crate | Rôle dans Boruta |
|---|---|
| `ndarray` | Stockage et manipulation des matrices X (features × observations). Opérations de slicing, shuffling par colonne, concaténation. |
| `smartcore` | Entraînement de la Random Forest à chaque itération. Fournit `feature_importances_` (MDI — Mean Decrease Impurity). |
| `rand` + `rand_distr` | Shuffling aléatoire des colonnes pour créer les shadow features. Reproductibilité via `SeedableRng`. |
| `statrs` | Test binomial (décision Confirmed/Rejected), correction de Bonferroni pour les tests multiples, calcul des p-values. |
| `rayon` | Parallélisation des arbres dans la Random Forest (`ndarray` + `rayon` via feature flag). |
| `serde` + `serde_json` | Sérialisation du résultat `BorutaResult` (liste de features confirmées, rejetées, tentatives). |

---

## 4. Étape 1 — Représentation des données

**Fichier : `src/lib.rs` et `src/boruta.rs`**

Toutes les données sont représentées sous forme de matrices `ndarray::Array2<f64>`.

```rust
// src/boruta.rs
use ndarray::{Array1, Array2};

/// Statut d'une feature après sélection
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FeatureStatus {
    Confirmed,
    Rejected,
    Tentative,
}

/// Résultat final de l'algorithme
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BorutaResult {
    pub statuses: Vec<FeatureStatus>,
    pub feature_names: Option<Vec<String>>,
    pub n_iterations: usize,
    pub importance_history: Vec<Vec<f64>>, // [itération][feature]
}

/// Configuration de l'algorithme
pub struct BorutaConfig {
    pub max_iter: usize,      // Nombre max d'itérations (défaut : 100)
    pub p_value: f64,         // Seuil du test binomial (défaut : 0.01)
    pub bonferroni: bool,     // Appliquer correction de Bonferroni (défaut : true)
    pub n_estimators: usize,  // Arbres dans la Random Forest (défaut : 100)
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

/// Struct principale
pub struct Boruta {
    config: BorutaConfig,
}
```

---

## 5. Étape 2 — Création des shadow features

**Fichier : `src/shadow.rs`**

Pour chaque feature originale, on crée une copie dont les valeurs sont **shufflées aléatoirement** sur l'axe des observations. Cela casse toute corrélation avec la variable cible tout en préservant la distribution marginale.

```rust
// src/shadow.rs
use ndarray::{Array2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Concatène X avec ses copies shufflées (shadow features).
/// Retourne une matrice de dimension [n_obs, 2 * n_features].
///
/// Les colonnes [0..n_features]          → features originales
/// Les colonnes [n_features..2*n_features] → shadow features
pub fn create_shadow_matrix(x: &Array2<f64>, seed: u64) -> Array2<f64> {
    let (n_obs, n_features) = x.dim();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Copie de X pour les shadows
    let mut shadow = x.clone();

    // Shuffle indépendant colonne par colonne
    for mut col in shadow.axis_iter_mut(Axis(1)) {
        // Convertir la vue en slice mutable pour SliceRandom
        let slice = col.as_slice_mut()
            .expect("La matrice doit être contiguous en mémoire");
        slice.shuffle(&mut rng);
    }

    // Concaténation horizontale : [X | shadow]
    ndarray::concatenate(Axis(1), &[x.view(), shadow.view()])
        .expect("Les matrices doivent avoir le même nombre de lignes")
}

/// Retourne uniquement les indices des colonnes shadow dans la matrice étendue.
pub fn shadow_indices(n_features: usize) -> std::ops::Range<usize> {
    n_features..(2 * n_features)
}
```

> **Pourquoi `ChaCha8Rng` ?**
> `rand_chacha` est reproductible, rapide, et disponible dans `rand`. Il suffit de passer un `seed` fixe pour rendre les expériences déterministes.

---

## 6. Étape 3 — Entraînement Random Forest et importance des features

**Fichier : `src/importance.rs`**

On utilise `smartcore::ensemble::random_forest_classifier::RandomForestClassifier` (ou `Regressor` pour la régression). Après fit, `feature_importances()` retourne le **Mean Decrease Impurity (MDI)** pour chaque colonne de la matrice étendue.

```rust
// src/importance.rs
use ndarray::{Array1, Array2};
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;

/// Entraîne une Random Forest sur (x_extended, y) et retourne
/// le vecteur d'importance de chaque colonne (longueur = 2 * n_features).
pub fn compute_importances(
    x_extended: &Array2<f64>,
    y: &Array1<u32>,
    n_estimators: usize,
    seed: u64,
) -> Vec<f64> {
    let (n_obs, n_cols) = x_extended.dim();

    // Conversion ndarray → DenseMatrix (format attendu par smartcore)
    let x_sm = DenseMatrix::from_2d_vec(
        &x_extended
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    );

    let y_vec: Vec<u32> = y.to_vec();

    // Paramètres de la Random Forest
    let params = RandomForestClassifierParameters::default()
        .with_n_trees(n_estimators)
        .with_seed(seed);

    let rf = RandomForestClassifier::fit(&x_sm, &y_vec, params)
        .expect("Échec de l'entraînement Random Forest");

    // feature_importances() retourne un Vec<f64> de longueur n_cols
    rf.feature_importances()
        .expect("Impossible d'extraire les importances")
}

/// Sépare le vecteur d'importances en :
///   - importances des features originales (indices 0..n_features)
///   - importances des shadow features     (indices n_features..2*n_features)
pub fn split_importances(
    importances: &[f64],
    n_features: usize,
) -> (&[f64], &[f64]) {
    (&importances[..n_features], &importances[n_features..])
}
```

> **Note sur `linfa` comme alternative :**
> Si vous préférez `linfa` + `linfa-trees`, l'API diffère légèrement mais le principe est identique : `linfa_trees::RandomForest::fit(dataset)` puis `.feature_importances()`. Consultez la doc de `linfa` pour l'intégration avec `linfa::Dataset`.

---

## 7. Étape 4 — Calcul des Z-scores

**Fichier : `src/stats.rs`**

Le Z-score d'une feature à l'itération `t` est calculé sur **l'historique cumulé** de ses importances :

```
Z = mean(importances_historique) / std(importances_historique)
```

L'algorithme compare ensuite chaque Z-score original au **maximum des Z-scores shadow** (MZSA).

```rust
// src/stats.rs — partie Z-score

/// Calcule le Z-score d'une série d'importances accumulées.
/// Retourne 0.0 si l'écart-type est nul (feature constante).
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

/// Calcule les Z-scores pour toutes les features (originales et shadows).
/// `history[i]` = Vec des importances de la feature i sur toutes les itérations.
pub fn compute_z_scores(history: &[Vec<f64>]) -> Vec<f64> {
    history.iter().map(|h| z_score(h)).collect()
}

/// Retourne le Maximum Z-Score among Shadow Attributes (MZSA).
pub fn mzsa(shadow_z_scores: &[f64]) -> f64 {
    shadow_z_scores
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
}
```

---

## 8. Étape 5 — Test statistique et prise de décision

**Fichier : `src/stats.rs` (suite) + `src/decision.rs`**

La décision repose sur un **test binomial** : à chaque itération, une feature "gagne" un hit si son importance dépasse le MZSA. Après `t` itérations, la probabilité d'obtenir `k` hits ou plus par hasard suit une loi `Binomial(t, 0.5)`.

```rust
// src/stats.rs — partie test binomial

use statrs::distribution::{Binomial, ContinuousCDF, DiscreteCDF};

/// p-value unilatérale supérieure : P(X >= k) sous H0 : p = 0.5
/// Utilisé pour confirmer une feature (elle bat souvent les shadows).
pub fn p_value_upper(hits: u64, n_trials: u64) -> f64 {
    if n_trials == 0 {
        return 1.0;
    }
    let binom = Binomial::new(0.5, n_trials)
        .expect("Paramètres binomiaux invalides");
    // P(X >= hits) = 1 - P(X <= hits - 1)
    if hits == 0 {
        1.0
    } else {
        1.0 - binom.cdf(hits - 1)
    }
}

/// p-value unilatérale inférieure : P(X <= k) sous H0 : p = 0.5
/// Utilisé pour rejeter une feature (elle bat rarement les shadows).
pub fn p_value_lower(hits: u64, n_trials: u64) -> f64 {
    if n_trials == 0 {
        return 1.0;
    }
    let binom = Binomial::new(0.5, n_trials)
        .expect("Paramètres binomiaux invalides");
    binom.cdf(hits)
}

/// Applique la correction de Bonferroni au seuil alpha
/// en divisant par le nombre de features encore indéterminées.
pub fn bonferroni_threshold(alpha: f64, n_undecided: usize) -> f64 {
    if n_undecided == 0 {
        alpha
    } else {
        alpha / n_undecided as f64
    }
}
```

```rust
// src/decision.rs

use crate::stats::{bonferroni_threshold, p_value_lower, p_value_upper};
use crate::boruta::FeatureStatus;

/// Met à jour le statut de chaque feature encore indéterminée.
///
/// - hits[i]     : nombre de fois que la feature i a battu le MZSA
/// - n_iter      : nombre d'itérations effectuées jusqu'ici
/// - alpha       : seuil de signification (typiquement 0.01)
/// - bonferroni  : appliquer la correction Bonferroni
/// - statuses    : tableau mutable des statuts (Tentative en entrée possible)
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
            continue; // déjà tranché, on ne revient pas en arrière
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
        // Sinon : reste Tentative
    }
}
```

---

## 9. Étape 6 — Boucle principale et convergence

**Fichier : `src/boruta.rs`**

C'est le cœur de l'algorithme. La boucle s'arrête quand :
- toutes les features sont **Confirmed** ou **Rejected**, ou
- `max_iter` est atteint (les Tentative restantes sont conservées comme telles).

```rust
// src/boruta.rs

use ndarray::{Array1, Array2};
use crate::shadow::{create_shadow_matrix, shadow_indices};
use crate::importance::{compute_importances, split_importances};
use crate::stats::{compute_z_scores, mzsa};
use crate::decision::update_decisions;

impl Boruta {
    pub fn new(config: BorutaConfig) -> Self {
        Self { config }
    }

    /// Exécute l'algorithme Boruta.
    ///
    /// # Arguments
    /// * `x` - Matrice des features [n_obs × n_features], type f64
    /// * `y` - Vecteur cible [n_obs], labels entiers (classification)
    ///
    /// # Retourne
    /// Un `BorutaResult` avec le statut de chaque feature.
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<u32>) -> BorutaResult {
        let n_features = x.ncols();
        let seed_base = self.config.random_seed.unwrap_or(42);

        // --- Initialisation ---
        let mut statuses = vec![FeatureStatus::Tentative; n_features];
        let mut hits = vec![0u64; n_features];
        // Historique des importances : importance_history[feature][itération]
        let mut importance_history: Vec<Vec<f64>> = vec![Vec::new(); n_features];

        let mut n_iter = 0;

        // --- Boucle principale ---
        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            // Vérification de convergence : toutes les features ont-elles une décision ?
            let all_decided = statuses
                .iter()
                .all(|s| *s != FeatureStatus::Tentative);
            if all_decided {
                log::info!("Convergence atteinte à l'itération {}", n_iter);
                break;
            }

            // 1. Créer la matrice étendue [X_actif | shadows]
            //    On garde uniquement les features non-rejetées pour alléger la RF.
            let active_mask: Vec<bool> = statuses
                .iter()
                .map(|s| *s != FeatureStatus::Rejected)
                .collect();

            let x_active = filter_columns(x, &active_mask);
            let n_active = x_active.ncols();

            let x_extended = create_shadow_matrix(&x_active, seed_base + iter as u64);

            // 2. Entraîner la RF et récupérer les importances
            let importances = compute_importances(
                &x_extended,
                y,
                self.config.n_estimators,
                seed_base + iter as u64,
            );

            let (orig_imp, shadow_imp) = split_importances(&importances, n_active);

            // 3. MZSA : maximum des importances shadow à cette itération
            let max_shadow = shadow_imp
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            // 4. Enregistrer les importances et compter les hits
            //    (on remet les importances dans l'espace des features originales)
            let mut active_idx = 0;
            for orig_idx in 0..n_features {
                if statuses[orig_idx] == FeatureStatus::Rejected {
                    // Feature exclue : importance = 0 par convention
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

            // 5. Mettre à jour les décisions via test binomial
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
            feature_names: None, // à remplir par l'appelant si besoin
            n_iterations: n_iter,
            importance_history,
        }
    }
}

/// Filtre les colonnes de `x` selon un masque booléen.
fn filter_columns(x: &Array2<f64>, mask: &[bool]) -> Array2<f64> {
    let selected: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
        .collect();

    let cols: Vec<_> = selected
        .iter()
        .map(|&i| x.column(i))
        .collect();

    ndarray::stack(ndarray::Axis(1), &cols)
        .expect("Echec du stack de colonnes")
}
```

---

## 10. Étape 7 — API publique et résultat

**Fichier : `src/lib.rs`**

```rust
// src/lib.rs

pub mod boruta;
pub mod decision;
pub mod importance;
pub mod shadow;
pub mod stats;

pub use boruta::{Boruta, BorutaConfig, BorutaResult, FeatureStatus};

impl BorutaResult {
    /// Indices des features confirmées.
    pub fn confirmed_indices(&self) -> Vec<usize> {
        self.statuses
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if *s == FeatureStatus::Confirmed { Some(i) } else { None }
            })
            .collect()
    }

    /// Indices des features rejetées.
    pub fn rejected_indices(&self) -> Vec<usize> {
        self.statuses
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if *s == FeatureStatus::Rejected { Some(i) } else { None }
            })
            .collect()
    }

    /// Indices des features indéterminées (après max_iter atteint).
    pub fn tentative_indices(&self) -> Vec<usize> {
        self.statuses
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if *s == FeatureStatus::Tentative { Some(i) } else { None }
            })
            .collect()
    }

    /// Sérialise le résultat en JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .expect("Erreur de sérialisation JSON")
    }

    /// Affiche un résumé lisible.
    pub fn summary(&self) {
        println!("=== Résultat Boruta ({} itérations) ===", self.n_iterations);
        println!("  Confirmed  : {:?}", self.confirmed_indices());
        println!("  Rejected   : {:?}", self.rejected_indices());
        println!("  Tentative  : {:?}", self.tentative_indices());
        if let Some(names) = &self.feature_names {
            println!("\nFeatures confirmées :");
            for i in self.confirmed_indices() {
                println!("  - {}", names[i]);
            }
        }
    }
}
```

---

## 11. Exemple d'utilisation complet

```rust
// examples/iris.rs
use boruta_rs::{Boruta, BorutaConfig};
use ndarray::array;

fn main() {
    env_logger::init(); // RUST_LOG=info cargo run --example iris

    // Données synthétiques : 3 features utiles, 2 bruitées
    // En pratique : charger depuis CSV avec le crate `csv` + `ndarray-csv`
    let x = array![
        [1.2, 0.3, 3.1, 0.01, 9.9],
        [2.1, 0.8, 2.9, 0.02, 8.8],
        // ... (n lignes)
    ];
    let y = array![0u32, 1, 0, 1]; // labels de classification

    let config = BorutaConfig {
        max_iter: 100,
        p_value: 0.01,
        bonferroni: true,
        n_estimators: 100,
        random_seed: Some(42),
    };

    let boruta = Boruta::new(config);
    let mut result = boruta.fit(&x, &y);

    // Optionnel : nommer les features
    result.feature_names = Some(vec![
        "age".into(), "score_A".into(), "score_B".into(),
        "bruit_1".into(), "bruit_2".into(),
    ]);

    result.summary();

    // Export JSON
    std::fs::write("boruta_result.json", result.to_json())
        .expect("Impossible d'écrire le fichier JSON");

    // Filtrer X pour ne garder que les features confirmées
    let kept = result.confirmed_indices();
    println!("Features retenues : {:?}", kept);
}
```

### Charger des données CSV

```rust
// Avec les crates `csv` et `ndarray-csv`
use csv::ReaderBuilder;
use ndarray::Array2;
use ndarray_csv::Array2Reader;

fn load_csv(path: &str) -> (Array2<f64>, Vec<u32>) {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .expect("Fichier introuvable");

    // Supposons que la dernière colonne est le label
    let data: Array2<f64> = rdr.deserialize_array2_dynamic()
        .expect("Erreur de lecture CSV");

    let n_cols = data.ncols();
    let x = data.slice(ndarray::s![.., ..n_cols - 1]).to_owned();
    let y: Vec<u32> = data
        .column(n_cols - 1)
        .iter()
        .map(|&v| v as u32)
        .collect();

    (x, y)
}
```

---

## 12. Tests

```rust
// src/stats.rs — tests unitaires

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_z_score_basique() {
        let h = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = z_score(&h);
        // mean=3, std=sqrt(2.5)≈1.581 → z≈1.897
        assert_relative_eq!(z, 1.897, epsilon = 0.01);
    }

    #[test]
    fn test_z_score_constant() {
        let h = vec![2.0, 2.0, 2.0];
        assert_eq!(z_score(&h), 0.0);
    }

    #[test]
    fn test_p_value_upper_certain() {
        // Si une feature bat le shadow à chaque itération (hits = n_trials),
        // la p-value doit être très faible.
        let p = p_value_upper(100, 100);
        assert!(p < 1e-10);
    }

    #[test]
    fn test_p_value_lower_jamais() {
        // Si une feature ne bat jamais le shadow (hits = 0),
        // la p-value inférieure doit être très faible.
        let p = p_value_lower(0, 100);
        assert!(p < 1e-10);
    }

    #[test]
    fn test_bonferroni() {
        let threshold = bonferroni_threshold(0.05, 10);
        assert_relative_eq!(threshold, 0.005, epsilon = 1e-9);
    }
}
```

```rust
// tests/integration_test.rs

use boruta_rs::{Boruta, BorutaConfig, FeatureStatus};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Génère un dataset synthétique avec n_informative features utiles
/// et n_noise features purement aléatoires.
fn make_classification(
    n_obs: usize,
    n_informative: usize,
    n_noise: usize,
    seed: u64,
) -> (Array2<f64>, Array1<u32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_features = n_informative + n_noise;
    let mut data = Vec::with_capacity(n_obs * n_features);
    let mut labels = Vec::with_capacity(n_obs);

    for _ in 0..n_obs {
        let label: u32 = rng.gen_range(0..2);
        labels.push(label);
        for j in 0..n_features {
            let val = if j < n_informative {
                // Features utiles : corrélées au label
                label as f64 + rng.gen::<f64>() * 0.5
            } else {
                // Features bruit : purement aléatoires
                rng.gen::<f64>()
            };
            data.push(val);
        }
    }

    let x = Array2::from_shape_vec((n_obs, n_features), data).unwrap();
    let y = Array1::from_vec(labels);
    (x, y)
}

#[test]
fn test_boruta_detecte_features_informatives() {
    let (x, y) = make_classification(300, 5, 5, 42);

    let config = BorutaConfig {
        max_iter: 50,
        p_value: 0.01,
        bonferroni: true,
        n_estimators: 50,
        random_seed: Some(42),
    };

    let result = Boruta::new(config).fit(&x, &y);

    // Les 5 premières features (informatives) doivent être confirmées
    for i in 0..5 {
        assert_eq!(
            result.statuses[i],
            FeatureStatus::Confirmed,
            "La feature informative {} devrait être Confirmed",
            i
        );
    }

    // Les 5 dernières (bruit) doivent être rejetées
    for i in 5..10 {
        assert_ne!(
            result.statuses[i],
            FeatureStatus::Confirmed,
            "La feature bruit {} ne devrait pas être Confirmed",
            i
        );
    }
}
```

---

## 13. Feuille de route

| Priorité | Fonctionnalité |
|---|---|
| 🔴 Haute | Support régression (target `f64`) en plus de la classification |
| 🔴 Haute | Benchmarks sur datasets réels (UCI, Iris, Wine) |
| 🟡 Moyenne | `TentativeRoughFix` : test de seuil simple pour trancher les Tentative restantes |
| 🟡 Moyenne | Importance par permutation (PFI) en alternative au MDI |
| 🟡 Moyenne | Support `linfa` en plus de `smartcore` (feature flag Cargo) |
| 🟢 Basse | Export des courbes d'importance (historique) vers CSV / plotters |
| 🟢 Basse | Interface Python via `pyo3` pour interopérabilité |
| 🟢 Basse | Publication sur crates.io |

---

## Références

- Kursa, M. B., & Rudnicki, W. R. (2010). *Feature Selection with the Boruta Package*. Journal of Statistical Software, 36(11). https://doi.org/10.18637/jss.v036.i11
- [BorutaPy — Python port](https://github.com/scikit-learn-contrib/boruta_py)
- [smartcore — Rust ML](https://docs.rs/smartcore)
- [ndarray — N-dimensional arrays](https://docs.rs/ndarray)
- [statrs — Statistics](https://docs.rs/statrs)
