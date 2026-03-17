/// PyO3 bindings for boruta-rs.
///
/// Build with maturin (see `python/pyproject.toml`):
///   cd python && maturin develop --features python
///
/// Then from Python:
///   from boruta_rs import Boruta
///   result = Boruta(max_iter=100, p_value=0.01, n_estimators=100).fit(X, y)
///   print(result.confirmed)      # list[int]
///   print(result.n_iterations)   # int
#[cfg(feature = "python")]
pub mod bindings {
    use crate::{Boruta, BorutaConfig};
    use numpy::{PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// Result of a Boruta run.
    #[pyclass(name = "BorutaResult")]
    pub struct PyBorutaResult {
        #[pyo3(get)]
        /// Indices of Confirmed features.
        pub confirmed: Vec<usize>,
        #[pyo3(get)]
        /// Indices of Rejected features.
        pub rejected: Vec<usize>,
        #[pyo3(get)]
        /// Indices of Tentative features (undecided after max_iter).
        pub tentative: Vec<usize>,
        #[pyo3(get)]
        /// Number of iterations performed.
        pub n_iterations: usize,
        /// Raw importance history [feature][iteration].
        inner: crate::BorutaResult,
    }

    #[pymethods]
    impl PyBorutaResult {
        /// Apply TentativeRoughFix: resolve remaining Tentative features via
        /// median threshold (mirrors R Boruta's TentativeRoughFix).
        pub fn rough_fix(&mut self) {
            self.inner.tentative_rough_fix();
            self.confirmed = self.inner.confirmed_indices();
            self.rejected = self.inner.rejected_indices();
            self.tentative = self.inner.tentative_indices();
        }

        /// Returns the importance history as a CSV string.
        ///
        /// Format: one row per iteration, columns = features.
        pub fn importance_history_csv(&self) -> String {
            self.inner.importance_history_to_csv()
        }

        /// Human-readable summary string.
        pub fn __repr__(&self) -> String {
            format!(
                "BorutaResult(confirmed={}, rejected={}, tentative={}, n_iterations={})",
                self.confirmed.len(),
                self.rejected.len(),
                self.tentative.len(),
                self.n_iterations,
            )
        }
    }

    /// Boruta feature selector.
    ///
    /// Parameters
    /// ----------
    /// max_iter : int, default 100
    /// p_value : float, default 0.01
    /// bonferroni : bool, default True
    /// n_estimators : int, default 100
    /// random_seed : int or None, default None
    #[pyclass(name = "Boruta")]
    pub struct PyBoruta {
        config: BorutaConfig,
    }

    #[pymethods]
    impl PyBoruta {
        #[new]
        #[pyo3(signature = (max_iter=100, p_value=0.01, bonferroni=true, n_estimators=100, random_seed=None))]
        pub fn new(
            max_iter: usize,
            p_value: f64,
            bonferroni: bool,
            n_estimators: usize,
            random_seed: Option<u64>,
        ) -> Self {
            Self {
                config: BorutaConfig {
                    max_iter,
                    p_value,
                    bonferroni,
                    n_estimators,
                    random_seed,
                },
            }
        }

        /// Run Boruta for classification.
        ///
        /// Parameters
        /// ----------
        /// x : np.ndarray, shape (n_obs, n_features), dtype float64
        /// y : np.ndarray, shape (n_obs,), dtype uint32 (integer class labels)
        ///
        /// Returns
        /// -------
        /// BorutaResult
        pub fn fit(
            &self,
            py: Python<'_>,
            x: PyReadonlyArray2<f64>,
            y: PyReadonlyArray1<u32>,
        ) -> PyResult<PyBorutaResult> {
            let x_nd = x.as_array().to_owned();
            let y_nd = y.as_array().to_owned();

            let boruta = Boruta::new(BorutaConfig { ..self.config.clone() });
            let result = py.allow_threads(|| boruta.fit(&x_nd, &y_nd));

            Ok(PyBorutaResult {
                confirmed: result.confirmed_indices(),
                rejected: result.rejected_indices(),
                tentative: result.tentative_indices(),
                n_iterations: result.n_iterations,
                inner: result,
            })
        }

        /// Run Boruta for regression (continuous target).
        ///
        /// Parameters
        /// ----------
        /// x : np.ndarray, shape (n_obs, n_features), dtype float64
        /// y : np.ndarray, shape (n_obs,), dtype float64
        ///
        /// Returns
        /// -------
        /// BorutaResult
        pub fn fit_regression(
            &self,
            py: Python<'_>,
            x: PyReadonlyArray2<f64>,
            y: PyReadonlyArray1<f64>,
        ) -> PyResult<PyBorutaResult> {
            let x_nd = x.as_array().to_owned();
            let y_nd = y.as_array().to_owned();

            let boruta = Boruta::new(BorutaConfig { ..self.config.clone() });
            let result = py.allow_threads(|| boruta.fit_regression(&x_nd, &y_nd));

            Ok(PyBorutaResult {
                confirmed: result.confirmed_indices(),
                rejected: result.rejected_indices(),
                tentative: result.tentative_indices(),
                n_iterations: result.n_iterations,
                inner: result,
            })
        }

        pub fn __repr__(&self) -> String {
            format!(
                "Boruta(max_iter={}, p_value={}, bonferroni={}, n_estimators={})",
                self.config.max_iter,
                self.config.p_value,
                self.config.bonferroni,
                self.config.n_estimators,
            )
        }
    }

    /// Registers the module's classes. Called from the `#[pymodule]` entry point in `lib.rs`.
    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyBoruta>()?;
        m.add_class::<PyBorutaResult>()?;
        Ok(())
    }
}
