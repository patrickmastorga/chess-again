// create a mock fenloader python object for testing purposes
use pyo3::prelude::*;
use crate::data;

#[pyclass(name = "FenLoader")]
pub struct PyFenLoader;

#[pymethods]
impl PyFenLoader {
    #[new]
    fn new() -> Self {
        PyFenLoader
    }

    fn get_fens(&self) -> Vec<String> {
        data::get_fens()
    }
}