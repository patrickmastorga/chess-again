// create a mock engine python object for testing purposes
use pyo3::prelude::*;
use crate::core;

#[pyclass(name = "Engine")]
pub struct PyEngine {
    board: core::Board,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new() -> Self {
        PyEngine {
            board: core::Board::new(),
        }
    }

    fn evaluate(&self) -> i32 {
        self.board.evaluate()
    }
}