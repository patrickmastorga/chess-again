use crate::core;

use pyo3::prelude::*;

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