pub mod engine_ext;
pub mod data_ext;

use pyo3::prelude::*;

#[pymodule]
fn chess_again(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::python_bindings::engine_ext::PyEngine>()?;
    m.add_class::<crate::python_bindings::data_ext::PyFenLoader>()?;
    Ok(())
}