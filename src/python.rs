pub mod engine_ext;

use pyo3::prelude::*;

#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine_ext::PyEngine>()?;
    Ok(())
}
