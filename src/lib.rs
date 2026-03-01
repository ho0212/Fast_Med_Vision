use std::result;

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};

#[pyfunction]
fn process_image<'py>(py: Python<'py>, input_array: PyReadonlyArrayDyn<f64>) -> &'py PyArrayDyn<f64> {

    // Convert the input NumPy array to a Rust array
    let rust_array  = input_array.as_array();

    // Simulating some processing on the Rust array (e.g., applying a simple transformation)
    let result = rust_array.mapv(|x| x * 2.0); //multiply each element by 2

    // Convert the processed Rust array back to a NumPy array and return it
    result.into_pyarray(py)
}

#[pymodule]
fn fast_med_vision(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_image, m)?)?;
    Ok(())
}