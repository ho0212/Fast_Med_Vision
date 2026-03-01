use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use ndarray::parallel::prelude::*;

/// This is a testing function that demonstrates how to receive a NumPy array from Python, 
/// process it in Rust, and return a new NumPy array back to Python. 
#[pyfunction]
fn process_image_test<'py>(py: Python<'py>, input_array: PyReadonlyArrayDyn<f64>) -> &'py PyArrayDyn<f64> {


    // Convert the input NumPy array to a Rust array
    let rust_array  = input_array.as_array();

    // Simulating some processing on the Rust array (e.g., applying a simple transformation)
    let result = rust_array.mapv(|x| x * 2.0); //multiply each element by 2

    // Convert the processed Rust array back to a NumPy array and return it
    result.into_pyarray(py)
}

/// This function normalises the input array by scaling its values to a range of [0, 1].
/// It takes a NumPy array as input, processes it in Rust, and returns the normalized array back to Python.
#[pyfunction]
fn fast_normalise<'py>(py: Python<'py>, input_array: PyReadonlyArrayDyn<f64>) -> &'py PyArrayDyn<f64> {


    // Convert the input NumPy array to a Rust array
    let rust_array  = input_array.as_array();

    // Calculate the mean and std deviation of the array
    let mean = rust_array.mean().unwrap_or(0.0);
    let std = rust_array.std(0.0);
    let std = if std == 0.0 { 1.0 } else { std }; // Avoid division by zero

    // Normalise the array using the mean and std deviation with parallel processing
    let mut result_array = rust_array.to_owned();
    result_array.par_mapv_inplace(|x| (x - mean) / std);

    // Convert the normalised Rust array back to a NumPy array and return it
    result_array.into_pyarray(py)
}


#[pymodule]
fn fast_med_vision(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_image_test, m)?)?;
    m.add_function(wrap_pyfunction!(fast_normalise, m)?)?;
    Ok(())
}