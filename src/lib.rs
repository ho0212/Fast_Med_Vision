use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use ndarray::parallel::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

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

/// This function adds Gaussian noise to the input array. 
/// It takes a NumPy array, the mean and standard deviation of the noise as input, processes it in Rust, and returns the noisy array back to Python.
#[pyfunction]
fn add_gaussian_noise<'py>(py: Python<'py>, input_array: PyReadonlyArrayDyn<f64>, noise_mean: f64, noise_std: f64) -> &'py PyArrayDyn<f64> {
    // Convert the input NumPy array to a Rust array
    let rust_array = input_array.as_array();

    // Create a normal distribution with the specified mean and standard deviation
    let normal_dist = Normal::new(noise_mean, noise_std).unwrap();

    let mut result_array = rust_array.to_owned();

    result_array.par_mapv_inplace(|x| {
        let mut rng = thread_rng();
        x + normal_dist.sample(&mut rng) // Add Gaussian noise to each element
    });

    // Convert the noisy Rust array back to a NumPy array and return it
    result_array.into_pyarray(py)
}

/// This function clips the values of the input array to the specified lower and upper percentiles.
/// It takes a NumPy array, the lower and upper percentiles as input, processes it in Rust, 
/// and returns the clipped array back to Python.
#[pyfunction]
fn percentile_clip<'py>(py: Python<'py>, input_array: PyReadonlyArrayDyn<f64>, lower_percentile: f64, upper_percentile: f64) -> &'py PyArrayDyn<f64> {
    // Convert the input NumPy array to a Rust array
    let rust_array = input_array.as_array();

    // Sort the array to find the percentiles
    let mut flat_vec: Vec<f64> = rust_array.iter().cloned().collect();
    flat_vec.par_sort_unstable_by(|a, b| a.total_cmp(b)); // Sort the vector in parallel

    // Calculate the indices for the lower and upper percentiles
    let lower_index = ((lower_percentile / 100.0) * (flat_vec.len() as f64)).floor() as usize;
    let upper_index = ((upper_percentile / 100.0) * (flat_vec.len() as f64)).floor() as usize;

    // Making sure the indices are within bounds
    let lower_index = lower_index.clamp(0, flat_vec.len().saturating_sub(1));
    let upper_index = upper_index.clamp(0, flat_vec.len().saturating_sub(1));

    // Get the lower and upper percentile values
    let lower_value = flat_vec[lower_index];
    let upper_value = flat_vec[upper_index];

    // Clip the original array values to the calculated percentiles
    let mut result_array = rust_array.to_owned();
    result_array.par_mapv_inplace(|x| {
        if x < lower_value {
            lower_value
        } else if x > upper_value {
            upper_value
        } else {
            x
        }
    });

    result_array.into_pyarray(py)
}

#[pymodule]
fn fast_med_vision(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_image_test, m)?)?;
    m.add_function(wrap_pyfunction!(fast_normalise, m)?)?;
    m.add_function(wrap_pyfunction!(add_gaussian_noise, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_clip, m)?)?;
    Ok(())
}