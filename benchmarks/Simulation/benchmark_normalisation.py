import numpy as np
import time
import fast_med_vision as fmv

np.random.seed(42)  # For reproducibility
shape = (100, 512, 512)
num_patients = 5 # Simulate multiple patients' MRI scans (Batch size of 5)

print(f"Simulating 3D medical images for {num_patients} patients with shape {shape} and dtype float64...")
# Simulate a 3D medical image with dimensions (512, 512, 100) and dtype float64
patient_data = [np.random.rand(*shape).astype(np.float64) for _ in range(num_patients)]


### --- NumPy Normalisation Benchmark ---
print("\nBenchmarking NumPy normalisation...")
start_time = time.time()

# Normalise the image using NumPy
np_normalised = []
for data in patient_data:
    np_mean = np.mean(data)
    std = np.std(data)
    std = std if std > 0 else 1.0 # Avoid division by zero
    np_normalised.append((data - np_mean) / std)

np_time = time.time() - start_time
print(f"NumPy normalisation completed in {np_time:.4f} seconds.")

### --- Rust Normalisation Benchmark ---
print("\nBenchmarking Rust normalisation...")
start_time = time.time()

# Normalise the image using the Rust function
rust_normalised = []
for data in patient_data:
    rust_normalised.append(fmv.fast_normalise(data))
rust_time = time.time() - start_time
print(f"Rust normalisation completed in {rust_time:.4f} seconds.")

# Verify the results
is_close = np.allclose(np_normalised, rust_normalised)
print("=" * 30)
print(f"Do the NumPy and Rust normalised arrays match? {'Yes' if is_close else 'No'}")
if not is_close:
    print("Warning: The arrays do not match exactly. This may be due to floating-point precision differences.")
if np_time > 0 and rust_time > 0:
    speedup = np_time / rust_time
    print(f"Rust normalisation is {speedup:.2f} times faster than NumPy.")