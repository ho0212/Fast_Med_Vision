import numpy as np
import fast_med_vision as fmv
import time

shape = (100, 512, 512)
num_patients = 5
print(f"Simulating 3D medical images for {num_patients} patients with shape {shape} and dtype float64...")
# Simulate a 3D medical image with dimensions (512, 512, 100) and dtype float64 (range 0 to 2000)
patient_data = [(np.random.rand(*shape).astype(np.float64) * 4000) - 1000 for _ in range(num_patients)]

# --- Numpy Min-Max Scaling Benchmark ---
print("\nBenchmarking NumPy min-max scaling...")
start_time = time.time()
np_scaled = []
for img in patient_data:
    min_value = np.min(img)
    max_value = np.max(img)
    result = (img - min_value) / (max_value - min_value)
    np_scaled.append(result)
end_time = time.time()
np_time = end_time - start_time
print(f"NumPy min-max scaling completed in {np_time:.4f} seconds.")

# --- Rust Min-Max Scaling Benchmark ---
print("\nBenchmarking Rust min-max scaling...")
start_time = time.time()
rust_scaled = []
for img in patient_data:
    rust_scaled.append(fmv.min_max_scale(img))
end_time = time.time()
rust_time = end_time - start_time
print(f"Rust min-max scaling completed in {rust_time:.4f} seconds.")

# Verify the results
print("=" * 30)
# This is a sanity check to ensure that the min-max scaling has been applied correctly.
rust_max = np.max(rust_scaled[0])
rust_min = np.min(rust_scaled[0])
np_max = np.max(np_scaled[0])
np_min = np.min(np_scaled[0])

print(f"Rust scaled max: {rust_max:.6f} (expected: 1.0) | Rust scaled min: {rust_min:.6f} (expected: 0.0)")
print(f"NumPy scaled max: {np_max:.6f} (expected: 1.0) | NumPy scaled min: {np_min:.6f} (expected: 0.0)")

# Check if the Rust and NumPy results are close enough (considering floating-point precision)
is_close = np.allclose(np_scaled, rust_scaled, atol=1e-6)
print(f"Do the NumPy and Rust min-max scaled arrays match? {'Yes' if is_close else 'No'}")

if np_time > 0 and rust_time > 0:
    speedup = np_time / rust_time
    print(f"Rust min-max scaling is {speedup:.2f} times faster than NumPy.")