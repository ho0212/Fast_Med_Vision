import numpy as np
import fast_med_vision as fmv
import time

shape = (100, 512, 512)
num_patients = 5

print(f"Simulating 3D medical images for {num_patients} patients with shape {shape} and dtype float64...")
# Simulate a 3D medical image with dimensions (512, 512, 100) and dtype float64 (range 0 to 100)
patient_data = [np.random.rand(*shape).astype(np.float64) * 100 for _ in range(num_patients)]

# Manually set some extreme values to test the clipping
for data in patient_data:
    data[0, 0, 0] = -99999  # Extreme low value
    data[0, 0, 1] = 99999  # Extreme high value
    data[0, 0, 2] = np.inf  # Infinite value
    data[0, 0, 3] = -np.inf  # Negative infinite value

# Percentiles to clip at
lower_percentile = 1.0
upper_percentile = 99.0

# --- Numpy Percentile Clipping Benchmark ---
print("\nBenchmarking NumPy percentile clipping...")
start_time = time.time()
np_clipped = []
for data in patient_data:
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    clipped_data = np.clip(data, lower_bound, upper_bound)
    np_clipped.append(clipped_data)
end_time = time.time()
np_time = end_time - start_time
print(f"NumPy percentile clipping completed in {np_time:.4f} seconds.")

# --- Rust Percentile Clipping Benchmark ---
print("\nBenchmarking Rust percentile clipping...")
start_time = time.time()
rust_clipped = []
for data in patient_data:
    rust_clipped.append(fmv.percentile_clip(data, lower_percentile, upper_percentile))
end_time = time.time()
rust_time = end_time - start_time
print(f"Rust percentile clipping completed in {rust_time:.4f} seconds.")

# Verify the results
print("=" * 30)
# This is a sanity check to ensure that the clipping has been applied correctly.
print("Verifying Clipped arrays...") 
rust_max = np.max(rust_clipped[0])
rust_min = np.min(rust_clipped[0])
np_max = np.max(np_clipped[0])
np_min = np.min(np_clipped[0])
print(f"Rust clipped max: {rust_max:.6f} | Rust clipped min: {rust_min:.6f}")
print(f"NumPy clipped max: {np_max:.6f} | NumPy clipped min: {np_min:.6f}")

# Check if the Rust and NumPy results are close enough (considering floating-point precision)
is_close = np.allclose(np_clipped, rust_clipped, atol=1.0)
print(f"Do the NumPy and Rust clipped arrays match? {'Yes' if is_close else 'No'}")

if np_time > 0 and rust_time > 0:
    speedup = np_time / rust_time
    print(f"Speedup of Rust percentile clipping over NumPy: {speedup:.2f}x")

