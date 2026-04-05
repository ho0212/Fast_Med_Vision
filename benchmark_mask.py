import numpy as np
import fast_med_vision as fmv
import time

np.random.seed(42)  # For reproducibility
shape = (100, 512, 512)
num_patients = 5
print(f"Simulating 3D medical images for {num_patients} patients with shape {shape} and dtype float64...")

# Simulate a 3D medical image with dimensions (512, 512, 100) and dtype float64 (range 0 to 2000)
patient_data = [np.random.rand(*shape).astype(np.float64) * 2000 for _ in range(num_patients)]

# Simulate a binary mask with the same shape, where 1 indicates valid pixels and 0 indicates invalid pixels
mask_data = []
for _ in range(num_patients):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[20:80, 150:350, 150:350] = 1  # Create a valid region in the center
    mask_data.append(mask)

# --- Numpy Masked Normalisation Benchmark ---
print("\nBenchmarking NumPy masked normalisation...")
start_time = time.time()
np_masked = []
for img, mask in zip(patient_data, mask_data):
    # Extract valid pixels using the mask
    valid_pixels = img[mask == 1]
    mean = valid_pixels.mean() if valid_pixels.size > 0 else 0.0
    std = valid_pixels.std() if valid_pixels.size > 0 else 1.0

    result = np.zeros_like(img)
    result[mask == 1] = (valid_pixels - mean) / std  # Normalise only the valid pixels
    np_masked.append(result)
end_time = time.time()
np_time = end_time - start_time
print(f"NumPy masked normalisation completed in {np_time:.4f} seconds.")

# --- Rust Masked Normalisation Benchmark ---
print("\nBenchmarking Rust masked normalisation...")
start_time = time.time()
rust_masked = []
for img, mask in zip(patient_data, mask_data):
    rust_masked.append(fmv.masked_normalise(img, mask))
end_time = time.time()
rust_time = end_time - start_time
print(f"Rust masked normalisation completed in {rust_time:.4f} seconds.")

# Verify the results
print("=" * 30)
# This is a sanity check to ensure that background pixels are set to zero and valid pixels are normalised.
print("Background:")
print(f"Rust masked normalised background value: {rust_masked[0][0, 0, 0]:.6f} (expected 0.0)")
print(f"NumPy masked normalised background value: {np_masked[0][0, 0, 0]:.6f} (expected 0.0)")
print("Valid region:")
valid_np = np_masked[0][mask_data[0] == 1]
valid_rust = rust_masked[0][mask_data[0] == 1]
print(f"Rust valid region Max: {np.max(valid_rust):.6f} | Min: {np.min(valid_rust):.6f}")
print(f"NumPy valid region Max: {np.max(valid_np):.6f} | Min: {np.min(valid_np):.6f}")

# Check if the Rust and NumPy results are close enough (considering floating-point precision)
is_close = np.allclose(np_masked, rust_masked, atol=1e-6)
print(f"Do the NumPy and Rust masked normalised arrays match? {'Yes' if is_close else 'No'}")

if np_time > 0 and rust_time > 0:
    speedup = np_time / rust_time
    print(f"Rust masked normalisation is {speedup:.2f} times faster than NumPy.")
