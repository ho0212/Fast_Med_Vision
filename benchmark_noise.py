import numpy as np
import fast_med_vision as fmv
import time

shape = (100, 512, 512)
num_patients = 5 # Simulate multiple patients' MRI scans (Batch size of 5)

print(f"Simulating 3D medical images for {num_patients} patients with shape {shape} and dtype float64...")
# Simulate a 3D medical image with dimensions (512, 512, 100) and dtype float64
patient_data = [np.random.rand(*shape).astype(np.float64) for _ in range(num_patients)]

# Define noise parameters
noise_mean = 0.0
noise_std = 0.1


### --- NumPy Noise Benchmark ---
print("\nBenchmarking NumPy Noise Adding...")
start_time = time.time()

# Add Gaussian noise to the image using NumPy
np_noisy = []
for data in patient_data:
    noise = np.random.normal(noise_mean, noise_std, data.shape)
    python_result = data + noise
    np_noisy.append(python_result)

np_time = time.time() - start_time
print(f"NumPy noise adding completed in {np_time:.4f} seconds.")

### --- Rust Noise Adding Benchmark ---
print("\nBenchmarking Rust noise adding...")
start_time = time.time()

# Add Gaussian noise to the image using the Rust function
rust_noisy = []
for data in patient_data:
    rust_noisy.append(fmv.add_gaussian_noise(data, noise_mean, noise_std))
rust_time = time.time() - start_time
print(f"Rust noise adding completed in {rust_time:.4f} seconds.")

# Verify the results
print("=" * 30)
print("Verifying Python and Rust noisy arrays...")

# Extract the noise added by both methods for the first patient
extracted_np_noisy = np_noisy[0] - patient_data[0]
extracted_rust_noisy = rust_noisy[0] - patient_data[0]

print("Python's results:")
print(f"Expected mean: {noise_mean:.6f} | Actual NumPy mean: {np.mean(extracted_np_noisy):.6f}")
print(f"Expected std: {noise_std:.6f} | Actual NumPy std: {np.std(extracted_np_noisy):.6f}")

print("Rust's results:")
print(f"Expected mean: {noise_mean:.6f} | Actual Rust mean: {np.mean(extracted_rust_noisy):.6f}")
print(f"Expected std: {noise_std:.6f} | Actual Rust std: {np.std(extracted_rust_noisy):.6f}")

if np_time > 0 and rust_time > 0:
    speedup = np_time / rust_time
    print(f"Speedup of Rust noise adding over NumPy: {speedup:.2f}x")