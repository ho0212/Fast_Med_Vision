import numpy as np
import fast_med_vision as fmv
import time
from statistics import mean, stdev

np.random.seed(42)  # For reproducibility
shape = (100, 512, 512)
num_patients = 5
iterations = 10

# Create mask data for the masked normalisation benchmark
mask_data = np.zeros(shape, dtype=np.uint8)
mask_data[20:80, 150:350, 150:350] = 1  # Create a valid region in the center

def run_benchmarks(name, np_logic, rust_logic):
    np_times = []
    rust_times = []
    print(f"Testing {name} for {iterations} iterations...")
    print("-" * 50)

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}...")
        # Simulate a 3D medical image with dimensions (512, 512, 100) and dtype float64
        data = [np.random.rand(*shape).astype(np.float64) * 2000 for _ in range(num_patients)]

        # --- Numpy Benchmark ---
        start_time = time.time()
        np_logic(data)
        np_times.append(time.time() - start_time)

        # --- Rust Benchmark ---
        start_time = time.time()
        rust_logic(data)
        rust_times.append(time.time() - start_time)

        print(f"Round {i + 1} completed. NumPy time: {np_times[-1]:.4f} seconds | Rust time: {rust_times[-1]:.4f} seconds")
    
    # Calculate average times and speedup
    np_avg_time = mean(np_times)
    np_std_time = stdev(np_times)
    rust_avg_time = mean(rust_times)
    rust_std_time = stdev(rust_times)
    print("-" * 50)
    print(f"{name} Benchmark Results:")
    print(f"NumPy average time: {np_avg_time:.4f} seconds (std: {np_std_time:.4f})")
    print(f"Rust average time: {rust_avg_time:.4f} seconds (std: {rust_std_time:.4f})")
    
    speedup = np_avg_time / rust_avg_time
    print(f"Rust {name} is {speedup:.2f} times faster than NumPy.")
    print("-" * 50)

    return np_avg_time, np_std_time, rust_avg_time, rust_std_time, speedup

# Run benchmarks for each operation
def np_normalisation(data):
    for img in data:
        np_mean = np.mean(img)
        std = np.std(img)
        std = std if std > 0 else 1.0 # Avoid division by zero
        _ = (img - np_mean) / std

def rust_normalisation(data):
    for img in data:
        _ = fmv.fast_normalise(img)

def np_noise(data):
    for img in data:
        noise = np.random.normal(0.0, 1.0, img.shape)
        _ = img + noise

def rust_noise(data):
    for img in data:
        _ = fmv.add_gaussian_noise(img, 0.0, 1.0)

def np_clip(data):
    for img in data:
        lower_percentile = np.percentile(img, 1.0)
        upper_percentile = np.percentile(img, 99.0)
        _ = np.clip(img, lower_percentile, upper_percentile)

def rust_clip(data):
    for img in data:
        _ = fmv.percentile_clip(img, 1.0, 99.0)

def np_minmax(data):
    for img in data:
        min_value = np.min(img)
        max_value = np.max(img)
        _ = (img - min_value) / (max_value - min_value) if max_value > min_value else img

def rust_minmax(data):
    for img in data:
        _ = fmv.min_max_scale(img)

def np_masked(data):
    for img in data:
        valid_pixels = img[mask_data == 1]
        mean = valid_pixels.mean() if valid_pixels.size > 0 else 0.0
        std = valid_pixels.std() if valid_pixels.size > 0 else 1.0
        result = np.zeros_like(img)
        result[mask_data == 1] = (valid_pixels - mean) / std

def rust_masked(data):
    for img in data:
        _ = fmv.masked_normalise(img, mask_data)

benchmarks = [
    ("Normalisation", np_normalisation, rust_normalisation),
    ("Noise Adding", np_noise, rust_noise),
    ("Percentile Clipping", np_clip, rust_clip),
    ("Min-Max Normalisation", np_minmax, rust_minmax),
    ("Masked Normalisation", np_masked, rust_masked)
]

print("Starting benchmarks...")

final_results = {}
for name, np_logic, rust_logic in benchmarks:
    np_avg_time, np_std_time, rust_avg_time, rust_std_time, speedup = run_benchmarks(name, np_logic, rust_logic)
    final_results[name] = {
        "numpy_avg_time": np_avg_time,
        "numpy_std_time": np_std_time,
        "rust_avg_time": rust_avg_time,
        "rust_std_time": rust_std_time,
        "speedup": speedup
    }

print("\nAll benchmarks completed. Summary of results:")
print("="*65)
print(f"{'Function':<25} | {'NumPy (s)':<12} | {'Rust (s)':<12} | {'Speedup'}")
print("-" * 65)
for name, stat in final_results.items():
    print(f"{name:<25} | {stat['numpy_avg_time']:.4f} (±{stat['numpy_std_time']:.2f}) | {stat['rust_avg_time']:.4f} (±{stat['rust_std_time']:.2f}) | {stat['speedup']:.2f}x")
print("="*65)