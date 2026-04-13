import os
import random
import time
import h5py
import numpy as np
import fast_med_vision as fmv
from statistics import mean, stdev

# === Load BraTS data ===
data_dir = "./data/brats2020-training-data"
iterations = 10

print(f"Loading real BraTS slices into memory...")
all_files = [os.path.join(root, f) for root, dirs, files in os.walk(data_dir) for f in files if f.endswith('.h5')]

test_files = [
    os.path.join(root, f) 
    for root, dirs, files in os.walk(data_dir) 
    for f in files if f.endswith('.h5') and 'volume_1_' in f
]
test_files.sort()
num_samples = len(test_files)

real_images = []
real_masks = []

for f_path in test_files:
    with h5py.File(f_path, 'r') as f:
        real_images.append(np.array(f['image']).astype(np.float64))
        real_masks.append(np.array(f['mask']).astype(np.uint64)) 

print(f"Loading completed. \n")
print("=" * 50)


# === Run benchmark ===
def run_benchmarks(name, np_logic, rust_logic, images, masks):
    np_times = []
    rust_times = []
    print(f"Testing {name}...")

    for i in range(iterations):
        # --- NumPy Benchmark ---
        start_time = time.time()
        np_logic(images, masks)
        np_times.append(time.time() - start_time)

        # --- Rust Benchmark ---
        start_time = time.time()
        rust_logic(images, masks)
        rust_times.append(time.time() - start_time)

    # Calculate statistical data
    np_avg_time = mean(np_times)
    np_std_time = stdev(np_times) if iterations > 1 else 0
    rust_avg_time = mean(rust_times)
    rust_std_time = stdev(rust_times) if iterations > 1 else 0
    speedup = np_avg_time / rust_avg_time if rust_avg_time > 0 else 0

    return np_avg_time, np_std_time, rust_avg_time, rust_std_time, speedup

# === Preprocessing Functions ===
def np_normalisation(images, masks):
    for img in images:
        np_mean = np.mean(img)
        std = np.std(img)
        std = std if std > 0 else 1.0 
        _ = (img - np_mean) / std

def rust_normalisation(images, masks):
    for img in images:
        _ = fmv.fast_normalise(img)

def np_noise(images, masks):
    for img in images:
        noise = np.random.normal(0.0, 1.0, img.shape)
        _ = img + noise

def rust_noise(images, masks):
    for img in images:
        _ = fmv.add_gaussian_noise(img, 0.0, 1.0)

def np_clip(images, masks):
    for img in images:
        lower_percentile = np.percentile(img, 1.0)
        upper_percentile = np.percentile(img, 99.0)
        _ = np.clip(img, lower_percentile, upper_percentile)

def rust_clip(images, masks):
    for img in images:
        _ = fmv.percentile_clip(img, 1.0, 99.0)

def np_minmax(images, masks):
    for img in images:
        min_value = np.min(img)
        max_value = np.max(img)
        _ = (img - min_value) / (max_value - min_value) if max_value > min_value else img

def rust_minmax(images, masks):
    for img in images:
        _ = fmv.min_max_scale(img)

def np_masked(images, masks):
    for img, mask in zip(images, masks):
        # Producting whole tumor (element will be 1 as long as it is 1 in any channel)
        spatial_mask = np.any(mask > 0, axis=-1)

        # Extract valid pixels
        valid_pixels = img[spatial_mask]
        
        # Calculate mean and std
        mean = valid_pixels.mean() if valid_pixels.size > 0 else 0.0
        std = valid_pixels.std() if valid_pixels.size > 0 else 1.0
        std = std if std > 0 else 1.0
        
        result = np.zeros_like(img)
        result[spatial_mask] = (img[spatial_mask] - mean) / std

def rust_masked(images, masks):
    for img, mask in zip(images, masks):
        # # Producting whole tumor (element will be 1 as long as it is 1 in any channel)
        spatial_mask_2d = np.any(mask > 0, axis=-1).astype(np.uint8)
        

        # Manually broadcast to (240, 240, 4)
        spatial_mask_3d = np.repeat(spatial_mask_2d[:, :, np.newaxis], img.shape[-1], axis=2)
        _ = fmv.masked_normalise(img, spatial_mask_3d)

# === Execute and export results ===
benchmarks = [
    ("Normalisation", np_normalisation, rust_normalisation),
    ("Noise Adding", np_noise, rust_noise),
    ("Percentile Clipping", np_clip, rust_clip),
    ("Min-Max Normalisation", np_minmax, rust_minmax),
    ("Masked Normalisation", np_masked, rust_masked)
]

final_results = {}
for name, np_logic, rust_logic in benchmarks:
    np_avg, np_std, rust_avg, rust_std, speedup = run_benchmarks(name, np_logic, rust_logic, real_images, real_masks)
    final_results[name] = (np_avg, np_std, rust_avg, rust_std, speedup)

print("\n" + "="*80)
print(f"{'Function (Real Data)':<25} | {'NumPy (s)':<17} | {'Rust (s)':<17} | {'Speedup'}")
print("-" * 80)
for name, (np_avg, np_std, rust_avg, rust_std, speedup) in final_results.items():
    print(f"{name:<25} | {np_avg:.4f} (±{np_std:.4f}) | {rust_avg:.4f} (±{rust_std:.4f}) | {speedup:>6.2f}x")
print("="*80)