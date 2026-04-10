import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

file_path = "./data/brats2020-training-data"
real_data_dir = None

# Walk through the directory to find the .h5 file
for root, dirs, files in os.walk(file_path):
    if any(f.endswith('.h5') for f in files):
        real_data_dir = root
        break

print(f"Real data directory: {real_data_dir}")

# Load the .h5 file
all_h5_files = [
    f"volume_1_slice_{i}.h5" 
    for i in range(155) 
    if os.path.exists(os.path.join(real_data_dir, f"volume_1_slice_{i}.h5"))
]
print(f"Found {len(all_h5_files)} .h5 files.")

# Load the first patient's .h5 files and visualize the data

tumor_file_path = None
tumor_mask = None
tumor_image = None

# Loop through the .h5 files to find the tumor image and mask
for filename in all_h5_files:
    file_path = os.path.join(real_data_dir, filename)

    with h5py.File(file_path, 'r') as f:
        mask_data = np.array(f['mask'])

        if np.sum(mask_data) > 0:  # Check if there is a tumor in the mask
            tumor_file_path = file_path
            tumor_mask = mask_data
            tumor_image = np.array(f['image'])
            break

if tumor_file_path:
    print(f"Found tumor in file: {tumor_file_path}")
    print(f"Tumor mask shape: {tumor_mask.shape}")
    print(f"Tumor image shape: {tumor_image.shape}")

    # Visualise the tumor image and mask
    plt.figure(figsize=(20, 4))

    modalities = ['FLAIR', 'T1', 'T1CE', 'T2']
    for i in range(4):
        plt.subplot(1, 5, i + 1)
        plt.imshow(tumor_image[:, :, i], cmap='gray')
        plt.title(f"Channel: {modalities[i]}")
        plt.axis('off')
    
    plt.subplot(1, 5, 5)
    combined_mask = np.max(tumor_mask, axis=2)  # Combine all mask channels for visualisation
    plt.imshow(combined_mask, cmap='jet')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("No tumor found in this patient's data.")
