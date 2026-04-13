import os
import random
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
from brats_dataset import FastMedicalNormalisation, ToPyTorchTensor

def main():
    # === Load the trained model ===
    data_dir = "./data/brats2020-training-data"
    model_path = "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialise the model and load the weights
    model = UNet(n_channels=4, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")

    # === Dynamically select a test file with large tumor ===
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".h5"):
                all_files.append(os.path.join(root, f))
    print("Searching for test files with large tumors...")

    random.shuffle(all_files)  # Shuffle to get a random sample
    search_subset = all_files[:200]  # Check the first 200 files for large tumors

    best_file = None
    max_tumor_pixel = 0

    for file_path in search_subset:
        with h5py.File(file_path, "r") as f:
            mask_np = np.array(f["mask"]).astype(np.float64)
            tumor_pixel = mask_np.sum()

            if tumor_pixel > max_tumor_pixel:
                max_tumor_pixel = tumor_pixel
                best_file = file_path
    
    if best_file is None or max_tumor_pixel == 0:
        print("No large tumors found in the search subset. Please re-run the program.")
        return

    print(f"Selected file: {best_file} with {max_tumor_pixel} tumor pixels.")

    with h5py.File(best_file, "r") as f:
        image_np = np.array(f["image"]).astype(np.float64)
        mask_np = np.array(f["mask"]).astype(np.float64)
    
    # === Preprocess the image ===
    norm = FastMedicalNormalisation()
    to_tensor = ToPyTorchTensor()

    img_norm, _ = norm(image_np, mask_np)
    img_tensor, mask_tensor = to_tensor(img_norm, mask_np)

    # add batch dimension and move to device
    img_batch = img_tensor.unsqueeze(0).to(device)  # Shape: (1, 4, 240, 240)

    # === Make prediction ===
    with torch.no_grad():
        output = model(img_batch)  # Shape: (1, 3, 240, 240)
        output = torch.sigmoid(output)  # Apply sigmoid to get probabilities
        output_mask = (output > 0.5).float()  # Threshold to get binary mask
    
    # === Visualise the results ===
    img_show = img_tensor[0].cpu().numpy()
    true_mask_show = mask_tensor[0].cpu().numpy()
    pred_mask_show = output_mask[0, 0].cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original MRI (FLAIR)")
    plt.imshow(img_show, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(true_mask_show, cmap="inferno")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("U-Net Predicted Mask")
    plt.imshow(pred_mask_show, cmap="inferno")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()