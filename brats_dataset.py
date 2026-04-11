import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import fast_med_vision as fmv

class FastMedicalNormalisation:
    """
    Normalises each channel of the image independently using fast_med_vision's normalisation."""
    def __call__(self, image, mask):

        for c in range(image.shape[2]):
            channel_data = image[:, :, c]
            if np.max(channel_data) > 0:
                fmv.fast_normalise(channel_data)
            image[:, :, c] = channel_data
        
        return image, mask

class ToPyTorchTensor:
    """
    Converts the image and mask to PyTorch tensors and changes the shape to (C, H, W).
    """
    def __call__(self, image, mask):

        image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W)
        mask = np.transpose(mask, (2, 0, 1))    # Change to (C, H, W)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

class ComposeTransforms:
    """
    Composes multiple transforms together. (Pipelines the transforms)
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask

class BraTSDataset(Dataset):
    """
    PyTorch Dataset for loading BraTS 2020 data from .h5 files. 
    Each .h5 file contains a single slice of a patient's brain scan, with 4 channels (FLAIR, T1, T1CE, T2) and a corresponding mask. 
    The dataset applies specified transforms to the images and masks before returning them as PyTorch tensors.
    """
    def __init__(self, data_dir, transforms=None):
        self.files = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".h5"):
                    self.files.append(os.path.join(root, f))

        self.transforms = transforms
        
        print(f"Found {len(self.files)} .h5 files in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as f:
            image = np.array(f['image']).astype(np.float64)
            mask = np.array(f['mask']).astype(np.float64)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        
        return image, mask

if __name__ == "__main__":
    data_dir = "./data/brats2020-training-data"
    train_transforms = ComposeTransforms([
        FastMedicalNormalisation(),
        ToPyTorchTensor()
    ])
    test_transforms = ComposeTransforms([
        FastMedicalNormalisation(),
        ToPyTorchTensor()
    ])

    train_set = BraTSDataset(data_dir, transforms=train_transforms)
    test_set = BraTSDataset(data_dir, transforms=test_transforms)

    dataloader = DataLoader(train_set, batch_size=8, shuffle=True)

    images, masks = next(iter(dataloader))
    print(f"Batch of images shape: {images.shape}")  # Should be (B, C, H, W)
    print(f"Batch of masks shape: {masks.shape}")    # Should be (B, C, H, W)
    print("Min and max pixel values in images:", images.min().item(), images.max().item())  # Check if normalisation worked
    print(f"Unique values in masks: {torch.unique(masks)}")  # Check unique values in masks to ensure they are correct (e.g., 0 and 1)
