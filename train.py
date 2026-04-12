import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from brats_dataset import BraTSDataset, ComposeTransforms, FastMedicalNormalisation, ToPyTorchTensor
from unet import UNet
from collections import defaultdict

class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation. It measures the overlap between the predicted segmentation and the ground truth.
    The loss is calculated as 1 - Dice Coefficient, where the Dice Coefficient is defined as:
    Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
    where X is the predicted binary mask and Y is the ground truth binary mask.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        # Calculate Dice Coefficient
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1 - dice_coeff

def main():
    data_dir = "./data/brats2020-training-data"
    batch_size = 8
    num_epochs = 5
    patience = 2
    learning_rate = 1e-4

    # Detect if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Train Test Split ===
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".h5"):
                all_files.append(os.path.join(root, f))
    
    # Group files by patient ID
    patient_to_files = defaultdict(list)

    for file_path in all_files:
        file_name = os.path.basename(file_path) # volume_x_slice_x.h5
        patient_id = file_name.split("_slice_")[0] # volume_x
        patient_to_files[patient_id].append(file_path)
    
    unique_patients = list(patient_to_files.keys())
    print(f"Found {len(unique_patients)} unique patients in the dataset.")

    # Split patients into train and test sets
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)
    print(f"Training on {len(train_patients)} patients, testing on {len(test_patients)} patients.")

    # Create datasets and dataloaders
    train_files = []
    for pid in train_patients:
        train_files.extend(patient_to_files[pid])

    test_files = []
    for pid in test_patients:
        test_files.extend(patient_to_files[pid])
    
    print(f"Training set: {len(train_files)} files, Test set: {len(test_files)} files.")

    # Define transforms
    train_transforms = ComposeTransforms([
        FastMedicalNormalisation(),
        ToPyTorchTensor()
    ])
    test_transforms = ComposeTransforms([
        FastMedicalNormalisation(),
        ToPyTorchTensor()
    ])

    # Create datasets
    train_set = BraTSDataset(train_files, transforms=train_transforms)
    test_set = BraTSDataset(test_files, transforms=test_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # === Model Initialization ===
    model = UNet(n_channels=4, n_classes=3).to(device)
    bce_loss = nn.BCEWithLogitsLoss() # Define the loss function (Binary Cross-Entropy with Logits)
    dice_loss = DiceLoss() # Define the Dice loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Define the optimizer (Adam)

    # === Training Loop ===
    best_val_loss = float('inf') # Initialize the best validation loss to infinity
    save_path = "best_model.pth" # Path to save the best model
    patience_counter = 0 # Counter for early stopping
    test_iterator = iter(test_loader) # Create an iterator for the test dataloader

    for epoch in range(num_epochs):

        print(f"Epoch {epoch+1}/{num_epochs}")

        # === Training Phase ===
        model.train() # Set the model to training mode
        running_loss = 0.0 # Initialize a variable to accumulate the training loss

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device) # Move the input images to the device (GPU or CPU)
            masks = masks.to(device) # Move the target masks to the device

            optimizer.zero_grad() # Clear the gradients from the previous step
            outputs = model(images) # Forward pass: compute the model's predictions
            bce = bce_loss(outputs, masks)
            dice = dice_loss(outputs, masks)
            loss = bce + dice # Compute the loss between predictions and targets
            loss.backward() # Backward pass: compute gradients of the loss with respect to model parameters
            optimizer.step() # Update model parameters based on computed gradients

            running_loss += loss.item() # Accumulate the training loss

            model.eval() # Set the model to evaluation mode for validation
            with torch.no_grad(): # Disable gradient computation for validation
                try:
                    val_images, val_masks = next(test_iterator) # Get the next batch of validation data
                except StopIteration:
                    test_iterator = iter(test_loader) # Reset the iterator if we have exhausted the validation data
                    val_images, val_masks = next(test_iterator) # Get the next batch of validation data

                val_images = val_images.to(device) # Move validation images to the device
                val_masks = val_masks.to(device) # Move validation masks to the device

                val_outputs = model(val_images) # Forward pass on validation data
                val_bce = bce_loss(val_outputs, val_masks)
                val_dice = dice_loss(val_outputs, val_masks)
                val_loss = val_bce + val_dice # Compute validation loss

            if (batch_idx + 1) % 10 == 0: # Print the average loss every 10 batches
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Training Loss: {running_loss / (batch_idx + 1):.4f}, Validation Loss: {val_loss.item():.4f}")

            model.train() # Set the model back to training mode for the next batch

        
        # print the average loss for the epoch after processing all batches
        epoch_loss = running_loss / len(train_loader) # Calculate the average loss for the epoch
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss:.4f}")

        # === Validation Phase ===
        model.eval() # Set the model to evaluation mode
        val_loss = 0.0 # Initialize a variable to accumulate the validation loss

        with torch.no_grad(): # Disable gradient computation for validation
            for images, masks in test_loader:
                images = images.to(device) # Move the input images to the device
                masks = masks.to(device) # Move the target masks to the device

                outputs = model(images) # Forward pass: compute the model's predictions

                bce = bce_loss(outputs, masks)
                dice = dice_loss(outputs, masks)
                loss = bce + dice # Compute the loss between predictions and targets
                val_loss += loss.item() # Accumulate the validation loss

        val_loss /= len(test_loader) # Calculate the average validation loss
        print(f"Epoch {epoch+1} completed. Validation Loss: {val_loss:.4f}")

        # === Early Stopping and Checkpointing ===
        print(f"Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()