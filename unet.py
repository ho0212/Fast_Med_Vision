import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class DoubleConv(nn.Module):
    """
    A module consisting of two convolutional layers, each followed by batch normalization and ReLU activation.
    This is a common building block in UNet architectures.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """
    A module that performs downsampling using max pooling followed by a DoubleConv.
    This is used in the contracting path of the UNet.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    A module that performs upsampling using either bilinear interpolation or transposed convolution, followed by a DoubleConv.
    This is used in the expansive path of the UNet.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    """
    A module that performs a final convolution to reduce the number of channels to the desired output channels.
    This is used at the end of the UNet to produce the final segmentation map.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    The UNet architecture for image segmentation. It consists of a contracting path (encoder) and an expansive path (decoder).
    The encoder captures context while the decoder enables precise localization.
    """
    def __init__(self, n_channels=4, n_classes=3):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Input channels are 4 (FLAIR, T1, T1CE, T2) and output channels are 3 (background, tumor core, whole tumor)
        self.inc = DoubleConv(n_channels, 64)

        # Downsampling path
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # The bottleneck layer
        self.down4 = Down(512, 1024)

        # Upsampling path
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Final output convolution to get the desired number of classes
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

if __name__ == "__main__":

    # Detect if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Call the UNet model and test it with a dummy input
    model = UNet(n_channels=4, n_classes=3).to(device)
    # Print the model summary using torchinfo
    print("=" * 50)
    summary(model, input_size=(8, 4, 240, 240),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
            device=device)
    print("=" * 50)

    # Batch size of 8, 4 channels, 240x240 image
    dummy_input = torch.randn(8, 4, 240, 240).to(device)

    print("Input shape:", dummy_input.shape)
    output = model(dummy_input)
    print("Output shape:", output.shape)

    assert output.shape == (8, 3, 240, 240), "Output shape is incorrect. Expected (8, 3, 240, 240)."
    print("UNet model test passed successfully!")