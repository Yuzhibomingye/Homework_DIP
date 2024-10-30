import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=1, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=1, padding=1),  # Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1),  # Output channels: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        
        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=1, padding=1),  # Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=1, padding=1),  # Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=1, padding=1),  # Output channels: 3 (RGB)
            nn.Tanh() # Activation function to output RGB values in the range [0, 1]
        )
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function

    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Decoder forward pass
        x = self.deconv1(x)
        x = self.deconv2(x)
        output = self.deconv3(x)
        ### FILL: encoder-decoder forward pass
        
        return output
    