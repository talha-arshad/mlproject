"""Basic convolutional model building blocks."""
from typing import Any, Dict, Union

import torch
from torch import nn
import torch.nn.functional as F

from mlproject.data import mnist 

INPUT_DIMS, OUTPUT_DIMS, MAPPING = mnist.INPUT_DIMS, mnist.OUTPUT_DIMS, mnist.MAPPING
NUM_CLASSES = len(MAPPING)

CONV_CHANNELS = (32, 64)
CONV_KERNEL_SIZES = (3, 3)
POOL_KERNEL_SIZES = (2, 2)
CONV_PADDING='valid'

class ConvBlock(nn.Module):
    """
    Convolutional block composed of Conv2d, ReLU and MaxPool2d.
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, stride: int=1, padding: Union[str, int]='valid', pooling_kernel_size: int=2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the ConvBlock to x.

        Parameters
        ----------
        x
            (B, C, H, W) tensor

        Returns
        -------
        torch.Tensor
            (B, C, H, W) tensor
        """
        c = self.conv(x)
        r = self.relu(c)
        p = self.pool(r)
        return p

class CNN(nn.Module):
    """ CNN with 2 conv/relu/pool blocks for MNIST 
        Based on: https://keras.io/examples/vision/mnist_convnet/
    """

    def __init__(self, input_dims=INPUT_DIMS, num_classes=NUM_CLASSES, conv_channels=CONV_CHANNELS, conv_kernel_sizes=CONV_KERNEL_SIZES, pool_kernel_sizes=POOL_KERNEL_SIZES, conv_padding=CONV_PADDING) -> None:
        super().__init__()
        self.example_input_array = torch.Tensor(mnist.BATCH_SIZE, *mnist.INPUT_DIMS)
        self.conv1 = ConvBlock(input_channels=input_dims[0], 
                               output_channels=conv_channels[0], 
                               kernel_size=conv_kernel_sizes[0], 
                               padding=conv_padding, 
                               pooling_kernel_size=pool_kernel_sizes[0])
        self.conv2 = ConvBlock(input_channels=conv_channels[0], 
                               output_channels=conv_channels[1], 
                               kernel_size=conv_kernel_sizes[1], 
                               padding=conv_padding, 
                               pooling_kernel_size=pool_kernel_sizes[1])
        self.avepool = nn.AvgPool2d(kernel_size=5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=conv_channels[1], out_features=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the CNN to x.

        Parameters
        ----------
        x
            (B, Ch, 28, 28) tensor.

        Returns
        -------
        torch.Tensor
            (B, Cl) tensor
        """
        assert tuple(x.shape[1:]) == INPUT_DIMS, f"bad inputs to CNN with shape: {x.shape}"
        x = self.conv1(x)  # B, CONV_CHANNELS[0], 13, 13
        x = self.conv2(x)  # B, CONV_CHANNELS[1], 5, 5
        x = self.avepool(x) # B, CONV_CHANNELS[1], 1, 1
        x = self.flatten(x)  # B, CONV_CHANNELS[1]
        x = self.linear(x)  # B, NUM_CLASSES
        return x

def main():
    m = CNN()
    print(m)

if __name__ == '__main__':
    main()
    
