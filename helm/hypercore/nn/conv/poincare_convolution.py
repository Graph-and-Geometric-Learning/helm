from typing import Tuple

import torch
import torch.nn as nn
from scipy.special import beta

from ...manifolds import PoincareBall
from ...nn.conv import PoincareMLR


class PoincareConvolution2d(nn.Module):
    """
    Poincare 2D Convolution Layer.

    Based on:
        - Poincare ResNet (https://arxiv.org/abs/2303.14027)

    Args:
        manifold (PoincareBall): Instance of the Poincare Ball manifold.
        c (float): Curvature of the Poincare ball.
        in_channels (int): Number of input channels (including time dimension if needed).
        out_channels (int): Number of output channels.
        kernel_dims (Tuple[int, int]): Height and width of convolutional kernels.
        bias (bool, optional): If True, includes learnable bias. Default is True.
        stride (int, optional): Stride of the sliding window. Default is 1.
        padding (int, optional): Zero-padding added to both sides. Default is 0.
        id_init (bool, optional): If True, uses identity initialization for weights.
    """

    def __init__(
        self,
        manifold: PoincareBall,
        c,
        in_channels: int,
        out_channels: int,
        kernel_dims: Tuple[int, int],
        bias: bool = True,
        stride: int = 1,
        padding: int = 0,
        id_init: bool = True,
    ) -> None:
        # Store layer parameters
        super(PoincareConvolution2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_dims = kernel_dims
        self.kernel_size = kernel_dims[0] * kernel_dims[1]
        self.manifold = manifold
        self.c = c
        self.stride = stride
        self.padding = padding
        self.id_init = id_init

        # Unfolding layer
        self.unfold = nn.Unfold(
            kernel_size=kernel_dims,
            padding=padding,
            stride=stride,
        )

        # Create weights
        self.has_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.weights = nn.Parameter(
            torch.empty(self.kernel_size * in_channels, out_channels)
        )

        # Initialize weights
        self.reset_parameters()

        # Create beta's for concatenating receptive field features
        self.beta_ni = beta(self.in_channels / 2, 1 / 2)
        self.beta_n = beta(self.in_channels * self.kernel_size / 2, 1 / 2)
        self.mlr = PoincareMLR(self.manifold, self.c)

    def reset_parameters(self):
        # Identity initialization (1/2 factor to counter 2 inside the distance formula)
        if self.id_init:
            self.weights = nn.Parameter(
                1
                / 2
                * torch.eye(self.kernel_size * self.in_channels, self.out_channels)
            )
        else:
            nn.init.normal_(
                self.weights,
                mean=0,
                std=(2 * self.in_channels * self.kernel_size * self.out_channels)
                ** -0.5,
            )
        if self.has_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PoincareConvolution2d layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, out_height, out_width].
        """
        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        out_height = (
            height - self.kernel_dims[0] + 1 + 2 * self.padding
        ) // self.stride
        out_width = (width - self.kernel_dims[1] + 1 + 2 * self.padding) // self.stride

        # Scalar transform for concatenation
        x = x * self.beta_n / self.beta_ni

        # Apply sliding window to input to obtain features of each frame
        x = self.unfold(x)
        x = x.transpose(1, 2)

        # Project the receptive field features back onto the Poincare ball
        x = self.manifold.expmap0(x, self.c, dim=-1)

        # Apply the Poincare fully connected operation
        c_sqrt = self.c.sqrt()
        x = self.mlr(x, self.weights, self.bias)
        x = (c_sqrt * x).sinh() / c_sqrt
        x = x / (1 + (1 + self.c * x.pow(2).sum(dim=-1, keepdim=True)).sqrt())

        # Convert y back to the proper shape
        x = x.transpose(1, 2).reshape(
            batch_size, self.out_channels, out_height, out_width
        )

        return x
        # return self.manifold.logmap0(x, self.c, dim=1)