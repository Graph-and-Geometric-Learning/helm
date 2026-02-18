from .. import nn as hnn
import math
import torch
from torch import nn
import torch.nn.functional as F

from ..manifolds import Lorentz

class LorentzFeedForward(nn.Module):
    """

    Lorentz Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1: Linear layer for input-to-hidden transformation.
        w2: Linear layer for hidden-to-output transformation.
        w3: Additional linear layer for feature transformation.
    """
    def __init__(self, manifold: Lorentz, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            manifold: Input manifold
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.manifold = manifold
        self.c = manifold.c
        self.w1 = hnn.LorentzLinear(self.manifold, dim, inter_dim - 1)
        self.w2 = hnn.LorentzLinear(self.manifold, inter_dim, dim - 1)
        self.w3 = hnn.LorentzLinear(self.manifold, dim, inter_dim - 1)      

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        x1_time = F.silu(self.w1(x, return_space=True))
        x3_time = self.w3(x, return_space=True)
        x_space = x1_time * x3_time
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        return self.w2(x)