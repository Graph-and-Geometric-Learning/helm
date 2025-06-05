import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ...nn.conv import LorentzConv2d

class LorentzPatchEmbedding(nn.Module):
    """
    Lorentz Patch Embedding using Lorentzian 2D Convolution.

    Args:
        manifold_in: Lorentz manifold instance for the input space.
        image_size (int): Height (and width) of the input image. Assumes square images.
        patch_size (int): Size of each square patch.
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels (patch embedding dimension).
        manifold_out: Optional Lorentz manifold for the output space.
    """
        
    def __init__(self, manifold_in, image_size, patch_size, in_channel, out_channel, manifold_out=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = LorentzConv2d(manifold_in=manifold_in, in_channels=in_channel, out_channels=out_channel, kernel_size=self.patch_size, stride=self.patch_size, manifold_out=manifold_out)

    def forward(self, x):
        # make x channel-last
        # x = x.permute(0, 2, 3, 1)
        x = self.projection(x) # batch_size * num_patches_x * num_patches_y * channels
        x = x.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
        return x