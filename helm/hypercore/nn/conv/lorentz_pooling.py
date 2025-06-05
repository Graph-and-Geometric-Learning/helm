"""
Lorentz global average pooling layer

Based on:
    - Fully Hyperbolic Convolutional Neural Networks for Computer Vision (https://arxiv.org/abs/2303.15919)
"""

import torch
import torch.nn as nn

from ...manifolds import Lorentz

class LorentzGlobalAvgPool2d(torch.nn.Module):
    """ Implementation of a Lorentz Global Average Pooling based on Lorentz centroid defintion. 
    """
    def __init__(self, manifold_in: Lorentz, keep_dim=False, manifold_out=None):
        super(LorentzGlobalAvgPool2d, self).__init__()

        self.manifold = manifold_in
        self.keep_dim = keep_dim
        self.c = manifold_in.c
        self.manifold_out = manifold_out

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = self.manifold.lorentzian_centroid(x)
        if self.keep_dim:
            x = x.view(bs, 1, 1, c)

        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()

        return x