import torch
import torch.nn as nn
from geoopt import ManifoldParameter
import math
from ...manifolds import Lorentz

class LorentzCLS(nn.Module):
    """
    Lorentzian Classification Layer (CLS).

    This module learns a set of class embeddings in the Lorentz model
    and computes either negative distances, probabilities, or negative log-probabilities
    for classification.

    Same as lorentz decoder in graph_conv if negative distance

    Args:
        manifold (Lorentz): Lorentz manifold instance.
        in_channels (int): Input feature dimension (excluding Lorentzian time coordinate).
        out_channels (int): Number of output classes.
        bias (bool, optional): Whether to include learnable bias. Default is True.

    Based on:
        - Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space (https://arxiv.org/abs/2407.01290)
        - Fully Hyperbolic Neural Networks (https://arxiv.org/abs/2105.14686)
    """

    def __init__(self, manifold: Lorentz, in_channels, out_channels, bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c = manifold.c
        cls_emb = self.manifold.random_normal((self.out_channels, self.in_channels + 1), mean=0, std=1. / math.sqrt(self.in_channels + 1))
        self.cls = ManifoldParameter(cls_emb, self.manifold, requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def forward(self, x, x_manifold='hyp', return_type='neg_dist'):
        if x_manifold != 'hyp':
            x = self.manifold.expmap0(torch.cat([torch.zeros_like(x)[..., 0:1], x], dim=-1))  # project to Lorentz

        dist = -2 * self.c - 2 * self.cinner(x, self.cls) + self.bias
        dist = dist.clamp(min=0)

        if return_type == 'neg_dist':
            return - dist
        elif return_type == 'prob':
            return 10 / (1.0 + dist)
        elif return_type == 'neg_log_prob':
            return - 10*torch.log(1.0 + dist)
        else:
            raise NotImplementedError