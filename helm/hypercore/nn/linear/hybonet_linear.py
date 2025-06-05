import torch
import torch.nn as nn
import math

class HyboNetLinear(nn.Module):
    """
    HyboNet Linear Layer.

    Projects input features through a linear transformation, applies optional nonlinearity,
    dropout, and rescales outputs to satisfy Lorentzian hyperbolic geometry constraints.

    Args:
        manifold: Lorentzian manifold instance.
        in_features (int): Dimensionality of input features.
        out_features (int): Dimensionality of output features.
        bias (bool, optional): If True, adds bias to linear transformation. Default is True.
        dropout (float, optional): Dropout probability before linear projection. Default is 0.1.
        scale (float, optional): Initial scale factor for the time coordinate. Default is 10.
        fixscale (bool, optional): If True, scale is fixed during training. Default is False.
        nonlin (callable, optional): Optional activation function applied before linear layer.

    Based on:
        - Fully Hyperbolic Neural Networks (https://arxiv.org/abs/2105.14686)
    """
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super(HyboNetLinear, self).__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.c = manifold.c
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + (self.c.sqrt() + 0.5)
        scale = (time * time - self.c) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        x = torch.cat([time, x_narrow * scale.clamp_min(1e-6).sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)