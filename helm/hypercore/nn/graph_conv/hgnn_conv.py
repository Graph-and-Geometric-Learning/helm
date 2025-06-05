import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall
from ...nn.graph_conv.gcn_conv import GCNConv

class HGNNConv(nn.Module):
    """
    Hyperbolic Graph Neural Network Convolution Layer (HGNNConv).

    Args:
        manifold (PoincareBall): Poincare Ball manifold instance.
        in_features (int): Input feature dimensionality.
        out_features (int): Output feature dimensionality.
        dropout (float): Dropout probability.
        act (callable): Activation function (applied inside GCN).
        use_bias (bool): Whether to add a learnable bias.

    Based on
        - Hypergraph Neural Networks (https://arxiv.org/abs/1809.09401)
    """

    def __init__(self, manifold, in_features, out_features, dropout, act, use_bias):
        super(HGNNConv, self).__init__()
        self.conv = GCNConv(in_features, out_features, False, False, dropout, use_bias, act)
        self.p = dropout
        self.manifold = manifold

    def forward(self, input):
        x, adj = input
        h = self.manifold.logmap0(x)
        h, _ = self.conv((h, adj))
        h = F.dropout(h, p=self.p, training=self.training)
        h = self.manifold.expmap0(h)
        h = F.relu(h)
        output = h, adj
        return output