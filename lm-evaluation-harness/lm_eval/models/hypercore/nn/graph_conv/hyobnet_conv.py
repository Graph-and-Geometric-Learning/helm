import torch
import torch.nn as nn
import math
from torch.nn.modules.module import Module
from ...nn.linear import HyboNetLinear
class HybonetConv(nn.Module):
    """
    Fully hyperbolic Graph Convolution Layer (HyboNetConv).
    Args:
        manifold: Lorentz manifold instance.
        in_features (int): Input feature dimensionality.
        out_features (int): Output feature dimensionality.
        use_bias (bool): Whether to add a learnable bias in linear layer.
        dropout (float): Dropout probability for linear projection.
        use_att (bool): Whether to use attention during aggregation.
        local_agg (bool): Whether to aggregate in local tangent spaces.
        nonlin (callable, optional): Optional nonlinearity applied inside HyboNetLinear.

    Based on:
        - Fully Hyperbolic Neural Networks (https://arxiv.org/abs/2105.14686)
    """

    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att, local_agg, nonlin=None):
        super(HybonetConv, self).__init__()
        self.linear = HyboNetLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_features, dropout, use_att, local_agg)

    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        h = self.agg(h, adj)
        output = h, adj
        return output
    
class LorentzAgg(Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att, local_agg):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.c = manifold.c
        self.use_att = use_att
        if self.use_att:
            self.key_linear = HyboNetLinear(manifold, in_features, in_features)
            self.query_linear = HyboNetLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        if self.use_att:
            if self.local_agg:
                query = self.query_linear(x)
                key = self.key_linear(x)
                att_adj = 2 + 2 * self.manifold.cinner(query, key)
                att_adj = att_adj / self.scale + self.bias
                att_adj = torch.sigmoid(att_adj)
                att_adj = torch.mul(adj.to_dense(), att_adj)
                support_t = torch.matmul(att_adj, x)
            else:
                adj_att = self.att(x, adj)
                support_t = torch.matmul(adj_att, x)
        else:
            support_t = torch.spmm(adj, x)
        denom = (-self.manifold.l_inner(support_t, support_t, keep_dim=True))
        denom = denom.abs().clamp_min(1e-6).sqrt()
        denom = denom / self.c.sqrt()
        output = support_t / denom
        return output

    def attention(self, x, adj):
        pass
