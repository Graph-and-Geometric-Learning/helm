import torch
import torch.nn as nn
from ...nn.linear.lorentz_linear import LorentzLinear

class LorentzSparseSqDisAtt(nn.Module):
    def __init__(self, manifold, in_features, dropout):
        super(LorentzSparseSqDisAtt, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.manifold = manifold
        self.c = manifold.c
        self.weight_linear = LorentzLinear(manifold, in_features, in_features+1, dropout, True)

    def forward(self, x, adj):
        d = x.size(1) - 1
        x = self.weight_linear(x)
        index = adj._indices()
        _x = x[index[0, :]]
        _y = x[index[1, :]]
        _x_head = _x.narrow(1, 0, 1)
        _y_head = _y.narrow(1, 0, 1)
        _x_tail = _x.narrow(1, 1, d)
        _y_tail = _y.narrow(1, 1, d)
        l_inner = -_x_head.mul(_y_head).sum(-1) + _x_tail.mul(_y_tail).sum(-1)
        res = torch.clamp(-(self.c+l_inner), min=1e-10, max=1)
        res = torch.exp(-res)
        return (index, res, adj.size())