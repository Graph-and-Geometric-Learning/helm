import torch
import torch.nn as nn
import math
import torch.nn.init as init
from torch.nn.modules.module import Module
import torch.nn.functional as F
from ...nn.linear.pseudo_linear import PseudoHypLinear
from ...nn.graph_conv.att_layers import DenseAtt
class QGCNConv(nn.Module):
    """
    QGCN, TODO Adapt to current version of libary
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(QGCNConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = PseudoHypLinear(manifold, in_features, out_features, self.c_in, dropout, use_bias)
        self.agg = PseudoHypAgg(manifold, self.c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = PseudoHypAct(manifold, self.c_in, self.c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output
    
class PseudoHypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(PseudoHypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        assert not torch.isnan(x).any()
        x_tangent = self.manifold.logmap0(x, self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, self.c), self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.clamp(torch.matmul(adj_att, x_tangent), max=self.manifold.max_norm)
        else:
            support_t = torch.clamp(torch.spmm(adj, x_tangent), max=self.manifold.max_norm)

        assert not torch.isnan(x_tangent).any()
        assert not torch.isnan(support_t).any()
        res = self.manifold.proj_tan0(support_t,self.c)
        res = self.manifold.expmap0(res, self.c)
        output = self.manifold.proj(res,self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)