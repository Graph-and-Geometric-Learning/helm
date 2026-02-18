import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from ...nn.graph_conv.att_layers import DenseAtt
from ...nn.linear.hnn_layers import HypAct, HypLinear

class HGCNConv(nn.Module):
    """
    Hyperbolic Graph Convolution Layer (HGCNConv).

    Args:
        manifold_in: Input manifold instance.
        manifold_out: Output manifold instance.
        in_features (int): Input feature dimensionality.
        out_features (int): Output feature dimensionality.
        dropout (float): Dropout rate.
        act (callable): Activation function.
        use_bias (bool): Whether to add bias in linear layer.
        use_att (bool): Whether to use attention during aggregation.
        local_agg (bool): Whether to use local tangent aggregation.

    Based on:
        - Hyperbolic Graph Convolutional Neural Networks (https://arxiv.org/abs/1910.12933)
    """

    def __init__(self, manifold_in, manifold_out, in_features, out_features, dropout, act, use_bias, use_att, local_agg):
        super(HGCNConv, self).__init__()
        self.linear = HypLinear(manifold_in, in_features, out_features, dropout, use_bias)
        self.agg = HypAgg(manifold_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold_in, manifold_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output
    
class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = manifold.c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t))
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t))
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)