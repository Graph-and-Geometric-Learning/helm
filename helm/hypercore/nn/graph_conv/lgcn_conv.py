import torch
import torch.nn as nn
import math
import torch.nn.init as init
from torch.nn.modules.module import Module
import torch.nn.functional as F
from ...nn.graph_conv.att_layers import SpecialSpmm
from ...nn.attention.sparse_dist import LorentzSparseSqDisAtt
from ...nn.linear import HypAct

class LGCNConv(nn.Module):
    """
    Tangent-sapce based Lorentz Graph Convolution Layer (LGCNConv).

    Args:
        manifold_in: Input Lorentz manifold instance.
        manifold_out: Output Lorentz manifold instance.
        in_feature (int): Input feature dimension.
        out_features (int): Output feature dimension.
        dropout (float): Dropout probability.
        act (callable): Activation function.
        use_bias (bool): Whether to use bias in the linear layer.
        use_att (bool): Whether to use attention in aggregation.

    Based on:
        - Lorentzian Graph Convolutional Networks (https://arxiv.org/abs/2104.07477)
    """
    def __init__(self, manifold_in, manifold_out, in_feature, out_features, dropout, act, use_bias, use_att):
        super(LGCNConv, self).__init__()
        self.linear = LGCNLinear(manifold_in, in_feature, out_features, dropout, use_bias)
        self.agg = LGCNAgg(manifold_in, use_att, out_features, dropout)
        self.lorentz_act = HypAct(manifold_in, manifold_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x) ## problem is h1+
        h = self.agg.forward(h, adj)
        h = self.lorentz_act.forward(h)
        output = h, adj
        return output

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.agg.reset_parameters()

class LGCNLinear(nn.Module):
    # Lorentz Hyperbolic Graph Neural Layer
    def __init__(self, manifold, in_features, out_features, drop_out, use_bias):
        super(LGCNLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = manifold.c
        self.drop_out = drop_out
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features-1))   # -1 when use mine mat-vec multiply
        self.weight = nn.Parameter(torch.Tensor(out_features - 1, in_features))  # -1, 0 when use mine mat-vec multiply
        self.reset_parameters()

    def report_weight(self):
        print(self.weight)

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)
        # print('reset lorentz linear layer')

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.drop_out, training=self.training)
        mv = self.manifold.matvec_regular(drop_weight, x, self.bias, self.use_bias)
        return mv

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class LGCNAgg(Module):
    """
    Lorentz centroids aggregation layer
    """
    def __init__(self, manifold, use_att, in_features, dropout):
        super(LGCNAgg, self).__init__()
        self.manifold = manifold
        self.c = manifold.c
        self.use_att = use_att
        self.in_features = in_features
        self.dropout = dropout
        self.this_spmm = SpecialSpmm()
        if use_att:
            self.att = LorentzSparseSqDisAtt(manifold, in_features-1, dropout)


    def lorentz_centroid(self, weight, x, c):
        """
        Lorentz centroid
        :param weight: dense weight matrix. shape: [num_nodes, num_nodes]
        :param x: feature matrix [num_nodes, featur       bnbn     fffafdfdfsfdsfsdfdsvcvcvsfdsfes]
        :return: the centroids of nodes [num_nodes, features]
        """
        if self.use_att:
            sum_x = self.this_spmm(weight[0], weight[1], weight[2], x)
        else:
            sum_x = torch.spmm(weight, x)
        x_inner = self.manifold.l_inner(sum_x, sum_x)
        coefficient = (c ** 0.5) / torch.sqrt(torch.abs(x_inner))
        return torch.mul(coefficient, sum_x.transpose(-2, -1)).transpose(-2, -1)

    def forward(self, x, adj):
        if self.use_att:
            adj = self.att(x, adj)
        output = self.lorentz_centroid(adj, x, self.c)
        return output

    def extra_repr(self):
        return 'c={}, use_att={}'.format(
                self.c, self.use_att
        )

    def reset_parameters(self):
        if self.use_att:
            self.att.reset_parameters()