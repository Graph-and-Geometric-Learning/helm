import torch
import torch.nn as nn
from ...nn.graph_conv.hgat_conv import HGATConv
from ...nn.graph_conv.gat_conv import GATConv
from geoopt.manifolds import PoincareBall
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module

class GILConv(nn.Module):
    """
    Geometric Interactive Learning Convolution (GILConv).

    Applies both hyperbolic and Euclidean GAT-style convolutions on input features,
    followed by a fusion of representations from both spaces.

    Args:
        manifold: Lorentzian or Poincare manifold object.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        act: Activation function.
        heads (int, optional): Number of attention heads. Default is 1.
        concat (bool, optional): Whether to concatenate or average heads. Default is True.
        negative_slope (float, optional): Negative slope for LeakyReLU. Default is 0.2.
        dropout (float, optional): Dropout rate. Default is 0.
        use_bias (bool, optional): Whether to use bias in linear layers. Default is True.
        use_att (bool, optional): Whether to use attention mechanism. Default is True.
        dist (bool, optional): Whether to use distance-based attention. Default is True.
        e_fusion_dropout (float, optional): Dropout rate for Euclidean fusion. Default is 0.0.
        h_fusion_dropout (float, optional): Dropout rate for hyperbolic fusion. Default is 0.0.

    Based on:
        - Graph Geometric Interactive Learning (https://arxiv.org/abs/2010.12135)
    """
    def __init__(self, manifold, in_features, out_features, act, heads=1, concat=True, 
                    negative_slope=0.2, dropout=0, use_bias=True, use_att=True, dist=True, e_fusion_dropout=0.0, h_fusion_dropout=0.0):
        super(GILConv, self).__init__()
        self.conv = HGATConv(manifold, in_features, out_features, heads, concat, negative_slope, 
                             dropout, use_bias, act, use_att, dist)
        self.conv_e = GATConv(in_features, out_features, heads, concat, negative_slope,
                              dropout, use_bias, act)
        '''feature fusion'''
        self.h_fusion = HFusion(manifold, e_fusion_dropout)
        self.e_fusion = EFusion(manifold, h_fusion_dropout)

    def forward(self, input):
        x, x_e = input[0]
        adj = input[1]
        "hyper forward"
        input_h = x, adj
        x, adj = self.conv(input_h)

        "eucl forward"
        input_e = x_e, adj
        x_e, _ = self.conv_e(input_e)

        "feature fusion"
        x = self.h_fusion(x, x_e)
        x_e = self.e_fusion(x, x_e)

        return (x, x_e), adj
    
class HFusion(Module):
    """Hyperbolic Feature Fusion from Euclidean space"""

    def __init__(self, manifold, drop):
        super(HFusion, self).__init__()
        self.manifold = manifold
        self.att = Parameter(torch.Tensor(1, 1))
        self.drop = drop
        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.att)

    def forward(self, x_h, x_e):
        dist = self.manifold.dist(x_h, self.manifold.expmap0(x_e)) * self.att
        x_e = self.manifold.mobius_scalar_mul(dist.view([-1, 1]), self.manifold.expmap0(x_e))
        x_e = F.dropout(x_e, p=self.drop, training=self.training)
        x_h = self.manifold.mobius_add(x_h, x_e)
        return x_h

class EFusion(Module):
    """Euclidean Feature Fusion from hyperbolic space"""

    def __init__(self, manifold, drop):
        super(EFusion, self).__init__()
        self.manifold = manifold
        self.att = Parameter(torch.Tensor(1, 1))
        self.drop = drop
        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.att)

    def forward(self, x_h, x_e):
        dist = (self.manifold.logmap0(x_h) - x_e).pow(2).sum(dim=-1) * self.att
        x_h = dist.view([-1, 1]) * self.manifold.logmap0(x_h)
        x_h = F.dropout(x_h, p=self.drop, training=self.training)
        x_e = x_e + x_h
        return x_e
    
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)