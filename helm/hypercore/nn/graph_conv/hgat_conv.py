import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops, softmax
from geoopt.manifolds import PoincareBall
from ...nn.linear.hnn_layers import HypLinear
from ...nn.linear.lorentz_linear import LorentzLinear

class HGATConv(Module):
    """
    Hyperbolic Graph Attention Convolution (HGATConv).

    Args:
        manifold: Hyperbolic manifold (PoincareBall or Lorentz).
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension.
        heads (int, optional): Number of attention heads. Default is 1.
        concat (bool, optional): Whether to concatenate heads or average. Default is True.
        negative_slope (float, optional): Negative slope for LeakyReLU. Default is 0.2.
        dropout (float, optional): Dropout probability on attention coefficients. Default is 0.
        use_bias (bool, optional): If True, add learnable bias. Default is True.
        act (callable, optional): Activation function applied on output.
        use_att (bool, optional): If True, use attention mechanism. Default is True.
        dist (bool, optional): If True, use distance-aware attention weighting. Default is True.

    Based on:
        - Hyperbolic Graph Attention Network (https://arxiv.org/abs/1912.03046)
    """
    def __init__(self,
                 manifold,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 use_bias=True,
                 act=None,
                 use_att=True,
                 dist=True):
        super(HGATConv, self).__init__()

        self.manifold = manifold
        self.c = manifold.c
        self.concat = concat
        if use_bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if concat:
            self.out_channels = out_channels // heads
        else:
            self.out_channels = out_channels

        self.in_channels = in_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.act = act
        self.dist = dist
        self.atten = use_att
        if self.manifold.name == 'Lorentz':
            self.hy_linear = nn.Linear(in_channels, heads * (self.out_channels - 1))
        else:
            self.hy_linear = HypLinear(manifold, in_channels, heads * self.out_channels, dropout, use_bias)
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)
        self.hy_linear.reset_parameters()
    
    def project(self, x):
        x_space = x
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.c) ** 0.5
        x = torch.cat([x_time, x_space], dim=-1)
        return x

    def forward(self, input, input_type='adj'):
        if input_type == 'adj':
            x, adj = input
            edge_index = adj._indices()
        elif input_type == 'edges':
            x, edge_index = input
        else:
            raise not NotImplementedError('The input type needs to be adjacent matrices or edge indexes.')
        print(edge_index)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.hy_linear.forward(x)
        if self.manifold.name == 'Lorentz':
            # if Lorentz then x is only the time-like dimension
            original_x = x.view(-1, self.heads, self.out_channels - 1)
            original_x = self.project(original_x)
            out_shape = (x.shape[0], x.shape[1] + 1)
            log_x = self.manifold.logmap0(original_x)  # Get log(x) as input to GCN
        else:
            log_x = self.manifold.logmap0(x)  # Get log(x) as input to GCN
            out_shape = x.shape
            log_x = log_x.view(-1, self.heads, self.out_channels)
            original_x = x.view(-1, self.heads, self.out_channels)
        print('pinting log_x')
        print(log_x[:20])
        assert(not log_x.isnan().any())
        assert(not log_x.isinf().any())
        edge_i  = edge_index[0]
        edge_j  = edge_index[1]
        x_i = log_x[edge_i]
        x_j = log_x[edge_j]
        original_x_i = original_x[edge_i]
        original_x_j = original_x[edge_j]
        if self.atten:
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = alpha.view(-1, self.heads)
            if self.dist:
                dist = self.manifold.sqdist(original_x_i, original_x_j)
                dist = softmax(dist, edge_i, num_nodes=x.size(0))
                alpha = alpha * dist
            
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, edge_i, num_nodes=x.size(0))

            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)
            
            support = alpha.view(-1, self.heads, 1) * x_j
        else:
            support = x_j
        out = scatter(src=support, index=edge_i, dim=0, dim_size=x.size(0), reduce='sum')

        out = self.manifold.proj_tan0(out)

        out = self.act(out)
        out = self.manifold.proj_tan0(out)
        print('is printing something on tangent space')
        print(out[:20])
        if input_type == 'adj':
            return self.manifold.projx(self.manifold.expmap0(out)).reshape(out_shape, -1), adj
        else:
            return self.manifold.projx(self.manifold.expmap0(out)).reshape(out_shape, -1), edge_index

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
    
def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

def add_self_loops(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    dtype, device = edge_index.dtype, edge_index.device
    loop = torch.arange(0, num_nodes, dtype=dtype, device=device)
    loop = loop.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop], dim=1)

    return edge_index

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)