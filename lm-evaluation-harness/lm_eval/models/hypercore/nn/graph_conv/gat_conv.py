import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops, softmax

class GATConv(Module):
    """The graph attentional operator from the "Graph Attention Networks"
    Implementation based on torch_geometric
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 act=None):
        super(GATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.act = act
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if concat:
            self.out_channels = out_channels // heads
        else:
            self.out_channels = out_channels

        self.linear = torch.nn.Linear(in_channels, heads * self.out_channels, bias=bias)
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)
        self.linear.reset_parameters()

    def forward(self, input):
        """"""
        x, adj = input
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        edge_index = adj._indices()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.linear(x).view(-1, self.heads, self.out_channels)

        edge_i = edge_index[0]
        edge_j = edge_index[1]
        x_i = x[edge_i]
        x_j = x[edge_j]

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_i, num_nodes=x.size(0))

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        
        out = alpha.view(-1, self.heads, 1) * x_j
        out = scatter(src=out, index=edge_i, dim=0, dim_size=x.size(0), reduce='sum')

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias

        out = self.act(out)
        return out, adj

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