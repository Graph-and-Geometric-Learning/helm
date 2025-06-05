import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
import math

class HNNLayer(nn.Module):
    """
    Tanget-space-based hyperbolic Neural Network Layer.
    Args:
        manifold: Manifold instance.
        in_features (int): Input dimensionality.
        out_features (int): Output dimensionality.
        c (float): Curvature of the manifold.
        dropout (float): Dropout rate.
        act (callable): Activation function.
        use_bias (bool): Whether to add a bias term.

    Based on:
        - HNN (https://arxiv.org/abs/1805.09112)
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h
    
class HypLinear(nn.Module):
    """
    Tanget-space-based hyperbolic Linear Layer.

    Applies a Möbius matrix-vector multiplication followed by optional bias addition
    in hyperbolic space.

    Args:
        manifold: Manifold instance.
        in_features (int): Input dimensionality.
        out_features (int): Output dimensionality.
        dropout (float): Dropout rate.
        use_bias (bool): Whether to add a bias term.

    Based on:
        - HNN (https://arxiv.org/abs/1805.09112)
    """

    def __init__(self, manifold, in_features, out_features, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = manifold.c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x)
        res = self.manifold.projx(mv)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1))
            hyp_bias = self.manifold.expmap0(bias)
            hyp_bias = self.manifold.projx(hyp_bias)
            res = self.manifold.mobius_add(res, hyp_bias,)
            res = self.manifold.projx(res)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class HypAct(Module):
    """
    Tanget-space-based hyperbolic Activation Layer.

    Applies a standard activation function in the tangent space at the origin,
    then reprojects back to the hyperbolic manifold.

    Args:
        manifold_in: Input manifold.
        manifold_out: Output manifold.
        act (callable): Activation function (e.g., ReLU, Tanh).

    Based on:
        - HNN (https://arxiv.org/abs/1805.09112)
    """

    def __init__(self, manifold_in, manifold_out, act):
        super(HypAct, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold_in.logmap0(x))
        xt = self.manifold_out.proj_tan0(xt)
        return self.manifold_out.projx(self.manifold_out.expmap0(xt))

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.manifold_in.c, self.manifold_out.c
        )
    
class HypResidual(Module):
    """
    Tangent-space-based hyperbolic Activation Layer

    Args:
        manifold: Input manifold
    """

    def __init__(self, manifold):
        super(HypAct, self).__init__()
        self.manifold = manifold

    def forward(self, x, y):
        return self.manifold.mobius_add(x, y)