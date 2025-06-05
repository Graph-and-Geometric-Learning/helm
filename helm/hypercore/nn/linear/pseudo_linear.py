import torch
import torch.nn as nn
import torch.nn.init as init
import math
import torch.nn.functional as F

class PseudoHypLinear(nn.Module):
    """
    QGCN
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(PseudoHypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0.0001)

    def forward(self, x):
        assert not torch.isnan(x).any()
        # time_dim = self.manifold.time_dim
        if self.manifold.time_dim<self.manifold.dim:
            time_dim = self.manifold.time_dim
        else:
            time_dim = int((self.manifold.time_dim/self.manifold.dim)*x.shape[1])
        
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.mobius_matvec(drop_weight, x, self.c, time_dim=time_dim)
        res = self.manifold.proj(res, self.c)
        assert not torch.isnan(res).any()
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            # assert not torch.isnan(hyp_bias).any()
            res = self.manifold.mobius_add(res, hyp_bias, self.c)
            res = self.manifold.proj(res, self.c)
        # assert self.manifold._check_point_on_manifold(res,self.c)
        assert not torch.isnan(res).any()
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )