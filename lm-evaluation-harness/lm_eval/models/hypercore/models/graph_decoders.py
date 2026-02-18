"""Graph decoders."""
from .. import manifolds as manifolds
import torch.nn as nn
import torch.nn.functional as F

from ..nn import GraphAttentionLayer
from ..nn import GraphConvolution, Linear, GATConv, HGATConv

from geoopt import ManifoldParameter
import torch
import math
import geoopt
from geoopt.manifolds import PoincareBall

class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c
        self.decoder_name = None

    def decode(self, x, adj):
        if self.decoder_name is not None:
            input = (x, adj)
            probs = self.forward(input)
        else:
            if self.decode_adj:
                input = (x, adj)
                probs, _ = self.cls.forward(input)
            else:
                probs = self.cls.forward(x)
        return probs

    def forward(self, probs):
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, manifold, input_dim, output_dim, bias=1, dropout=0.0):
        super(LinearDecoder, self).__init__(manifold.c)
        self.manifold = manifold
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.cls = Linear(self.input_dim, self.output_dim, dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x))
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )

class LorentzDecoder(Decoder):
    """
    Lrentzian Decoder for Hyperbolic node classification models.
    """

    def __init__(self, manifold, input_dim, output_dim, use_bias):
        super(LorentzDecoder, self).__init__(manifold.c)
        self.manifold = manifold
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.cls = ManifoldParameter(self.manifold.random_normal((self.output_dim,self.input_dim), std=1./math.sqrt(self.input_dim)), manifold=self.manifold)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        self.decode_adj = False
        self.c = manifold.c
    def decode(self, x, adj):
        return (2 * self.c + 2 * self.manifold.cinner(x, self.cls)) + self.bias
        
class H2HDecoder(Decoder):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c, args):
        super(H2HDecoder, self).__init__(c)
        self.input_dim = args.num_centroid
        self.output_dim = args.n_classes
        act = lambda x: x
        self.cls = Linear(self.input_dim, self.output_dim, 0.0, act, args.bias)
        args.eucl_vars.append(self.cls.linear)
        self.decode_adj = False

    def decode(self, x, adj):
        h = x
        return super(H2HDecoder, self).decode(h, adj)
    
class DualDecoder(Decoder):
    def __init__(self, manifold, in_features, out_features, act, heads=1, concat=True, 
                    negative_slope=0.2, dropout=0, use_bias=True, use_att=True, dist=True, e_fusion_dropout=0.0, h_fusion_dropout=0.0, device='cpu'):
        super(DualDecoder, self).__init__(manifold.c)
        self.manifold = manifold
        self.in_features = in_features
        self.cls_e = GATConv(self.in_features, out_features, heads, concat, negative_slope, dropout, use_bias,
                                lambda x: x)
        self.cls_h = HGATConv(self.manifold, self.in_features, in_features, heads, concat, negative_slope,
                                dropout, use_bias, act, use_att=use_att, dist=dist)

        self.output_dim = out_features
        self.c = manifold.c
        self.sphere = sphere = geoopt.manifolds.Sphere()
        self.scale = nn.Parameter(torch.zeros(self.output_dim))
        point = torch.randn(self.output_dim, self.in_features) / 4
        point = manifold.expmap0(point.to(device), project=False)
        tangent = torch.randn(self.output_dim, self.in_features)
        self.point = geoopt.ManifoldParameter(point, manifold=manifold)
        with torch.no_grad():
            self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()
        self.decoder_name = 'DualDecoder'

        '''prob weight'''
        self.w_e = nn.Linear(out_features, 1, bias=False)
        self.w_h = nn.Linear(in_features, 1, bias=False)
        self.drop_e = e_fusion_dropout
        self.drop_h = h_fusion_dropout
        self.reset_param()

    def reset_param(self):
        self.w_e.reset_parameters()
        self.w_h.reset_parameters()

    def forward(self, input):
        x, x_e = input[0]
        adj = input[1]
        '''Euclidean probs'''
        probs_e, _ = self.cls_e((x_e, adj))

        '''Hyper probs'''
        x, adj = self.cls_h((x, adj))
        x = x.unsqueeze(-2)
        distance = self.manifold.dist2plane(
            x=x, p=self.point, a=self.tangent, signed=True
        )
        probs_h = distance * self.scale.exp()

        '''Prob. Assembling'''
        w_h = torch.sigmoid(self.w_h(self.manifold.logmap0(x.squeeze())))
        w_h = F.dropout(w_h, p=self.drop_h, training=self.training)
        w_e = torch.sigmoid(self.w_e(probs_e))
        w_e = F.dropout(w_e, p=self.drop_e, training=self.training)

        w = torch.cat([w_h.view(-1, 1), w_e.view(-1, 1)], dim=-1)
        w = F.normalize(w, p=1, dim=-1)
        probs = w[-1, 0] * probs_h + w[-1, 1] * probs_e

        return super(DualDecoder, self).forward(probs)
    
class MDDecoder(Decoder):
    """
    Graph Reconstruction Decoder for Hyperbolic/Euclidean node classification models.
    """
    def __init__(self, manifold, args):
        super(Decoder, self).__init__()
        self.manifold=manifold
        self.input_dim = args.dim
        self.decode_adj = False
        self.c = self.manifold.c
        if args.model == 'GIL':
            self.w_e = nn.Linear(args.dim, 1, bias=False)
            self.w_h = nn.Linear(args.dim, 1, bias=False)
            self.drop_e = args.drop_e
            self.drop_h = args.drop_h
            self.data = args.dataset
            self.model = args.model
            self.reset_param()
    def reset_param(self):
        self.w_e.reset_parameters()
        self.w_h.reset_parameters()

    def decode(self, x, adj):
        if isinstance(x, tuple):
            #GIl loss
            num, dim = x[0].size(0),x[0].size(1)
            device = x[0].device
            adj = torch.Tensor(adj['adj_train'].A).to(device)
            positive = adj.bool()
            negative = ~positive

            x_h_1 = x[0].repeat(num,1)
            x_h_2 = x[0].repeat_interleave(num,0)
            dist_h = self.manifold.sqdist(x_h_1, x_h_2).view(num,num)
            inner_h = self.manifold.inner(x_h_1, x_h_2, keepdim=True).view(num,num)

            simi_h = torch.clamp(torch.exp(-dist_h),min=1e-15)
            positive_sim_h = simi_h * (positive.long())
            negative_sim_h = simi_h * (negative.long())
            
            negative_sum_h = negative_sim_h.sum(dim=1).unsqueeze(1).repeat(1,num)
            loss_h = torch.clamp(torch.div(positive_sim_h, negative_sum_h)[positive],min=1e-15)

            x_e_1 = x[1].repeat(num,1)
            x_e_2 = x[1].repeat_interleave(num,0)
            dist_e = torch.sqrt((x_e_1 - x_e_1).pow(2).sum(dim=-1) + 1e-15).view(num,num)
            inner_e = (x_e_1 * x_e_2).sum(dim=-1, keepdim=True).view(num,num)

            simi_e = torch.clamp(torch.exp(-dist_e),min=1e-15)
            positive_sim_e = simi_e * (positive.long())
            negative_sim_e = simi_e * (negative.long())

            negative_sum_e = negative_sim_e.sum(dim=1).unsqueeze(1).repeat(1,num)
            loss_e = torch.clamp(torch.div(positive_sim_e, negative_sum_e)[positive],min=1e-15)
            
            w_h = torch.sigmoid(self.w_h(x_h_1 - x_h_2).view(-1))
            w_e = torch.sigmoid(self.w_e(x_e_2 - x_e_2).view(-1))
            w = torch.cat([w_h.view(-1, 1), w_e.view(-1, 1)], dim=-1)

            loss = w[-1, 0] * loss_h + w[-1, 1] * loss_e
            loss = (-torch.log(loss)).sum() 
            dist = w[-1, 0] * dist_h + w[-1, 1] * dist_e
            inner = w[-1, 0] * inner_h + w[-1, 1] * inner_e
            edge_inner = inner*adj.bool()
            max_inner, min_inner = edge_inner.max().item(),edge_inner.min().item()
            return x, dist, loss, dist.max().item(), max_inner, min_inner

        num, dim = x.size(0),x.size(1)
        device = x.device
        adj = torch.Tensor(adj['adj_train'].A).to(device)
        positive = adj.bool()
        negative = ~positive

        x_1 = x.repeat(num,1)
        x_2 = x.repeat_interleave(num,0)

        dist = self.manifold.sqdist(x_1, x_2).view(num,num)
        inner = self.manifold.inner(x_1, x_2, keepdim=True).view(num,num)

        simi = torch.clamp(torch.exp(-dist),min=1e-15)
        positive_sim = simi * (positive.long())
        negative_sim = simi * (negative.long())
        
        edge_inner = inner*adj.bool()
        max_inner, min_inner = edge_inner.max().item(),edge_inner.min().item()
        
        negative_sum = negative_sim.sum(dim=1).unsqueeze(1).repeat(1,num)
        loss = torch.clamp(torch.div(positive_sim, negative_sum)[positive],min=1e-15)
        loss = (-torch.log(loss)).sum() 
        return x, dist, loss, dist.max().item(), max_inner, min_inner

    def extra_repr(self):
        return None


model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
    'HyboNet': LorentzDecoder,
    'LGCN': LinearDecoder,
    'H2HGCN': H2HDecoder,
    'QGCN' : LinearDecoder,
    'GIL' : DualDecoder,
    'HGAT': LinearDecoder,
    'HGNN' : LinearDecoder
}