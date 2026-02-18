"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.graph_decoders import model2decoder, MDDecoder
from ..utils.eval_utils import acc_f1, MarginLoss
from ..manifolds import Lorentz
import networkx as nx
import scipy.sparse as sp
from ..utils import distortions as dis
from geoopt.manifolds import PoincareBall

class BaseModel(nn.Module):

    def __init__(self, encoder, manifold, device='cpu'):
        super(BaseModel, self).__init__()
        self.manifold_name = manifold.name
        self.c = encoder.c
        self.manifold = manifold
        self.encoder = encoder
        self.device = device
    def encode(self, x, input=None):
        h = self.encoder.encode(x, input)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, encoder, decoder, act=None, device='cpu'):
        super(NCModel, self).__init__(encoder, decoder.manifold, device)
        self.decoder = decoder
        if decoder.output_dim > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        self.weights = torch.Tensor([1.] * decoder.output_dim)
        self.weights = self.weights.to(self.device)
        self.act = act
    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        if self.act is not None:
            return self.act(output[idx], dim=1)
        # if self.model_name == 'H2HGCN' or self.model_name == 'QGCN' or self.model_name == 'GIL' or self.model_name == 'HGNN' or self.model_name == 'HGAT':
        #     return F.log_softmax(output[idx], dim=1)
        return output[idx]

    def compute_metrics(self, embeddings, data, split, loss_fn):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = loss_fn(output, data['labels'][idx], self.weights)
        # if self.model_name == 'HyboNet':
        #     correct = output.gather(1, data['labels'][idx].unsqueeze(-1))
        #     loss = F.relu(self.margin - correct + output).mean()
        # elif self.model_name == 'H2HGCN' or self.model_name == 'QGCN' or self.model_name == 'GIL' or self.model_name == 'HGNN' or self.model_name == 'HGAT':
        #     loss = F.nll_loss(output, data['labels'][idx], self.weights)
        # else:
        #     loss = F.cross_entropy(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, encoder, manifold, nb_false_edges, nb_edges,  weights_dim=None, device='cpu', max_norm=None, decode_type='probability'):
        super(LPModel, self).__init__(encoder, manifold, device)
        self.dc = FermiDiracDecoder(r=2., t=1.)
        self.nb_false_edges = nb_false_edges
        self.nb_edges = nb_edges
        if weights_dim:
            self.w_e = nn.Linear(weights_dim, 1, bias=False)
            self.w_h = nn.Linear(weights_dim, 1, bias=False)
            self.reset_param()
        self.max_norm = max_norm
        self.decode_type = decode_type
    def reset_param(self):
        self.w_e.reset_parameters()
        self.w_h.reset_parameters()

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        if isinstance(h, tuple):
            # GIL
            emb_in = h[0][idx[:, 0], :]
            emb_out = h[0][idx[:, 1], :]
            "compute hyperbolic dist"
            emb_in = self.manifold.logmap0(emb_in)
            emb_out = self.manifold.logmap0(emb_out)
            sqdist_h = torch.sqrt((emb_in - emb_out).pow(2).sum(dim=-1) + 1e-15)
            if self.max_norm:
                sqdist_h = sqdist_h.clamp_max(self.max_norm)
            probs_h = self.dc.forward(sqdist_h)

            "compute dist in Euclidean"
            emb_in_e = h[1][idx[:, 0], :]
            emb_out_e = h[1][idx[:, 1], :]
            sqdist_e = torch.sqrt((emb_in_e - emb_out_e).pow(2).sum(dim=-1) + 1e-15)
            probs_e = self.dc.forward(sqdist_e)

            # sub
            w_h = torch.sigmoid(self.w_h(emb_in - emb_out).view(-1))
            w_e = torch.sigmoid(self.w_e(emb_in_e - emb_out_e).view(-1))
            w = torch.cat([w_h.view(-1, 1), w_e.view(-1, 1)], dim=-1)
            # if self.data == 'pubmed':
            #     w = F.normalize(w, p=1, dim=-1)
            probs = torch.sigmoid(w[-1, 0] * probs_h + w[-1, 1] * probs_e)
            assert torch.min(probs) >= 0
            assert torch.max(probs) <= 1
            return probs
        
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out)
        if self.max_norm:
            sqdist = sqdist.clamp_max(self.max_norm)
        if self.decode_type == 'sqdist':
            return -sqdist
        elif self.decode_type == 'probability':
            probs = self.dc.forward(sqdist)
            return probs
        else:
            raise NotImplementedError('LP decoder only supported for hyperbolic square distance or edge probability')

    def compute_metrics(self, embeddings, data, split, loss_fn):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = loss_fn(pos_scores, neg_scores)
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])
    
class MDModel(BaseModel):
    """
    Base model for minimizing distrotion task.
    """

    def __init__(self, args):
        super(MDModel, self).__init__(args)
        self.decoder = MDDecoder(self.c, self.manifold, args)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return output

    def compute_metrics(self, embeddings, data, split):
        if isinstance(embeddings, tuple):
            device = embeddings[0].device
        else:
            device = embeddings.device
        x, emb_dist, loss, max_dist,imax, imin = self.decode(embeddings,data,None)
        G = data['G']
        n = G.order()
        G = nx.to_scipy_sparse_array(G, nodelist=list(range(G.order())))
        true_dist = (torch.Tensor(data['labels'])).to(device)

        mask = np.array(np.triu(np.ones((true_dist.shape[0],true_dist.shape[0]))) - np.eye(true_dist.shape[0], true_dist.shape[0]), dtype=bool)
        mapscore = dis.map_score(np.array(sp.csr_matrix.todense(G)), emb_dist.cpu().detach().numpy(), n, 16)

        true_dist = true_dist[mask] 
        emb_dist = emb_dist[mask]
        
        distortion = (((emb_dist)/(true_dist)-1)**2).mean()
        
        metrics = {'loss': loss, 'distortion':distortion, 'mapscore':mapscore,'c': self.c.item(),'max_dist':max_dist, 'imax':imax, 'imin':imin}
        return metrics

    def init_metric_dict(self):
        return {'distortion':1, 'mapscore': -1,'c': -1,'max_dist':0}

    def has_improved(self, m1, m2):
        return m1["mapscore"] < m2["mapscore"]

class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs