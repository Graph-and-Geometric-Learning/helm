import torch
import torch.nn as nn
from ...nn.linear import LorentzLinear
from ...nn.conv import LResNet
from ...manifolds import Lorentz
from geoopt import ManifoldParameter
import warnings
import math
class LorentzEmbeddings(nn.Module):
    """Words embeddings for encoder/decoder, includes positional embedding
    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.
    Adapted from "Fully hyperbolic neural networks" (chen2021fully)
    Args:
        manifold_in (Lorentz): manifold of the inputs
        num_embeddings (int): size of the vocab
        embedding_dim (int): size of the word embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feat_padding_idx (List[int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes (List[int], optional): list of size of dictionary
            of embeddings for each feature.
        position_encoding (bool): see :class:`~onmt.modules.PositionalEncoding`
        feat_merge (string): merge action for the features embeddings:
            concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
            embedding size is N^feat_dim_exponent, where N is the
            number of values the feature takes.
        feat_vec_size (int): embedding dimension for features when using
            `-feat_merge mlp`
        dropout (float): dropout probability.
        posit_embed (bool): if True, adds positional embedding internally, else return only the embedding
        manifold_out (Lorentz): manifold of the outputs, should only be set if posit_embed is true
    """

    def __init__(self, 
                 manifold_in: Lorentz,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 feat_merge="concat",
                 feat_vec_exponent=0.7,
                 feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 max_len = 5000,
                 posit_embed=True,
                 manifold_out=None):
        super(LorentzEmbeddings, self).__init__()
        word_vocab_size = num_embeddings
        word_vec_size = embedding_dim
        word_padding_idx = padding_idx
        if feat_vocab_sizes is None:
            feat_vocab_sizes = []
        if feat_padding_idx is None:
            feat_padding_idx = []
        self._validate_args(feat_merge, feat_vocab_sizes, feat_vec_exponent,
                            feat_vec_size, feat_padding_idx)

        self.word_padding_idx = word_padding_idx

        word_vec_size = word_vec_size
        self.word_vec_size = word_vec_size
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.posit_embed = posit_embed
        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = []
        if word_padding_idx is not None:
            pad_indices.append(word_padding_idx)

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)
        if self.posit_embed:
            self.poisitional_encoding = ManifoldParameter(self.manifold_in.random_normal((max_len, 1, self.embedding_size), std=math.sqrt(0.02)), manifold=self.manifold_in, requires_grad=True)
            self.point = LorentzLinear(self.manifold_in, self.embedding_size, self.embedding_size - 1, manifold_out=self.manifold_out)
        self.add_pos = LResNet(self.manifold_in, use_scale=True)
        self.embedding = ManifoldParameter(self.manifold_in.random_normal(
            (vocab_sizes[0], emb_dims[0])), manifold=self.manifold_in)
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}

    def _validate_args(self, feat_merge, feat_vocab_sizes, feat_vec_exponent,
                       feat_vec_size, feat_padding_idx):
        if feat_merge == "sum":
            # features must use word_vec_size
            if feat_vec_exponent != 0.7:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_exponent. It will be unused.")
            if feat_vec_size != -1:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_size. It will be unused.")
        elif feat_vec_size > 0:
            # features will use feat_vec_size
            if feat_vec_exponent != -1:
                warnings.warn("Not merging with sum and positive "
                              "feat_vec_size, but got non-default "
                              "feat_vec_exponent. It will be unused.")
        else:
            if feat_vec_exponent <= 0:
                raise ValueError("Using feat_vec_exponent to determine "
                                 "feature vec size, but got feat_vec_exponent "
                                 "less than or equal to 0.")
        n_feats = len(feat_vocab_sizes)
        if n_feats != len(feat_padding_idx):
            raise ValueError("Got unequal number of feat_vocab_sizes and "
                             "feat_padding_idx ({:d} != {:d})".format(
                                 n_feats, len(feat_padding_idx)))
    def forward(self, source, use_feats=False, batch_first=True, step=None):
        assert source.max() < self.embedding.size(0), "Token index cannot exceed vocab size"
        if use_feats:
            if batch_first:
                source = source.permute(1, 0, 2).contiguous()
            shape = source.shape[:-1]
            source = self.embedding.index_select(
                0, source.view(-1, )).view(shape + (-1,))
        else:
            if batch_first:
                source = source.permute(1, 0).contiguous()
            source = self.embedding.index_select(
                0, source.view(-1, )).view(source.shape + (-1, ))
        if self.posit_embed:
            pe = self.poisitional_encoding[:source.size(0)] if step is None else self.poisitional_encoding[step]
            emb = self.add_pos(source, pe)
            if batch_first:
                emb = emb.permute(1, 0, 2)
            emb = self.point(emb)
        else:
            emb = source
            emb = emb * (self.manifold_out.c / self.manifold_in.c).sqrt()
            emb = emb.permute(1, 0, 2)
        return emb