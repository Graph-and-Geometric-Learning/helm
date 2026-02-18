import torch
import torch.nn as nn
import math
from ...nn import PoincareLinear
from ...manifolds import PoincareBall
import torch.nn.functional as F

class PoincareAttentionLayer(nn.Module):
    def __init__(self, manifold: PoincareBall, c, conv_channels, embed_dim, bmm=None):
        super().__init__()
        self.manifold = manifold
        self.c = c
        # projects from output of convolution to embedding dimension
        self.in_projection = PoincareLinear(self.manifold, self.c, conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = PoincareLinear(self.manifold, self.c, embed_dim, conv_channels)

        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, target_embedding, encoder_out, encoder_padding_mask):
        residual = x

        # attention
        x = self.manifold.mobius_scalar_mul(
            torch.FloatTensor([math.sqrt(0.5)]).to(x), 
            self.manifold.mobius_add(target_embedding, self.in_projection(x), self.c), self.c)
        x = - self.manifold.dist(x, encoder_out[0], self.c) / self.scale.exp()

        # don't attend over padding
        if encoder_padding_mask is not None:
            x = x.masked_fill(
                encoder_padding_mask.unsqueeze(1),
                float('-inf')
            )

        # softmax over last dim
        x = F.softmax(x, dim=-1)
        attn_scores = x

        if hasattr(self, "beam_mm"):
            x = self.beam_mm(encoder_out[1], x)
        else:
            x = self.manifold.weighted_midpoint_bmm(encoder_out[1], x, self.c)

        # scale attention output (respecting potentially different lengths)
        s = encoder_out[1].size(1)
        if encoder_padding_mask is None:
            x = self.manifold.mobius_scalar_mul(
                torch.FloatTensor([math.sqrt(s)]).to(x), 
                x, self.c)
        else:
            s = s - encoder_padding_mask.type_as(x).sum(dim=1, keepdim=True)  # exclude padding
            s = s.unsqueeze(-1)
            x = self.manifold.mobius_scalar_mul(s.sqrt(), x, self.c)

        # project back
        x = self.manifold.mobius_scalar_mul(
            torch.FloatTensor([math.sqrt(0.5)]).to(x), 
            self.manifold.mobius_add(residual, self.out_projection(x), self.c), self.c)
        return x, attn_scores

    # def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
    #     """Replace torch.bmm with BeamableMM."""
    #     if beamable_mm_beam_size is not None:
    #         self.add_module('beam_mm', PoincareBeamableMM(beamable_mm_beam_size, ball=self.ball))