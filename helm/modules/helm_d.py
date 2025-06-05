import torch
from collections import OrderedDict
# import torch.nn as nn
import helm.hypercore.nn as nn
from helm.hypercore.manifolds import Lorentz
import re
import math
from typing import Literal
import numpy as np
from dataclasses import dataclass
import torch.nn.functional as F
from helm.hypercore.models import LorentzFeedForward
import math

def precompute_theta_pos_frequencies(head_dim, seq_len, theta: float = 10000.0):
    head_dim -= 1
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)) # (Head_Dim / 2)
    m = torch.arange(seq_len)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

class _LTransformerDecoderBlock(torch.nn.Module):
    """
    A single Transformer block for the decoder.
    - Uses **masked** self-attention (causal).
    - Uses hyperbolic normalization and activation.
    """

    def __init__(self, manifold, d_model: int, n_head: int):
        super().__init__()
        dim_per_head = d_model // n_head
        self.manifold = manifold

        self.attn = nn.LorentzMultiheadAttention(
            manifold, dim_per_head, dim_per_head, n_head,
            attention_type='full', trans_heads_concat=True
        )

        self.ln_1 = nn.LorentzRMSNorm(manifold, d_model - 1)

        # MLP (Feed-forward network)
        self.mlp = LorentzFeedForward(manifold, d_model, d_model * 4)

        self.ln_2 = nn.LorentzRMSNorm(manifold, d_model - 1)
        self.res1 = nn.LResNet(manifold, use_scale=True, scale=4.0 * math.sqrt(d_model))
        self.res2 = nn.LResNet(manifold, use_scale=True, scale=4.0 * math.sqrt(d_model))

    def forward(self, x, attn_mask=None, rope=None):
        lx = self.ln_1(x)
        ax = self.attn(lx, lx, output_attentions=False, mask=attn_mask, rot_pos=rope)  # Masked attention
        x = self.res1(x, ax)
        x = self.res2(x, self.mlp(self.ln_2(x)))
        return x
    
class LTransformerDecoder(torch.nn.Module):
    """
    A decoder-only Transformer (like LLAMA) that:
    - Uses **causal attention mask** (future tokens are masked).
    - Outputs **logits** for next-token prediction.
    """

    def __init__(
        self,
        manifold_in: Lorentz,
        manifold_hidden: Lorentz,
        manifold_out: Lorentz,
        arch: str,
        vocab_size: int,
        context_length: int,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.grad_checkpointing = grad_checkpointing
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        # Parse architecture string
        self.layers = int(re.search(r"L(\d+)", arch).group(1))
        self.width = int(re.search(r"W(\d+)", arch).group(1))
        _attn = re.search(r"A(\d+)", arch)
        self.heads = int(_attn.group(1)) if _attn else self.width // 64
        # Token Embeddings (Lorentz)
        self.token_embed = nn.LorentzEmbeddings(manifold_in, vocab_size, self.width, manifold_out=manifold_hidden, posit_embed=False)  # Adds positional embedding automatically

        # Transformer Blocks (Decoder Only)
        self.resblocks = torch.nn.ModuleList([
            _LTransformerDecoderBlock(manifold_hidden, self.width, self.heads)
            for _ in range(self.layers)
        ])

        # Final normalization and projection
        self.ln_final = nn.LorentzRMSNorm(manifold_hidden, self.width - 1)
        self.final_proj = nn.LorentzLinear(
            manifold_hidden, self.width, self.width - 1, manifold_out=manifold_hidden
        )

        self.mapping = torch.nn.Linear(self.width, self.vocab_size, bias=False)

        # **Causal Attention Mask (Precomputed)**
        attn_mask = torch.triu(
            torch.full((context_length, context_length), float("-inf")), diagonal=1
        )
        self.register_buffer("attn_mask", attn_mask.bool())
        rope_vals = precompute_theta_pos_frequencies(self.width// self.heads, self.context_length)
        self.register_buffer("freqs_complex", rope_vals)
    def forward(self, 
            text_tokens: torch.Tensor,  
            attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Given input tokens, return logits for next-token prediction.
        """
        max_len = text_tokens.shape[-1]
        causal = self.attn_mask[:max_len, :max_len]   
        if attn_mask is not None:                      # (B, N, N) 
            _attn_mask = causal.unsqueeze(0) | attn_mask
        else:
            _attn_mask = causal

        # shape: (batch_size, context_length, width)
        token_embeddings = self.token_embed(text_tokens)
        freqs_cis = self.freqs_complex[:max_len]

        # Forward pass through Transformer blocks
        decoder_features = token_embeddings
        for block in self.resblocks:
            decoder_features = block(decoder_features, _attn_mask, freqs_cis)

        decoder_features = self.final_proj(decoder_features)
        decoder_features = self.ln_final(decoder_features)

        # shape: (batch_size, context_length, hidden_dim(width)+1)
        logits = self.mapping(decoder_features).float()
        # shape: (batch_size, context_length, vocab_size)
        return logits