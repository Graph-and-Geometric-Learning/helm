import helm.hypercore.nn as nn
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
# from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from helm.hypercore.manifolds import Lorentz
import math
import numpy as np
from helm.hypercore.models.lorentz_feedforward import LorentzFeedForward
from .hmla import LorentzMLA
from .mice import LorentzMoE

global world_size, rank
world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0

def precompute_freqs_cis(args) -> torch.Tensor:
    """
    Taken from Deepseekv3: 

    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args: Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim - 1
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

class Block(torch.nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        manifold (Lorentz): Input manifold
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
        train (bool): if True, return relevant information for load balancing
    """
    def __init__(self, manifold: Lorentz, layer_id: int, args):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args: Model arguments containing block parameters.
        """
        super().__init__()
        self.manifold = manifold
        self.attn = LorentzMLA(self.manifold, args)
        self.ffn = LorentzFeedForward(manifold, args.dim, args.inter_dim) if layer_id < args.n_dense_layers else LorentzMoE(manifold, args)
        self.attn_norm = nn.LorentzRMSNorm(self.manifold, args.dim - 1)
        self.ffn_norm = nn.LorentzRMSNorm(self.manifold, args.dim - 1)
        self.attn_res = nn.LResNet(self.manifold, use_scale=True, scale=math.sqrt(args.dim), learn_scale=False)
        self.ffn_res = nn.LResNet(self.manifold, use_scale=True, scale=math.sqrt(args.dim), learn_scale=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = self.attn_res(x, self.attn(self.attn_norm(x), start_pos, freqs_cis, mask))
        if self.training:
            if isinstance(self.ffn, LorentzMoE):
                x_ffn, idx, scores = self.ffn(self.ffn_norm(x))     # MoE returns indices
            else:
                x_ffn = self.ffn(self.ffn_norm(x))     # dense FFN
                idx = None
                scores = None
            x = self.ffn_res(x, x_ffn)
            return x, idx, scores
        else:
            x_ffn = self.ffn(self.ffn_norm(x)) 
            x = self.ffn_res(x, x_ffn)
            return x
        
class LorentzDeepSeekV3(torch.nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        manifold_in (Lorentz): input manifold
        manifold_hidden (Lorentz): intermediate embedding manifold
        manifold_out (Lorentz): output manifold
        max_seq_len (int): Maximum sequence length for the transformer.
        embed: Lorentz word embedding layer for input tokens.
        layers: List of Lorentz transformer blocks.
        norm: Lorentz RMS layer normalization applied after all blocks.
        head: Output projection layer mapping to vocabulary size, Lorentz linear layer
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
        train (bool): if True, return relevant information for load balancing 
    """
    def __init__(self, args, manifold_in, manifold_hidden, manifold_out):
        """
        Initializes the Transformer model.

        Args:
            args: Model arguments containing transformer parameters.
        """
        global rank
        rank = dist.get_rank() if dist.is_initialized() else 0
        super().__init__()

        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        self.max_seq_len = args.max_seq_len
        self.train_curv = args.train_curv
        self.project_emb = args.project_emb
        if not self.project_emb:
            self.embed = nn.LorentzEmbeddings(self.manifold_in, args.vocab_size, args.dim, manifold_out=self.manifold_hidden, posit_embed=False)
        else:
            self.embed = torch.nn.Embedding(args.vocab_size, args.dim-1)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(manifold_hidden, layer_id, args))
        self.final_proj = nn.LorentzLinear(manifold_hidden, args.dim, args.dim-1, manifold_out=manifold_out)
        self.norm = nn.LorentzRMSNorm(self.manifold_out, args.dim - 1)

        self.head = torch.nn.Linear(args.dim, args.vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        context_length = args.max_seq_len
        attn_mask = torch.triu(
            torch.full((context_length, context_length), float("-inf")), diagonal=1
        )
        self.register_buffer("attn_mask", attn_mask.bool())
    
    def project(self, x):
        x_time = ((x ** 2).sum(dim=-1, keepdims=True) + self.manifold_in.c) ** 0.5
        x = torch.cat([x_time, x], dim=-1)
        return x

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, attn_mask: torch.Tensor | None = None):
        seqlen = tokens.size(-1)
        if self.project_emb:
            h = self.project(self.embed(tokens))
        else:
            h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        causal = self.attn_mask[:seqlen, :seqlen] 
        if attn_mask is not None:                      # (B, N, N) 
            _attn_mask = causal.unsqueeze(0) | attn_mask
        else:
            _attn_mask = causal
        if self.training:
            all_indices, all_scores = [], []
            for layer in self.layers:
                h, idx, scr  = layer(h, start_pos, freqs_cis, _attn_mask)
                if idx is not None:
                    all_indices.append(idx)
                    all_scores.append(scr)
            h = self.final_proj(h, return_space=True)
            h = self.norm(h, space_only=True)
            logits = self.head(h).float()
            return logits, all_indices, all_scores
        else:
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, _attn_mask)
            h = self.final_proj(h, return_space=True)
            h = self.norm(h, space_only=True)
            logits = self.head(h).float()
            return logits