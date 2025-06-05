from __future__ import annotations
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
from ..optimizers import Optimizer, LR_Scheduler
import numpy as np
import random
import time
import math
import logging
import torch
import torch.nn as nn
from .. import nn as hnn
import os
from ..manifolds import Lorentz
import re
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint

class _LTransformerBlock(nn.Module):
    def __init__(self, manifold, d_model: int, n_head: int):
        super().__init__()
        dim_per_head = d_model // n_head
        self.manifold = manifold
        self.attn = hnn.LorentzMultiheadAttention(manifold, dim_per_head, dim_per_head, n_head, attention_type='full', trans_heads_concat=True)
        self.ln_1 = hnn.LorentzLayerNorm(manifold, d_model - 1)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", hnn.LorentzLinear(manifold, d_model, d_model * 4 - 1)),
                    ("gelu", hnn.LorentzActivation(manifold, activation=nn.GELU())),
                    ("c_proj", hnn.LorentzLinear(manifold, d_model * 4, d_model - 1)),
                ]
            )
        )
        self.ln_2 = hnn.LorentzLayerNorm(manifold, d_model - 1)
        self.res1 = hnn.LResNet(manifold, use_scale=True, scale=50.0)
        self.res2 = hnn.LResNet(manifold, use_scale=True, scale=50.0)

    def forward(self, x, attn_mask=None):
        lx = self.ln_1(x)
        ax = self.attn(lx, lx, output_attentions=False, mask=attn_mask)
        x = self.res1(x, ax)
        x = self.res2(x, self.mlp(self.ln_2(x)))
        return x


class LTransformerEncoder(nn.Module):
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
        """
        Args:
            arch: Architecture config for transformer, describing layers, width,
                and number of attention heads. For example, `L12_W512_A8` has 1
                layer, 512 width, 8 heads. Width of MLP will always be `4 * W`,
                per transformer paper. `A` is optional and will default to
                (`A = H/64`) per transformer paper.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.grad_checkpointing = grad_checkpointing
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        # Parse architecture str: layers, width, heads, feed-forward size.
        self.layers = int(re.search(r"L(\d+)", arch).group(1))
        self.width = int(re.search(r"W(\d+)", arch).group(1))

        # Find heads in architecture else use (H // 64) per (Vaswani et al.)
        _attn = re.search(r"A(\d+)", arch)
        self.heads = int(_attn.group(1)) if _attn else self.width // 64

        self.token_embed = hnn.LorentzEmbeddings(manifold_in, vocab_size, self.width, manifold_out=manifold_hidden) #this step automatically adds the positional embedding

        # Make a sequential module of transformer encoder blocks.
        _resblocks = [
            _LTransformerBlock(manifold_hidden, self.width, self.heads) for _ in range(self.layers)
        ]
        self.resblocks = nn.ModuleList(_resblocks)
        self.ln_final = hnn.LorentzLayerNorm(manifold_out, self.width - 1)
        self.final_proj = hnn.LorentzLinear(manifold_hidden, self.width, self.width - 1, manifold_out=manifold_out)

        # Generate a unidirectional mask for self-attention. As per PyTorch API,
        # masked positions are set to `-inf`.
        attn_mask = torch.triu(
            torch.full((context_length, context_length), float("-inf")), diagonal=1
        )
        self.register_buffer("attn_mask", attn_mask.bool())

        # Initialize all modules like CLIP:
        # nn.init.normal_(self.token_embed.weight, std=0.02)
        # nn.init.normal_(self.posit_embed.data, std=0.01)

        out_proj_std = (2 * self.width * self.layers) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.mlp[0].linear.weight, std=(self.width) ** -0.5)
            nn.init.normal_(block.mlp[2].linear.weight, std=out_proj_std)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        max_len = text_tokens.shape[-1]
        _attn_mask = self.attn_mask[:max_len, :max_len]

        # shape: (batch_size, context_length, width)
        token_embeddings = self.token_embed(text_tokens)

        # Forward pass through transformer, optionally with grad checkpointing.
        textual_features = token_embeddings
        for block in self.resblocks:
            if self.grad_checkpointing and self.training:
                # shape: (context_length, batch_size, width)
                textual_features = checkpoint(block, textual_features, _attn_mask)
            else:
                textual_features = block(textual_features, _attn_mask)
        textual_features = self.final_proj(textual_features)
        textual_features = self.ln_final(textual_features)
        return textual_features