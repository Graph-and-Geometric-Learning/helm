'''
Hyperbolic CLIP model example. Please see examples for hyperbolic ResNet, ViT, and Transformer encoder files for details on
how to implement the encoders.
'''
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
from ..utils import distributed as dist
from torch.distributed.nn import all_gather as nn_all_gather

def get_rank() -> int:
    """Return rank of current process in the process group."""
    return dist.get_rank() if dist.is_initialized() else 0

def gather_across_processes(t: torch.Tensor) -> list[torch.Tensor]:
    """
    Gather tensors from multiple GPU processes in a list. The order of elements
    is preserved by GPU process IDs. This operation is differentiable; gradients
    will be scattered back to devices in the backward pass.

    Args:
        t: Tensor to gather across processes.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [t]

    output = list(nn_all_gather(t))
    return output


class LCLIP(nn.Module):
    """
    Lorentz CLIP model, uses the hyperbolic contrastive and entrailment loss from MERU:
    Reference: 
    """

    def __init__(
        self,
        manifold,
        visual,
        textual,
        embed_dim: int,
        entail_weight: float = 0.0,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Un-documented args are same as `CLIPBaseline`.

        Args:
            entail_weight: Weight for the entailment loss component.
            manifold: embedding manifold of the CLIP model
            visual: Visual encoder
            textual: Text encoder
            embed_dim: embedding dimension of the CLIP model
        """
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.embed_dim = embed_dim
        self.entail_weight = entail_weight
        self.embed_dim =embed_dim

        self.manifold = manifold # embedding manifold of the CLIP model
        self.visual_manifold = self.visual.manifold_out # embedding manifold of the output of the visual encoder
        self.textual_manifold = self.textual.manifold_out # embedding manifold of the output of the text encoder
        self.entail_weight = entail_weight

        self.visual_proj = hnn.LorentzLinear(self.visual_manifold, visual.width, embed_dim, manifold_out=self.manifold) # project visual embedding to correct shape and manifold
        self.textual_proj = hnn.LorentzLinear(self.textual_manifold, textual.width, embed_dim, manifold_out=self.manifold) # project textual embedding to correct shape and manifold

        # CLIP-style initialization of projection layers.
        nn.init.normal_(self.visual_proj.linear.weight, std=visual.width**-0.5)
        nn.init.normal_(self.textual_proj.linear.weight, std=textual.width**-0.5)

        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        # Get rank of current GPU process for gathering features.
        self._rank = dist.get_rank()
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))
    
    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image(self, images: torch.Tensor):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """

        images = (images - self.pixel_mean) / self.pixel_std
        image_feats = self.visual(images)
        image_feats = self.visual_proj(image_feats)
        image_feats = hnn.LorentzNormalization(self.manifold)(image_feats)
        return image_feats

    def encode_text(self, tokens: list[torch.Tensor]):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        # Truncate tokens that are longer than context_length:
        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > self.textual.context_length:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.textual.context_length]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens

        # Pad all tokens on the right.
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = tokens.to(self.device)

        # shape: (batch_size, context_length, textual.width)
        text_feats = self.textual(tokens)
        # Get features for [EOS] position and apply projection. `[EOS]` token ID
        # is the largest number in the vocabulary of tokenizer.
        _eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0])
        text_feats = text_feats[batch_idxs, _eos_indices]
        text_feats = self.textual_proj(text_feats)
        text_feats = hnn.LorentzNormalization(self.manifold)(text_feats)
        return text_feats

    def forward(
        self, images: torch.Tensor, tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """
        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images)
        text_feats = self.encode_text(tokens)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute logits for contrastive loss.
            image_logits = -self.manifold.dist(image_feats.unsqueeze(0), all_text_feats.unsqueeze(1), keepdim=False)
            text_logits = -self.manifold.dist(text_feats.unsqueeze(0), all_image_feats.unsqueeze(1), keepdim=False)

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
            )

            # Hyperbolic entailment loss: text should entail matching image.
            _angle = self.manifold.oxy_angle(text_feats, image_feats)
            _aperture = self.manifold.half_aperture(text_feats)
            entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
            },
        }