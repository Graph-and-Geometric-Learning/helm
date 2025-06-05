'''
Example of building a hyperbolic vision transformer
'''
from tqdm import tqdm
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from .. import nn as hnn
import torch.nn.functional as F
from ..manifolds import Lorentz
import numpy as np
import math
from geoopt import ManifoldParameter


class HyperbolicMLP(nn.Module):
    """
    A hyperbolic multi-layer perceptron module.
    """

    def __init__(self, manifold, in_channel, hidden_channel, dropout=0):
        super().__init__()
        self.manifold = manifold
        self.dense_1 = hnn.LorentzLinear(self.manifold, in_channel, hidden_channel - 1)
        self.dense_2 = hnn.LorentzLinear(self.manifold, hidden_channel, in_channel - 1)
        self.activation = hnn.LorentzActivation(manifold, activation=nn.GELU())
        self.dropout = hnn.LorentzDropout(self.manifold, dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
    
class LorentzTransformerBlock(nn.Module):
    """
    A single lorentz transformer block.
    """

    def __init__(self, manifold, in_channel, hidden_channel, dropout=0, num_heads=1, output_attentions=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = hnn.LorentzMultiheadAttention(manifold, in_channel, in_channel, num_heads, attention_type='full', trans_heads_concat=True)
        self.layernorm_1 = hnn.LorentzLayerNorm(manifold, num_heads * in_channel - 1)
        self.layernorm_2 = hnn.LorentzLayerNorm(manifold, num_heads * in_channel - 1)
        self.mlp = HyperbolicMLP(manifold, num_heads * in_channel, hidden_channel, dropout=dropout)
        self.manifold = manifold
        self.residual_1 = hnn.LResNet(manifold, use_scale=True, scale=27.5)
        self.residual_2 = hnn.LResNet(manifold, use_scale=True, scale=27.5)
        self.in_channel = in_channel

    def forward(self, x, output_attentions=False):
        # Self-attention
        x = self.layernorm_1(x)
        attention_output = self.attention(x, x, output_attentions=output_attentions)
        #Skip connection
        x = self.residual_1(x, attention_output)
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = self.residual_2(x, mlp_output)
        return x
    
class LViTEncoder(nn.Module):
    """
    The lorentzian vision transformer encoder module.
    """

    def __init__(self, manifold_in, num_layers, in_channel, hidden_channel, num_heads=1, dropout=0, output_attentions=False, manifold_out=None):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        self.num_heads = num_heads
        for _ in range(num_layers):
            block = LorentzTransformerBlock(manifold_in, in_channel, hidden_channel, dropout, num_heads, output_attentions)
            self.blocks.append(block)
        # self.fc = hnn.LorentzLinear(manifold_in, self.num_heads * in_channel, self.num_heads * in_channel - 1, manifold_out=manifold_out)
        self.manifold_out = manifold_out
        self.manifold = manifold_in
        self.in_channel = in_channel

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        for block in self.blocks:
            x = block(x, output_attentions=output_attentions)
        return x


class LViT(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, 
                 manifold_in, 
                 manifold_hidden, 
                 manifold_out,
                 image_size=224, 
                 patch_size=16,
                 num_layers=12, 
                 in_channel=3, 
                 hidden_channel=65, 
                 out_channel=1000, 
                 mlp_hidden_size=65*4*12 + 1, 
                 num_heads=12, 
                 dropout=0.1, 
                 output_attentions=False):
        super().__init__()
        self.in_channel = in_channel + 1
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_patches = (image_size // patch_size) ** 2
        self.num_heads = num_heads
        self.width = self.num_heads * self.hidden_channel
        # Create the embedding module
        self.patch_embedding = hnn.LorentzPatchEmbedding(manifold_in, image_size, patch_size, self.in_channel, self.num_heads * self.hidden_channel - 1)
        self.pe = ManifoldParameter(self.manifold_in.random_normal((1, self.num_patches, num_heads * self.hidden_channel)), manifold=self.manifold_in, requires_grad=True)
        self.add_pos = hnn.LResNet(manifold_in, use_scale=True, scale=1.0)
        # Create the transformer encoder module
        self.encoder = LViTEncoder(self.manifold_hidden, self.num_layers, self.hidden_channel, mlp_hidden_size, num_heads, dropout, output_attentions)
        if self.out_channel > 0:
            self.classifier = hnn.LorentzMLR(self.manifold_out, self.num_heads * self.hidden_channel, self.out_channel)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        assert(not self.pe.isnan().any())
        assert(not self.pe.isinf().any())
        x = x.permute(0, 2, 3, 1) 
        x_hyp = self.manifold_in.projx(F.pad(x, pad=(1, 0)))
        embedding_output = self.patch_embedding(x_hyp)
        embedding_output = self.add_pos(embedding_output, self.pe)
        # Calculate the encoder's output
        encoder_output = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits and return
        if self.out_channel > 0:
            out = self.classifier(self.manifold_out.lorentzian_centroid(encoder_output))
        else:
            out = self.manifold_out.lorentzian_centroid(encoder_output)
        assert(not out.isnan().any())
        assert(not out.isinf().any())
        return out