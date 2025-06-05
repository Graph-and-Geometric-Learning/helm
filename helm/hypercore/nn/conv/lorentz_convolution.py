"""
Lorentz convolutional layers

Based on:
    - Fully Hyperbolic Convolutional Neural Networks for Computer Vision (https://arxiv.org/abs/2303.15919)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from ...manifolds import Lorentz
from ...nn.linear import LorentzLinear
from ...nn.conv.conv_util_layers import *

class LorentzConv1d(nn.Module):
    """
    Lorentzian 1D Convolution Layer using the Lorentz model of hyperbolic space.

    Args:
        manifold_in (Lorentz): Input Lorentz manifold.
        in_channels (int): Number of input channels (including time dimension).
        out_channels (int): Number of output channels (excluding time).
        kernel_size (int): Width of the 1D convolutional kernel.
        stride (int): Stride of the convolution. Default: 1.
        padding (int): Zero-padding to apply on both sides. Default: 0.
        bias (bool): If True, adds a learnable bias to output. Default: True.
        normalize (bool): If True, applies LorentzNormalization after projection.
        manifold_out (Lorentz, optional): Target manifold for output.
    """
    def __init__(
            self,
            manifold_in: Lorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            normalize=False,
            manifold_out=None
    ):
        super(LorentzConv1d, self).__init__()

        self.manifold = manifold_in
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.c = manifold_in.c

        lin_features = (self.in_channels - 1) * self.kernel_size + 1

        self.linearized_kernel = LorentzLinear(
            manifold_in,
            lin_features, 
            self.out_channels, 
            self.c,
            bias=bias,
            manifold_out=manifold_out
        )

        self.manifold_out = manifold_out

        if normalize:
            if self.manifold_out:
                self.normalization = LorentzNormalization(manifold_out)
            else:
                self.normalization = LorentzNormalization(manifold_in)
        else:
            self.normalization = None

    def forward(self, x):
        """
        Forward pass of LorentzConv1d.

        Args:
            x (Tensor): Input tensor of shape [batch, length, channels].

        Returns:
            Tensor: Output tensor after Lorentz convolution.
        """

        """ x has to be in channel-last representation -> Shape = bs x len x C """
        bsz = x.shape[0]

        # origin padding
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.c.sqrt()) 

        patches = x.unfold(1, self.kernel_size, self.stride)
        # Lorentz direct concatenation of features within patches
        patches_time = patches.narrow(2, 0, 1)
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=(-2,-1), keepdim=True) - ((self.kernel_size - 1) * self.c))
        patches_time_rescaled = patches_time_rescaled.view(bsz, patches.shape[1], -1)

        patches_space = patches.narrow(2, 1, patches.shape[2]-1).reshape(bsz, patches.shape[1], -1)
        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        out = self.linearized_kernel(patches_pre_kernel)
        if self.normalization:
            out = self.normalization(out)
        return out


class LorentzConv2d(nn.Module):
    """
    Lorentzian 2D Convolution Layer using the Lorentz model of hyperbolic space.

    Args:
        manifold_in (Lorentz): Input Lorentz manifold.
        in_channels (int): Number of input channels (including time).
        out_channels (int): Number of output channels (excluding time).
        kernel_size (int or tuple): Size of 2D convolutional kernel.
        stride (int or tuple): Stride of the convolution. Default: 1.
        padding (int or tuple): Amount of zero-padding. Default: 0.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
        bias (bool): If True, adds bias to the output. Default: True.
        normalize (bool): Whether to apply LorentzNormalization after projection.
        manifold_out (Lorentz, optional): Output manifold for projection.
    """
    def __init__(
            self,
            manifold_in: Lorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            normalize=False,
            manifold_out=None
    ):
        super(LorentzConv2d, self).__init__()

        self.manifold = manifold_in
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.c = manifold_in.c
        self.manifold_out = manifold_out

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.linearized_kernel = LorentzLinear(
            self.manifold,
            lin_features, 
            self.out_channels,
            bias=bias,
            manifold_out=self.manifold_out
        )

        if normalize:
            if self.manifold_out:
                self.normalization = LorentzNormalization(manifold_out)
            else:
                self.normalization = LorentzNormalization(manifold_in)
        else:
            self.normalization = None

        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        self.linearized_kernel.linear.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.linearized_kernel.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Forward pass of LorentzConv2d.

        Args:
            x (Tensor): Input tensor with shape [batch, height, width, channels].

        Returns:
            Tensor: Output tensor of shape [batch, H_out, W_out, out_channels+1].
        """

        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]
        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, channels * elements/window, windows
        patches = patches.permute(0, 2, 1)
        
        # Now we have flattened patches with multiple time elements -> fix the concatenation to perform Lorentz direct concatenation by Qu et al. (2022)
        patches_time = torch.clamp(patches.narrow(-1, 0, self.kernel_len), min=self.c.sqrt())  # Fix zero (origin) padding
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((self.kernel_len - 1) * self.c))

        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape) # No need, but seems to improve runtime??
        
        patches_pre_kernel = torch.cat((patches_time_rescaled, patches_space), dim=-1)
        out = self.linearized_kernel(patches_pre_kernel)
        if self.normalization:
            out = self.normalization(out)
        out = out.view(bsz, h_out, w_out, self.out_channels + 1)

        return out

class LorentzConvTranspose2d(nn.Module):
    """
    Lorentzian Transposed 2D Convolution (a.k.a. Deconvolution) Layer.

    Args:
        manifold_in (Lorentz): Input Lorentz manifold.
        in_channels (int): Number of input channels (including time).
        out_channels (int): Number of output channels (excluding time).
        kernel_size (int or tuple): Size of the kernel.
        stride (int or tuple): Stride size.
        padding (int or tuple): Padding size.
        output_padding (int or tuple): Padding added to the output shape.
        bias (bool): Whether to include bias.
        normalize (bool): Whether to apply normalization.
        manifold_out (Lorentz, optional): Output manifold for projection.
    """
    def __init__(
            self, 
            manifold_in: Lorentz,
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            output_padding=0, 
            bias=True,
            normalize=False,
            manifold_out=None
        ):
        super(LorentzConvTranspose2d, self).__init__()

        self.manifold = manifold_in
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c = manifold_in.c
        self.manifold_out = manifold_out

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding)
        else:
            self.output_padding = output_padding

        padding_implicit = [0,0]
        padding_implicit[0] = kernel_size - self.padding[0] - 1 # Ensure padding > kernel_size
        padding_implicit[1] = kernel_size - self.padding[1] - 1 # Ensure padding > kernel_size

        self.pad_weight = nn.Parameter(F.pad(torch.ones((self.in_channels,1,1,1)),(1,1,1,1)), requires_grad=False)

        self.conv = LorentzConv2d(
            manifold=self.manifold,
            c=self.c,
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding_implicit, 
            bias=bias, 
            normalize=normalize,
            manifold_out=self.manifold_out
        )

    def forward(self, x):
        """
        Forward pass for LorentzConvTranspose2d.

        Args:
            x (Tensor): Input tensor [batch, height, width, channels].

        Returns:
            Tensor: Upsampled output tensor.
        """

        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        if self.stride[0] > 1 or self.stride[1] > 1:
            # Insert hyperbolic origin vectors between features
            x = x.permute(0,3,1,2)
            # -> Insert zero vectors
            x = F.conv_transpose2d(x, self.pad_weight,stride=self.stride,padding=1, groups=self.in_channels)
            x = x.permute(0,2,3,1)
            x[..., 0].clamp_(min=self.c.sqrt())

        x = self.conv(x)

        if self.output_padding[0] > 0 or self.output_padding[1] > 0:
            x = F.pad(x, pad=(0, self.output_padding[1], 0, self.output_padding[0])) # Pad one side of each dimension (bottom+right) (see PyTorch documentation)
            if self.manifold_out: # Fix origin padding
                x[..., 0].clamp_(min=self.manifold_out.c.sqrt())
            else:
                x[..., 0].clamp_(min=self.c.sqrt()) 
        return x
