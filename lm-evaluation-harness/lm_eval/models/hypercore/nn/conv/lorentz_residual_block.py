import torch.nn as nn
import torch.nn.functional as F
import torch

from ...manifolds import Lorentz
from ...nn.conv import *


def get_Conv2d(manifold_in, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, normalize=False, manifold_out=None):
    return LorentzConv2d(
        manifold_in=manifold_in,
        in_channels=in_channels+1, 
        out_channels=out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=bias, 
        normalize=normalize,
        manifold_out=manifold_out
    )

def get_BatchNorm2d(manifold_in, num_channels, manifold_out=None, space_method=False, momentum=0.1):
    return LorentzBatchNorm2d(manifold_in=manifold_in, num_channels=num_channels+1, manifold_out=manifold_out, space_method=space_method, momentum=momentum)

def get_Activation(manifold_in, act=F.relu, manifold_out=None):
    return LorentzActivation(manifold_in, act, manifold_out)


class LorentzInputBlock(nn.Module):
    """ Input Block of Pre-buidlt ResNet model """

    def __init__(self, manifold_in: Lorentz, img_dim, in_channels, bias=True, manifold_out=None):
        super(LorentzInputBlock, self).__init__()

        self.manifold = manifold_in
        if manifold_out is None:
            manifold_out = manifold_in

        self.conv = nn.Sequential(
            get_Conv2d(
                self.manifold,
                img_dim,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=bias.as_integer_ratio,
                manifold_out=manifold_out
            ),
            get_BatchNorm2d(manifold_out, in_channels + 1),
            get_Activation(manifold_out),
        )

        self.c = manifold_in.c

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # Make channel last (bs x H x W x C)
        x = self.manifold.projx(F.pad(x, pad=(1, 0)))
        return self.conv(x)


class LorentzResidualBlock(nn.Module):
    """ Basic Pre-built Block for Lorentz ResNet-10, ResNet-18 and ResNet-34 """

    expansion = 1

    def __init__(self, manifold_in: Lorentz, in_channels, out_channels, act=F.relu, stride=1, bias=True):
        super(LorentzResidualBlock, self).__init__()

        self.manifold = manifold_in

        self.c = manifold_in.c

        self.activation = get_Activation(self.manifold, self.c, act)

        self.conv = nn.Sequential(
            get_Conv2d(
                self.manifold,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels),
            get_Activation(self.manifold, act),
            get_Conv2d(
                self.manifold,
                out_channels,
                out_channels * LorentzResidualBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels * LorentzResidualBlock.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzResidualBlock.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    self.manifold,
                    in_channels,
                    out_channels * LorentzResidualBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias
                ),
                get_BatchNorm2d(
                    self.manifold, out_channels * LorentzResidualBlock.expansion
                ),
            )

        self.residual_connection = LResNet(self.manifold, scale=2.)

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)
        out = self.residual_connection(out, res)
        out = self.activation(out)
        return out


class LorentzBottleneck(nn.Module):
    """ Residual block for Lorentz ResNet with > 50 layers """

    expansion = 4

    def __init__(self, manifold_in: Lorentz, in_channels, out_channels, act=F.relu, stride=1, bias=False):
        super(LorentzBottleneck, self).__init__()

        self.manifold = manifold_in

        self.c = manifold_in.c

        self.activation = get_Activation(self.manifold, act)

        self.conv = nn.Sequential(
            get_Conv2d(
                self.manifold,
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels),
            get_Activation(self.manifold, act),
            get_Conv2d(
                self.manifold,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels),
            get_Activation(self.manifold, act),
            get_Conv2d(
                self.manifold,
                out_channels,
                out_channels * LorentzBottleneck.expansion,
                kernel_size=1,
                padding=0,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels * LorentzBottleneck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzBottleneck.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    self.manifold,
                    in_channels,
                    out_channels * LorentzBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias
                ),
                get_BatchNorm2d(
                    self.manifold, out_channels * LorentzBottleneck.expansion
                ),
            )
        
        self.residual_connection = LResNet(self.manifold, scale=2.)

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)
        out = self.residual_connection(out, res)
        out = self.activation(out)
        return out
