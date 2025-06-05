import torch.nn as nn
import torch
from ..nn.conv import *
from ..manifolds import Lorentz

class Lorentz_ResNet(nn.Module):
    """ Implementation of ResNet models on manifolds. """
    def __init__(
        self,
        num_blocks,
        manifold_in:Lorentz,
        manifold_hidden:Lorentz,
        manifold_out:Lorentz,
        img_dim=[3,32,32],
        embed_dim=512,
        num_classes=100,
        bias=True,
        remove_linear=False,
    ):
        super(Lorentz_ResNet, self).__init__()

        self.img_dim = img_dim[0]
        self.in_channels = 64
        self.conv3_dim = 128
        self.conv4_dim = 256
        self.embed_dim = embed_dim
        self.bias = bias

        self.c = torch.Tensor([0.1]).cpu()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        self.conv1 = self._get_inConv()
        self.conv2_x = self._make_layer(out_channels=self.in_channels, num_blocks=num_blocks[0], stride=1)
        self.conv3_x = self._make_layer(out_channels=self.conv3_dim, num_blocks=num_blocks[1], stride=2)
        self.conv4_x = self._make_layer(out_channels=self.conv4_dim, num_blocks=num_blocks[2], stride=2)
        self.conv5_x = self._make_layer(out_channels=self.embed_dim, num_blocks=num_blocks[3], stride=2)
        self.avg_pool = self._get_GlobalAveragePooling()

        if remove_linear:
            self.predictor = None
        else:
            self.predictor = self._get_predictor(self.embed_dim*LorentzResidualBlock.expansion, num_classes)

    def forward(self, x):
        out = self.conv1(x)

        out_1 = self.conv2_x(out)
        out_2 = self.conv3_x(out_1)
        out_3 = self.conv4_x(out_2)
        out_4 = self.conv5_x(out_3)
        out = self.avg_pool(out_4)
        out = out.view(out.size(0), -1)

        if self.predictor is not None:
            out = self.predictor(out)

        return out

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(
                LorentzResidualBlock(
                    self.manifold_hidden,
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                )
            )
            self.in_channels = out_channels * LorentzResidualBlock.expansion + 1

        return nn.Sequential(*layers)

    def _get_inConv(self):
        return LorentzInputBlock(
            self.manifold_in,
            self.img_dim, 
            self.in_channels, 
            self.bias,
            self.manifold_hidden
        )

    def _get_predictor(self, in_features, num_classes):
        return LorentzMLR(self.manifold_out, in_features+1, num_classes)

    def _get_GlobalAveragePooling(self):
        return LorentzGlobalAvgPool2d(manifold_in=self.manifold_hidden, keep_dim=True, manifold_out=self.manifold_out)
    
def Lorentz_resnet18(manifold_in:Lorentz, manifold_hidden=None, manifold_out=None, **kwargs):
    if manifold_out is None:
        if manifold_hidden:
            manifold_out = manifold_hidden
        else:
            manifold_hidden = manifold_in
            manifold_out = manifold_in
    model = Lorentz_ResNet([2, 2, 2, 2], manifold_in, manifold_hidden, manifold_out, **kwargs)
    return model

def Lorentz_resnet34(manifold_in:Lorentz, manifold_hidden=None, manifold_out=None, **kwargs):
    if manifold_out is None:
        if manifold_hidden:
            manifold_out = manifold_hidden
        else:
            manifold_hidden = manifold_in
            manifold_out = manifold_in
    model = Lorentz_ResNet([3, 4, 6, 3], manifold_in, manifold_hidden, manifold_out, **kwargs)
    return model

def Lorentz_resnet50(manifold_in:Lorentz, manifold_hidden=None, manifold_out=None, **kwargs):
    if manifold_out is None:
        if manifold_hidden:
            manifold_out = manifold_hidden
        else:
            manifold_hidden = manifold_in
            manifold_out = manifold_in
    model = Lorentz_ResNet([3, 4, 6, 3], manifold_in, manifold_hidden, manifold_out, **kwargs)
    return model

def Lorentz_resnet101(manifold_in:Lorentz, manifold_hidden=None, manifold_out=None, **kwargs):
    if manifold_out is None:
        if manifold_hidden:
            manifold_out = manifold_hidden
        else:
            manifold_hidden = manifold_in
            manifold_out = manifold_in
    model = Lorentz_ResNet([3, 4, 23, 3], manifold_in, manifold_hidden, manifold_out, **kwargs)
    return model

def Lorentz_resnet152(manifold_in:Lorentz, manifold_hidden=None, manifold_out=None, **kwargs):
    if manifold_out is None:
        if manifold_hidden:
            manifold_out = manifold_hidden
        else:
            manifold_hidden = manifold_in
            manifold_out = manifold_in
    model = Lorentz_ResNet([3, 8, 36, 3], manifold_in, manifold_hidden, manifold_out, **kwargs)
    return model