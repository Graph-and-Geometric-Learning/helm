import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out