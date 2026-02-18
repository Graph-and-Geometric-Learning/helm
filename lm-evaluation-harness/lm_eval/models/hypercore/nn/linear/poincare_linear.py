import torch
import torch.nn as nn
from ...manifolds import PoincareBall
# from ...nn.conv import PoincareMLR

class PoincareLinear(nn.Module):
    """
    Poincare Fully Connected Linear Layer.

    Applies a hyperbolic linear transformation using Möbius matrix operations
    in the Poincare Ball model.

    Args:
        manifold (PoincareBall): Instance of the Poincare Ball manifold.
        c (float): Curvature of the Poincare Ball.
        in_features (int): Dimensionality of input features.
        out_features (int): Dimensionality of output features.
        use_bias (bool, optional): If True, includes a learnable bias. Default is True.
        id_init (bool, optional): If True, initialize weights as scaled identity. Default is True.

    Based on:
        - HNN++ (https://arxiv.org/abs/2006.08210)
    """

    def __init__(
        self,
        manifold: PoincareBall,
        c,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        id_init: bool = True,
    ) -> None:
        super(PoincareLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.has_bias = use_bias
        self.id_init = id_init
        self.c = c

        self.z = nn.Parameter(torch.empty(in_features, out_features))
        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        self.mlr = PoincareMLR(self.manifold, c)

    def reset_parameters(self) -> None:
        if self.id_init:
            self.z = nn.Parameter(
                1 / 2 * torch.eye(self.in_features, self.out_features)
            )
        else:
            nn.init.normal_(
                self.z, mean=0, std=(2 * self.in_features * self.out_features) ** -0.5
            )
        if self.has_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.manifold.expmap0(x, self.c, dim=-1)
        c_sqrt = self.c.sqrt()
        x = self.mlr(x, self.z, self.bias)
        x = (c_sqrt * x).sinh() / c_sqrt
        y = x / (1 + (1 + self.c * x.pow(2).sum(dim=-1, keepdim=True)).sqrt())
        return y
        # return self.manifold.logmap0(y, self.c, dim=-1)