import torch
import torch.nn as nn

from ...manifolds import PoincareBall

class PoincareBatchNorm(nn.Module):
    """
    Basic implementation of batch normalization in the Poincare ball model.

    Based on:
        - Differentiating through the Fréchet Mean (https://arxiv.org/abs/2003.00335)
        - Poincare ResNet (https://arxiv.org/abs/2303.14027)

    Args:
        manifold (PoincareBall): Poincare Ball manifold instance.
        c (float): Curvature of the Poincare Ball.
        features (int): Number of features (excluding batch dimension).
    """

    def __init__(
        self,
        manifold: PoincareBall,
        c,
        features: int,
    ) -> None:
        super(PoincareBatchNorm, self).__init__()
        self.features = features
        self.manifold = manifold
        self.c = c
        self.mean = nn.Parameter(torch.zeros(features))
        self.var = nn.Parameter(torch.tensor(1.0))

        # statistics
        self.register_buffer("running_mean", torch.zeros(1, features))
        self.register_buffer("running_var", torch.tensor(1.0))
        self.updates = 0

    def forward(self, x, momentum=0.9):
        # x = self.manifold.expmap0(x, self.c, dim=-1)
        mean_on_ball = self.manifold.expmap0(self.mean, self.c, dim=-1)
        input_mean = self.manifold.poincare_midpoint(x, self.c, vec_dim=-1, batch_dim=0)
        input_var = self.manifold.frechet_variance(x, input_mean, self.c, dim=-1)

        input_logm = self.manifold.ptransp(
            x=input_mean,
            y=mean_on_ball,
            u=self.manifold.logmap(input_mean, x, self.c),
            c=self.c
        )

        input_logm = (self.var / (input_var + 1e-6)).sqrt() * input_logm

        output = self.manifold.expmap(mean_on_ball.unsqueeze(-2), input_logm, self.c)

        self.updates += 1

        return output
        # return self.manifold.logmap0(output, self.c, dim=-1)


class PoincareBatchNorm2d(nn.Module):
    """
    2D Batch Normalization on the Poincare Ball Model.

    Based on:
        - Differentiating through the Fréchet Mean (https://arxiv.org/abs/2003.00335)
        - Poincare ResNet (https://arxiv.org/abs/2303.14027)

    Args:
        manifold (PoincareBall): Poincare Ball manifold instance.
        c (float): Curvature of the Poincare Ball.
        features (int): Number of channels/features.
    """

    def __init__(
        self,
        manifold: PoincareBall,
        c,
        features: int,
    ) -> None:
        super(PoincareBatchNorm2d, self).__init__()
        self.manifold = features
        self.ball = manifold
        self.c = c
        self.norm = PoincareBatchNorm(
            self.manifold,
            self.c,
            features=features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input dimensions
        batch_size, height, width = x.size(0), x.size(2), x.size(3)

        # Swap batch and channel dimensions and flatten everything but channel dimension
        x = x.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)

        # Apply batchnorm
        x = self.norm(x)

        # Reshape to original dimensions
        x = x.reshape(batch_size, height, width, self.features).permute(0, 3, 1, 2)

        return x