'''
Hyperbolic Layers that makes up hyperolic convolution

Based on: 
    - Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space (https://arxiv.org/abs/2407.01290)
    - Lorentzian Residual Neural Networks (https://arxiv.org/abs/2412.14695)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LorentzLayerNorm(nn.Module):
    """
    Implements hyperbolic layer normalization by appling standard LayerNorm to the spatial 
    part (excluding time coordinate) of a Lorentzian vector, then recomputes the time 
    component to satisfy the Lorentzian constraint.

    Args:
        manifold_in: Lorentz manifold object (input space).
        in_features (int): Dimensionality of spatial input features.
        manifold_out: Optional Lorentz manifold for projecting output.
    """
    def __init__(self, manifold_in, in_features, manifold_out=None):
        super(LorentzLayerNorm, self).__init__()
        self.in_features = in_features
        self.manifold = manifold_in
        self.c = manifold_in.c
        self.manifold_out = manifold_out
        self.layer = nn.LayerNorm(self.in_features)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, space_only=False, return_space=False):
        """
        Forward pass of LorentzLayerNorm.

        Args:
            x (torch.Tensor): Input tensor with Lorentzian coordinates [B, ..., D+1].
            space_only (bool): If true, the input is only the space-like dimension of the Lorentz vector
            return_space (bool): If true, returns only the space-like dimension of the results to save computation

        Returns:
            torch.Tensor: Normalized tensor with updated time component.
        """
        if space_only:
            x_space = x
        else:
            x_space = x[..., 1:]
        x_space = self.layer(x_space)
        if return_space:
            x = x_space
        else:
            x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-4).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()
        return x

class LorentzNormalization(nn.Module):
    """
    Normalizes spatial components to unit norm and recomputes time component
    to satisfy Lorentz geometry constraints.

    Args:
        manifold_in: Lorentz manifold object (input space).
        manifold_out: Optional target manifold for output projection.
        
    """
    def __init__(self, manifold_in, manifold_out=None, return_space=False):
        super(LorentzNormalization, self).__init__()
        self.manifold = manifold_in
        self.manifold_out = manifold_out
        self.c = manifold_in.c

    def forward(self, x, norm_factor=None, space_only=False, return_space=False):
        """
        Forward pass of LorentzNormalization.

        Args:
            x (torch.Tensor): Input tensor with Lorentzian coordinates.
            norm_factor (torch.Tensor, optional): Precomputed normalization factors.
            space_only (bool): If true, the input is only the space-like dimension of the Lorentz vector
            return_space (bool): If true, returns only the space-like dimension of the results to save computation

        Returns:
            torch.Tensor: Lorentz-normalized tensor.
        """
        if space_only:
            x_space = x
        else:
            x_space = x[..., 1:]
        if norm_factor is not None:
            x_space = x_space * norm_factor
        else:
            x_space = x_space / x_space.norm(dim=-1, keepdim=True)
        if return_space:
            x = x_space
        else:
            x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-4).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()
        return x

class LorentzActivation(nn.Module):
    """
    Applies a nonlinear activation to the spatial part of a Lorentzian vector,
    followed by recomputing the time component.

    Args:
        manifold_in: Input Lorentz manifold.
        activation (Callable): Activation function (e.g., nn.ReLU()).
        manifold_out: Optional output Lorentz manifold.
    """
    def __init__(self, manifold_in, activation, manifold_out=None):
        super(LorentzActivation, self).__init__()
        self.manifold = manifold_in
        self.manifold_out = manifold_out
        self.activation = activation
        self.c = manifold_in.c

    def forward(self, x, space_only=False, return_space=False):
        """
        Applies the activation and recomputes time.

        Args:
            x (torch.Tensor): Input tensor in Lorentz coordinates.
            space_only (bool): If true, the input is only the space-like dimension of the Lorentz vector
            return_space (bool): If true, returns only the space-like dimension of the results to save computation

        Returns:
            torch.Tensor: Activated Lorentz vector.
        """
        if space_only:
            x_space = x
        else:
            x_space = x[..., 1:]
        x_space = self.activation(x_space)
        if return_space:
            x = x_space
        else:    
            x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-4).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
                x = x * (self.manifold_out.c / self.c).sqrt()
        return x

class LorentzDropout(nn.Module):
    """
    Applies dropout to spatial coordinates of a Lorentzian vector and updates
    time coordinate accordingly.

    Args:
        manifold_in: Input Lorentz manifold.
        dropout (float): Dropout probability.
        manifold_out: Optional output manifold for projection.
    """
    def __init__(self, manifold_in, dropout, manifold_out=None):
        super(LorentzDropout, self).__init__()
        self.manifold = manifold_in
        self.manifold_out = manifold_out
        self.dropout = nn.Dropout(dropout)
        self.c = manifold_in.c

    def forward(self, x, space_only=False, return_space=False):
        """
        Forward pass of LorentzDropout.

        Args:
            x (torch.Tensor): Input Lorentz tensor.
            training (bool): If True, apply dropout.
            space_only (bool): If true, the input is only the space-like dimension of the Lorentz vector
            return_space (bool): If true, returns only the space-like dimension of the results to save computation

        Returns:
            torch.Tensor: Tensor after dropout with corrected time.
        """
        if self.training:
            if space_only:
                x_space = x
            else:
                x_space = x[..., 1:]
            x_space = self.dropout(x_space)
            if return_space:
                x = x_space
            else:
                x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-4).sqrt()
                x = torch.cat([x_time, x_space], dim=-1)
            if self.manifold_out is not None:
                x = x * (self.manifold_out.c / self.c).sqrt()
        return x
    
class LResNet(nn.Module):
    """
    Residual block in Lorentz space with optional learnable scaling.

    Args:
        manifold_in: Input manifold.
        weight (Tensor or None): Initial weight tensor (optional).
        batch_size (int or None): Batch size for per-sample weights.
        use_scale (bool): Whether to scale spatial output.
        scale (float or None): Fixed or use learnable scaling when None.
        manifold_out: Optional target manifold for output projection.
    """
    def __init__(self, manifold_in, weight=None, batch_size=None, use_scale=False, scale=None, learn_scale=False, manifold_out=None):
        super(LResNet, self).__init__()
        self.manifold = manifold_in
        if weight is not None:
            self.w_y = weight
        else:
            # using learnable weights
            if batch_size:
                # use separate weight for each vector
                self.w_y = nn.Parameter(torch.ones((batch_size, 1)))
            else:
                self.w_y = nn.Parameter(torch.tensor(1.0))
        self.scale = None
        if use_scale:
            if scale:
                if learn_scale:
                    self.scale = nn.Parameter(torch.tensor(math.log(scale)))
                    self.learned_scale = True
                else:
                    self.scale = scale
                    self.learned_scale = False
            else:
                self.scale = nn.Parameter(torch.tensor(4.0))
                self.learned_scale = True
        self.c = manifold_in.c
        self.manifold_out = manifold_out

    def forward(self, x, y, weight=None):
        """
        Forward pass for LResNet residual block.

        Args:
            x, y (torch.Tensor): Lorentzian vectors.
            weight: If not None, overwrites the weight
        Returns:
            torch.Tensor: Resulting Lorentzian residual.
        """
        if weight is not None:
            w_y = weight
        else:
            w_y = self.w_y
        ave = x + y * w_y
        denom = (-self.manifold.l_inner(ave, ave, dim=-1, keep_dim=True)).abs().clamp_min(1e-4).sqrt()
        x = self.c.sqrt() * ave / denom
        if self.scale:
            if self.learned_scale:
                x_space = self.scale.exp() * x[..., 1:]
            else:
                x_space = self.scale * x[..., 1:]
            x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-4).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
        
        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()

        return x

class LorentzRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization in Lorentz geometry, implemented by us

    Args:
        manifold_in: Input manifold.
        dim (int): Dimensionality of spatial vector.
        eps (float): Small value for numerical stability.
        manifold_out: Optional output manifold.
    """
    def __init__(self, manifold_in, dim: int, eps: float = 1e-8, manifold_out=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.c = manifold_in.c

    def forward(self, x: torch.Tensor, space_only=False, return_space=False):
        """
        Forward pass for LorentzRMSNorm.

        Args:
            x (torch.Tensor): Input Lorentz tensor.
            space_only (bool): If true, the input is only the space-like dimension of the Lorentz vector
            return_space (bool): If true, returns only the space-like dimension of the results to save computation

        Returns:
            torch.Tensor: RMS-normalized tensor in Lorentz space.
        """
        if space_only:
            x_space = x
        else:
            x_space = x[..., 1:]
        normed_space = F.rms_norm(x_space, (self.dim,), self.weight, self.eps)
        if return_space:
            x = normed_space
        else:
            x_time = ((normed_space ** 2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-4).sqrt()
            x = torch.cat([x_time, normed_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()
        return x