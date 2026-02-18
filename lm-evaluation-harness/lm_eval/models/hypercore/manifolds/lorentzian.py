# The hyperboloid manifold Class
# 2019.12.20

import torch
from ..utils import arcosh, artanh, tanh
import numpy as np
from . import lmath as math
import geoopt
# from geoopt.utils import size2shape
from geoopt import Lorentz as LorentzOri
from typing import Tuple, Optional

def arcosh(x: torch.Tensor):
    z = torch.sqrt(torch.clamp_min(x.double().pow(2) - 1.0, 1e-6))
    return torch.log(x + z).to(x.dtype)

class Lorentz(LorentzOri):
    """
    Hyperboloid Manifold class.
    for x in (d+1)-dimension Euclidean space
    -x0^2 + x1^2 + x2^2 + … + xd = -c, x0 > 0, c > 0
    negative curvature - 1 / c
    """

    def __init__(self, c=1.0, learnable=False):
        super(Lorentz, self).__init__(c, learnable=learnable)
        self.max_norm = 50.0
        self.min_norm = 1e-6
        self.eps = {torch.float32: 1e-6, torch.float64: 1e-8}
        self.name = 'Lorentz'
        self.c = self.k

    def l_inner(self, x, y, keep_dim=False, dim=-1):
        return math._inner(x, y, keep_dim, dim)
    

    def sqdist(self, x, y, norm_control=True):
        return self.lorentzian_distance(x, y)

    def induced_distance(self, x, y):
        xy_inner = self.l_inner(x, y)
        sqrt_c = self.c ** 0.5
        return sqrt_c * arcosh((xy_inner / self.c).clamp_min(1 + self.eps[x.dtype]))

    def lorentzian_distance(self, x, y):
        # the squared Lorentzian distance
        xy_inner = self.l_inner(x, y)
        return -2 * (self.c + xy_inner)
    
    def proj(self, p, dim=-1):
        return self.projx(p, dim)

    def proj_tan_zero(self, u,):
        zeros = torch.zeros_like(u)
        # print(zeros)
        zeros[:, 0] = self.c ** 0.5
        return self.proju(zeros, u)

    def proj_tan0(self, u):
        return self.proj_tan_zero(u)
    

    def normalize_input(self, x):
        num_nodes = x.size(0)
        zeros = torch.zeros(num_nodes, 1, dtype=x.dtype, device=x.device)
        x_tan = torch.cat((zeros, x), dim=1)
        return self.expmap0(x_tan)
    
    def normalize_tan0(self, p_tan):
        zeros = torch.zeros_like(p_tan)
        zeros[:, 0] = self.c ** 0.5
        return self.proju(zeros, p_tan)

    def matvec_regular(self, m, x, b, use_bias):
        d = x.size(1) - 1
        x_tan = self.logmap0(x)
        x_head = x_tan.narrow(1, 0, 1)
        x_tail = x_tan.narrow(1, 1, d)
        mx = x_tail @ m.transpose(-1, -2)
        if use_bias:
            mx_b = mx + b
        else:
            mx_b = mx
        mx = torch.cat((x_head, mx_b), dim=1)
        mx = self.normalize_tan0(mx)
        mx = self.expmap0(mx)
        cond = (mx==0).prod(-1, keepdim=True, dtype=torch.uint8)
        res = torch.zeros(1, dtype=mx.dtype, device=mx.device)
        res = torch.where(cond, res, mx)
        return res

    def lorentzian_centroid(self, x, weight=None, dim=-1):
        if weight is not None:
            ave = weight @ (x)
        else:
            ave = x.mean(dim=-2)
        denom = (-self.l_inner(ave, ave, dim=dim, keep_dim=True)).abs().clamp_min(self.eps[x.dtype]).sqrt()
        return self.c.sqrt() * ave / denom

    def ptransp0(self, y, v):
        # y: target point
        zeros = torch.zeros_like(v)
        zeros[:, 0] = self.c ** 0.5
        v = self.normalize_tan0(v)
        return self.ptransp(zeros, y, v)
    
    def ptransp(self, x, y, v):
        # transport v from x to y
        K = 1. / self.c
        yv = self.l_inner(y, v, keep_dim=True)
        xy = self.l_inner(x, y, keep_dim=True)
        _frac = K * yv / (1 - K * xy).clamp_min(1e-6)
        return v + _frac * (x + y)
    
    def cinner(self, x: torch.Tensor, y: torch.Tensor):
        return math.cinner(x, y)
    
    def random_normal(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        r"""
        Create a point on the manifold, measure is induced by Normal distribution on the tangent space of zero.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device

        Returns
        -------
        ManifoldTensor
            random points on Hyperboloid

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        # self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.c.device:
            raise ValueError(
                "`device` does not match the projector `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.c.dtype:
            raise ValueError(
                "`dtype` does not match the projector `dtype`, set the `dtype` arguement to None"
            )
        tens = torch.randn(*size, device=self.c.device, dtype=self.c.dtype) * std + mean
        tens /= tens.norm(dim=-1, keepdim=True)
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

    def origin(
        self, *size, dtype=None, device=None, seed=42
    ) -> "geoopt.ManifoldTensor":
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
            zero point on the manifold
        """
        if dtype is None:
            dtype = self.c.dtype
        if device is None:
            device = self.c.device

        zero_point = torch.zeros(*size, dtype=dtype, device=device)
        zero_point[..., 0] = torch.sqrt(self.c)
        return geoopt.ManifoldTensor(zero_point, manifold=self)
    
    def mobius_add(self, x, y):
        u = self.logmap0(y)
        v = self.ptransp0(x, u)
        return self.expmap(v, x)

    def mobius_matvec(self, m, x):
        u = self.logmap0(x)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu)
    
    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1) -> Tuple[bool, Optional[str]]:
        """
        Check if a point lies on the manifold.

        Parameters:
            x (torch.Tensor): Point to check.
            atol (float): Absolute tolerance.
            rtol (float): Relative tolerance.
            dim (int): Dimension to check.

        Returns:
            Tuple[bool, Optional[str]]: A boolean indicating if the point is on the manifold, and an optional reason string.
        """
        dn = x.size(dim) - 1
        x = x ** 2
        quad_form = -x.narrow(dim, 0, 1) + x.narrow(dim, 1, dn).sum(dim=dim, keepdim=True)
        ok = torch.allclose(quad_form, -self.k, atol=atol, rtol=rtol)
        reason = None if ok else f"'x' minkowski quadratic form is not equal to {-self.k.item()}"
        return ok, reason
    
    def lorentz_to_poincare(self, x, dim=-1):
        dn = x.size(dim) - 1
        beta_sqrt = self.c.reciprocal().sqrt()
        x_space = x[..., 1:]
        x_space = beta_sqrt * x_space
        return x_space / (x.narrow(dim, 0, 1) + beta_sqrt)
    
    def poincare_to_lorentz(self, x, dim=-1, eps=1e-6):
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
            torch.cat((1/self.c + x_norm_square, 2/self.c * x), dim=dim)
            / (1/self.c - x_norm_square + eps)
        )
        return (self.c.reciprocal().sqrt()) * res

    def oxy_angle(self, x, y, eps: float = 1e-8):
        """
        Given two vectors `x` and `y` on the hyperboloid, compute the exterior
        angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
        of the hyperboloid.
        Args:
            x: Tensor of shape `(B, D)` vectors on the hyperboloid.
            y: Tensor of same shape as `x` giving another batch of vectors.
        Returns:
            Tensor of shape `(B, )`, angle in `(0, pi)`.
        """

        # Calculate time components of inputs (multiplied with `sqrt(curv)`):
        x_time = x[..., :1]
        y_time = y[..., :1]
        x_space = x[..., 1:]
        y_space = y[..., 1:]
        # Lorentz inner product, computed manually to save space since we only need the diagonal
        c_xyl = self.c * (torch.sum(x_space[..., 1:] * y_space[..., 1:], dim=-1) - x_time * y_time)

        # Make the numerator and denominator for input to arc-cosh, shape: (B, )
        acos_numer = y_time + c_xyl * x_time
        acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

        acos_input = acos_numer / (torch.norm(x_space[..., 1:], dim=-1) * acos_denom + eps)
        _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
        return _angle

    def half_aperture(self, x, min_radius: float = 0.1, eps: float = 1e-8):
        """
        Compute the half aperture angle of the entailment cone formed by vectors on
        the hyperboloid. 
        Args:
            x: Tensor of shape `(B, D)` giving vectors on the hyperboloid.
            min_radius: Radius of a small neighborhood around vertex of the hyperboloid
                where cone aperture is left undefined. Input vectors lying inside this
                neighborhood (having smaller norm) will be projected on the boundary.
            eps: For numerical stability
        Returns:
            Tensor of shape `(B, )` giving the half-aperture of entailment cones in `(0, pi/2)`.
        """

        # Ensure numerical stability in arc-sin by clamping input.
        asin_input = 2 * min_radius / (torch.norm(x[..., 1:], dim=-1) * self.c**0.5 + eps)
        _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))
        return _half_aperture