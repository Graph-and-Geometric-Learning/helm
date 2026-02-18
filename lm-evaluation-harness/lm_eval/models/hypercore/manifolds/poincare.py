"""Poincare ball manifold."""

import torch

from ..utils import artanh, tanh
from typing import Optional
from geoopt import Stereographic

class PoincareBall(Stereographic):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """
    @property
    def k(self):
        return -self.c
    def __init__(self, c=1.0, learnable=False):
        super(PoincareBall, self).__init__(-c, learnable=False)
        # self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        # self.k = torch.nn.Parameter(torch.tensor([-c]), requires_grad=learnable)
        self.c = torch.nn.Parameter(torch.tensor([c]), requires_grad=learnable)
        self.curvature = self.c

    def sqdist(self, p1, p2, dim=-1):
        c = self.curvature
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2,dim=dim).norm(dim=dim, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2
    
    # def dist(self, p1, p2):
    #     c = self.curvature
    #     sqrt_c = c ** 0.5
    #     dist_c = 2.0 * artanh(
    #         sqrt_c * self.mobius_add(-p1, p2, dim=-1).norm(dim=-1, p=2, keepdim=False)
    #     )
    #     dist = dist_c / sqrt_c
    #     return dist
    
    # def dist2(self, p1, p2):
    #     return self.sqdist(p1, p2)
    
    def _lambda_x(self, x, dim=-1, keepdim=True):
        c = self.curvature
        x_sqnorm = torch.sum(x.data.pow(2), dim=dim, keepdim=keepdim)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp):
        c = self.curvature
        lambda_p = self._lambda_x(p)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, dim=-1):
        c = self.curvature
        norm = torch.clamp_min(x.norm(dim=dim, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
    
    # def projx(self, x, dim=-1):
    #     return self.proj(x, dim=dim)

    def proj_tan(self, u, p):
        return u
    
    def proju(self, u, p):
        return u

    def proj_tan0(self, u):
        return u

    # def expmap(self, u, p, dim=-1, project=True):
    #     c = self.curvature
    #     sqrt_c = c ** 0.5
    #     u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(self.min_norm)
    #     second_term = (
    #             tanh(sqrt_c / 2 * self._lambda_x(p) * u_norm)
    #             * u
    #             / (sqrt_c * u_norm)
    #     )
    #     gamma_1 = self.mobius_add(p, second_term)
    #     if project:
    #         gamma_1 = self.proj(gamma_1)
    #     return gamma_1

    # def logmap(self, p1, p2, dim=-1, project=True):
    #     c = self.curvature
    #     sub = self.mobius_add(-p1, p2)
    #     sub_norm = sub.norm(dim=dim, p=2, keepdim=True).clamp_min(self.min_norm)
    #     lam = self._lambda_x(p1)
    #     sqrt_c = c ** 0.5
    #     ret = 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm
    #     return ret

    # def expmap0(self, u, dim=-1, project=True):
    #     c = self.curvature
    #     sqrt_c = c ** 0.5
    #     u_norm = torch.clamp_min(u.norm(dim=dim, p=2, keepdim=True), self.min_norm)
    #     gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    #     if project:
    #         gamma_1 = self.proj(gamma_1)
    #     return gamma_1

    # def logmap0(self, p, dim=-1):
    #     c = self.curvature
    #     sqrt_c = c ** 0.5
    #     p_norm = p.norm(dim=dim, p=2, keepdim=True).clamp_min(self.min_norm)
    #     scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
    #     return scale * p

    # def mobius_add(self, x, y, dim=-1):
    #     c = self.curvature
    #     x2 = x.pow(2).sum(dim=dim, keepdim=True)
    #     y2 = y.pow(2).sum(dim=dim, keepdim=True)
    #     xy = (x * y).sum(dim=dim, keepdim=True)
    #     num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    #     denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    #     return num / denom.clamp_min(self.min_norm)
    
    # def mobius_scalar_mul(self, r, x, dim=-1):
    #     x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    #     res_c = (1 / self.curvature.sqrt()) *  tanh(r * artanh(x_norm * self.curvature.sqrt())) * (x / x_norm)
    #     return res_c

    # def mobius_matvec(self, m, x):
    #     c = self.curvature
    #     sqrt_c = c ** 0.5
    #     x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
    #     mx = x @ m.transpose(-1, -2)
    #     mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
    #     res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    #     cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    #     res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    #     res = torch.where(cond, res_0, res_c)
    #     return res

    def init_weights(self, w, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, dim: int = -1):
        c = self.curvature
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, u, v=None, keepdim=False):
        c = self.curvature
        if v is None:
            v = u
        lambda_x = self._lambda_x(x)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, dim=-1):
        c = self.curvature
        lambda_x = self._lambda_x(x, dim=dim)
        lambda_y = self._lambda_x(y, dim=dim)
        return self._gyration(y, -x, u, dim=dim) * lambda_x / lambda_y

    def transp(self, x, y, u, dim=-1):
        return self.ptransp(x, y, u, dim=dim)

    def ptransp_(self, x, y, u):
        c = self.curvature
        lambda_x = self._lambda_x(x)
        lambda_y = self._lambda_x(y)
        return self._gyration(y, -x, u) * lambda_x / lambda_y

    def ptransp0(self, x, u):
        c = self.curvature
        lambda_x = self._lambda_x(x)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x):
        c = self.curvature
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

    def poincare_midpoint(
        self,
        x: torch.Tensor,
        vec_dim: int = -1,
        batch_dim: int = 0,
    ):  
        c = self.curvature
        gamma_sq = 1 / (1 - c * x.pow(2).sum(dim=vec_dim, keepdim=True)).clamp_min(1e-15)
        numerator = (gamma_sq * x).sum(dim=batch_dim, keepdim=True)
        denominator = gamma_sq.sum(dim=batch_dim, keepdim=True) - x.size(batch_dim) / 2
        m = numerator / denominator
        gamma_m = 1 / (1 - c * m.pow(2).sum(dim=vec_dim, keepdim=True)).sqrt().clamp_min(
            1e-15
        )
        return gamma_m / (1 + gamma_m) * m
    
    def frechet_variance(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        dim: int = -1,
        w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = self.curvature
        distance: torch.Tensor = self.sqdist(x, mu, dim=dim)

        if w is None:
            return distance.mean(dim=dim)
        else:
            return (distance * w).sum(dim=dim)
    def weighted_midpoint_bmm(
        self,
        xs: torch.Tensor,
        weights: torch.Tensor,
        lincomb: bool = False,
        dim=-1
    ):
        c = self.curvature
        gamma = self._lambda_x(xs, dim=dim, keepdim=True)
        denominator = torch.matmul(weights.abs(), gamma - 1)
        nominator = torch.matmul(weights, gamma * xs)
        two_mean = nominator / denominator.clamp_min(1e-10) ## instead of clamp_abs
        a_mean = two_mean / (1. + (1. + c * two_mean.pow(2).sum(dim=-1, keepdim=True)).sqrt())

        if lincomb:
            alpha = weights.abs().sum(dim=dim, keepdim=True)
            a_mean = self.mobius_scalar_mul(alpha, a_mean, dim=dim)
        return self.proj(a_mean, dim=dim)