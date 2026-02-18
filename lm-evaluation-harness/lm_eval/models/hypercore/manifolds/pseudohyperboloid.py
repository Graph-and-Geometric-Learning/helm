"""Hyperboloid manifold."""

import torch

from .base import Manifold
from ..utils import arcosh, cosh, sinh 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math


class PseudoHyperboloid(Manifold):
    
    def __init__(self,space_dim=15, time_dim=1, beta=-1):
        super(PseudoHyperboloid, self).__init__()
        self.name = 'PseudoHyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6
        self.dim = space_dim + time_dim
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.dim = space_dim + time_dim
        # self.hyperboloid = hp
       
        
    def _check_point_on_manifold(self, x, rtol=1e-08, atol=1e-04):
        inner = self.inner(x, x)
        ok = torch.allclose(inner, inner.new((1,)).fill_(self.beta.item()), atol=atol, rtol=rtol)
        if not ok:
            return False
        return True

    def _check_vector_on_tangent(self, x, u, rtol=1e-08, atol=1e-04):
        inner = self.inner(x, u)
        ok = torch.allclose(inner, inner.new_zeros((1,)), atol=atol, rtol=rtol)
        if not ok:
            return False
        return True

    def _check_vector_on_tangent0(self, x, atol=1e-5, rtol=1e-5):
        origin = x.clone()
        origin[:,:] = 0
        origin[:,0] = self.beta.abs()**0.5
        inner = self.inner(origin, x)
        ok = torch.allclose(inner, inner.new_zeros((1,)), atol=atol, rtol=rtol)
        if not ok:
            return False
        return True
    
    def inner(self, x, y, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        z = x * y
        res = torch.clamp(z[:,time_dim:].sum(dim=-1) - z[:, 0:time_dim].sum(dim=-1), max=self.max_norm)
        return res

    def initialization(self,x,beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        # self.beta = self.beta.cuda().to(x.get_device())
        inner = self.inner(x, x, time_dim=time_dim)
        initial_embedding = x
        epsilon = 0.00000000000
        positive = inner <= epsilon
        negative = inner > epsilon
        assert not True in negative
        initial_embedding = beta.abs().sqrt()*x/(inner.abs().sqrt()).unsqueeze(1)
        return initial_embedding


    # def sqdist(self, x, y, beta, time_dim=None):
    #     time_dim = self.time_dim if time_dim==None else time_dim
    #     # self.beta = self.beta.cuda().to(y.get_device())
    #     inner = self.inner(x, y, time_dim=time_dim)
    #     epsilon = 0.000001
    #     # print(beta,x.max(),x.min())
    #     K = inner/beta.abs()
    #     c1 = K < -1.0 - epsilon
    #     c2 = (K <= -1.0 + epsilon) & (K >= -1.0 - epsilon)
    #     c3 = (K > -1.0 + epsilon) & (K < 1)
    #     c4 = K >= 1
    #     other = (~c1) & (~c2) & (~c3) & (~c4)
    #     dist2 = x[:,0].clone()
    #     assert dist2[other].shape[0] == 0
        
    #     device = y.get_device()
    #     if True in c1:
    #         # print('dist:hyperbolic_like')
    #         dist2[c1] = (beta.abs()**0.5) * torch.clamp(torch.acosh(inner[c1]/beta),min=self.min_norm, max=self.max_norm)
    #     if True in c2:
    #         # print('dist:Euclidean_like')
    #         dist2[c2] = 0
    #     if True in c3:
    #         # print('dist:shperical_like')
    #         dist2[c3] = (beta.abs()**0.5) * torch.clamp(torch.acos(inner[c3]/beta),min=self.min_norm, max=self.max_norm)
    #     if True in c4:
    #         # print('dist:positive_like')
    #         # if device != -1:
    #         #     dist2[c4] = (abs(self.beta)**0.5) * torch.clamp((torch.Tensor([math.pi]).cuda().to(device)/2 + inner[c4]/abs(self.beta)),min=self.min_norm, max=self.max_norm)
    #         # else:
    #         #     dist2[c4] = (abs(self.beta)**0.5) * torch.clamp((torch.Tensor([math.pi])/2 + inner[c4]/abs(self.beta)),min=self.min_norm, max=self.max_norm)
    #         if device != -1:
    #             dist2[c4] = (beta.abs()**0.5) * (torch.Tensor([math.pi]).cuda().to(device)) + self.sqdist(-x[c4], y[c4], beta)
    #         else:
    #             dist2[c4] = (beta.abs()**0.5) * torch.Tensor([math.pi]) + self.sqdist(-x[c4], y[c4], beta)
    #     return torch.clamp(dist2, min=0.00001, max=50.0)
    def sqdist(self, x, y, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        # self.beta = self.beta.cuda().to(y.get_device())
        inner = self.inner(x, y, time_dim=time_dim)
        epsilon = 0.000001
        # print(beta,x.max(),x.min())
        K = inner/beta.abs()
        c1 = K < -1.0 - epsilon
        c2 = (K <= -1.0 + epsilon) & (K >= -1.0 - epsilon)
        c3 = (K > -1.0 + epsilon) & (K < 1 )
        # c5 = (K>= 1 - epsilon) & (K <= 1 + epsilon)
        c4 = K >= 1 
        
        other = (~c1) & (~c2) & (~c3) & (~c4) 
        dist2 = x[:,0].clone()
        assert dist2[other].shape[0] == 0
        
        device = y.get_device()
        if True in c1:
            # print('dist:hyperbolic_like')
            u = self.logmap_n(x[c1],y[c1],beta)
            d = u.norm(dim=1,p=2)
            dist2[c1] = torch.clamp((beta.abs()**0.5) * d,max=self.max_norm)
            # dist2[c1] = (beta.abs()**0.5) * torch.clamp(torch.acosh(inner[c1]/beta),min=self.min_norm, max=self.max_norm) 
        if True in c2:
            # print('dist:Euclidean_like')
            # u = self.logmap_n(x[c2],y[c2])
            # d = u.norm(dim=1,p=2)
            # dist2[c1] = (beta.abs()**0.5) * d
            u = y[c2] - x[c2]
            d = u.norm(dim=1,p=2)
            dist2[c2] = torch.clamp((beta.abs()**0.5)*d,max=self.max_norm)
        if True in c3:
            # print('dist:shperical_like')
            u = self.logmap_n(x[c3],y[c3],beta)
            d = u.norm(dim=1,p=2)
            dist2[c3] = torch.clamp((beta.abs()**0.5)*d, max=self.max_norm)
            # dist2[c3] = (beta.abs()**0.5) * torch.clamp(torch.acos(inner[c3]/beta),min=self.min_norm, max=self.max_norm)
        if True in c4:
            #  d = torch.min(self.sqdist(x[c4], -x[c4], beta)+ self.sqdist(-x[c4], y[c4], beta), self.sqdist(y[c4], -y[c4], beta)+self.sqdist(x[c4], -y[c4], beta))
            d = torch.min(self.cycle_dist(x[c4], beta) + self.sqdist(-x[c4], y[c4], beta), self.cycle_dist(y[c4],beta) + self.sqdist(x[c4], -y[c4], beta))
            dist2[c4] = torch.clamp(d,max=self.max_norm)
            # if device != -1:
            #     dist2[c4] = (beta.abs()**0.5) * (torch.Tensor([math.pi]).cuda().to(device)) + self.sqdist(-x[c4], y[c4], beta)
            # else:
            #     dist2[c4] = (beta.abs()**0.5) * torch.Tensor([math.pi]) + self.sqdist(-x[c4], y[c4], beta)
        # if True in c5:
        #     dist2[c5] = self.cycle_dist()

        return torch.clamp(dist2, min=0.00001, max=50.0)

    def cycle_dist(self,x,beta):
        return (beta.abs()**0.5)*math.pi * x.norm(dim=1,p=2)

    def expmap(self, x, v, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        epsilon = 0.000001
        n = v.shape[0]
        d = v.shape[1]
        inner = self.inner(v, v, time_dim=time_dim)
        norm_product = torch.clamp(inner.abs(),min=self.min_norm).sqrt()
        norm_product = torch.clamp(norm_product, max=50).view(norm_product.size(0),-1)

        space_like = inner < -epsilon
        time_like = inner > epsilon
        null_geodesic = (~space_like) & (~time_like)
        other = (~time_like) & (~space_like) & (~null_geodesic)
        U = v.clone()
        # print(beta)
        abs_beta = 1/(abs(beta) ** 0.5)
        if True in time_like:
            # print('exp:hyperbolic_like')
            beta_product = torch.clamp(abs_beta*norm_product[time_like],max=self.max_norm)
            # print(norm_product[time_like].max().item(), norm_product[time_like].min().item())
            U[time_like,:] = x[time_like,:]*torch.clamp(torch.cosh(beta_product),max=self.max_norm) +  torch.clamp( torch.clamp(v[time_like,:]*torch.sinh(beta_product), max=self.max_norm)/beta_product,  max=self.max_norm)
            # print(beta_product.max().item())
            assert not torch.isnan( U[time_like,:]).any()
        if True in space_like:
            # print('exp:spherical_like')
            beta_product = torch.clamp(abs_beta*norm_product[space_like],max=self.max_norm)
            U[space_like,:] = x[space_like,:]*torch.clamp(torch.cos(beta_product),max=self.max_norm) +  torch.clamp(torch.clamp(v[space_like,:]*torch.sin(beta_product), max=self.max_norm)/beta_product,  max=self.max_norm)
            assert not torch.isnan(  U[space_like,:]).any()
            # U[space_like,:] = sp.expmap(x[space_like,:], v[space_like,:])
        if True in null_geodesic:
            # print('exp:null_like')
            U[null_geodesic,:] = torch.clamp(x[null_geodesic,:] + v[null_geodesic,:], max=self.max_norm)
            assert not torch.isnan(v[null_geodesic,:] ).any()
        assert not torch.isnan(U).any()
        return self.proj(U,beta)

    def expmap0(self,v, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        # self.beta = self.beta.cuda().to(v.get_device())
        origin = v.clone()
        origin[:,:] = 0
        origin[:,0] = abs(beta)**0.5
        return self.expmap(origin, v, beta, time_dim=time_dim)

    def logmap_n(self, x, y, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        # self.beta = self.beta.cuda().to(y.get_device())
        d = x.shape[1]
        n = x.shape[0]
        inner_positive = self.inner(x,y, time_dim=time_dim)
        inner_positive = torch.clamp(inner_positive, max=self.max_norm)
        abs_beta = abs(beta)
        epsilon = 0.00000001
        time_like_positive = inner_positive/abs_beta < -1 - epsilon
        null_geodesic_positive = (inner_positive/abs_beta>= -1 - epsilon) & (inner_positive/abs_beta<= -1 + epsilon)
        space_like_positive = (inner_positive/abs_beta > -1 + epsilon) & (inner_positive/abs_beta < 1)
        other = (~time_like_positive) & (~null_geodesic_positive) & (~space_like_positive)
                
        U = y.clone()
        # assert U[other].shape[0] == 0
        U[other,:] = 0
        beta_product_positive = (inner_positive/beta).view(inner_positive.size(0), -1)
        # assert not torch.isnan(beta_product_positive).any()
        abs_da = torch.clamp((beta_product_positive**2 - 1).abs(), min=self.min_norm)
        sqrt_minus_positive = (abs_da** 0.5).view(beta_product_positive.size(0), -1)
        if True in space_like_positive:
            # print('log:spherical_like')
            up = torch.clamp(torch.acos(beta_product_positive[space_like_positive]), min=self.min_norm, max=self.max_norm)
            low = torch.clamp(sqrt_minus_positive[space_like_positive], min=self.min_norm, max=self.max_norm)
            U[space_like_positive,:] = ((up/low).repeat(1,d))* torch.clamp((y[space_like_positive,:]-x[space_like_positive,:]*beta_product_positive[space_like_positive].repeat(1,d)),max=self.max_norm)
            assert not torch.isnan(U[space_like_positive,:]).any()
        if True in time_like_positive:
            # print('log:hyperbolic_like')
            up = torch.acosh(torch.clamp(beta_product_positive[time_like_positive], min=self.min_norm, max=self.max_norm))
            low = torch.clamp(sqrt_minus_positive[time_like_positive], min=self.min_norm, max=self.max_norm)
            U[time_like_positive,:] = ((up/low).repeat(1,d))*torch.clamp( (y[time_like_positive,:]-x[time_like_positive,:]*beta_product_positive[time_like_positive].repeat(1,d)),max=self.max_norm)
            assert not torch.isnan(U[time_like_positive,:]).any()

        if True in null_geodesic_positive:
            # print('log:null_like')
            U[null_geodesic_positive,:] = y[null_geodesic_positive,:] - x[null_geodesic_positive,:]
            assert not torch.isnan(U[null_geodesic_positive,:]).any()
        assert not torch.isnan(U).any()
        return U

    def logmap(self,x, y, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_positive = self.inner(x, y, time_dim=time_dim)
        epsilon = 0.00001
        positive_log_map = inner_positive < abs(beta) - epsilon
        negative_log_map = inner_positive >= abs(beta) + epsilon
        neutral = (~positive_log_map) & (~negative_log_map)
        U = y.clone()
        other = (~positive_log_map) & (~negative_log_map) & (~neutral)
        assert U[other].shape[0] == 0
        if True in positive_log_map:
            U[positive_log_map] = self.logmap_n(x[positive_log_map], y[positive_log_map], beta, time_dim=time_dim)
            # U[positive_log_map][:,0] = y[positive_log_map][:,0]
        if True in negative_log_map:
            print("negative_log_map")
        U[neutral] = y[neutral] - x[neutral]
        U = self.proj_tan(U, x, beta)
        return U
    

    def logmap0(self,y, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        # self.beta = self.beta.cuda().to(y.get_device())
        origin = y.clone()
        origin[:,:] = 0
        origin[:,0] = abs(beta)**0.5
        return self.logmap(origin,y, beta,time_dim=time_dim)

    def proj(self, x, beta,time_dim=None):
        time_dim = self.time_dim 
        # self.beta = self.beta.cuda().to(x.get_device())
        if time_dim == self.dim:
            U =  F.normalize(x)
            return U
        Xtime = torch.clamp(F.normalize(x[:,0:time_dim]),max=self.max_norm)
        Xspace = torch.clamp(x[:,time_dim:].div(beta.abs().sqrt()),max=self.max_norm)
        spaceNorm = torch.clamp(torch.sum(Xspace*Xspace, dim=1, keepdim=True),max=self.max_norm)
        if self.time_dim == 1:
            Xtime = torch.sqrt((spaceNorm).add(1.0)).view(-1,1)
        else:
            Xtime = torch.clamp(torch.clamp(torch.sqrt(spaceNorm.add(1.0)),max=self.max_norm).expand_as(Xtime) * Xtime, max=self.max_norm)
        U =  torch.cat((Xtime,Xspace),1)
        return U

    def proj_tan(self,z, x,beta, time_dim=None):
        # print(x)
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_zx = self.inner(z,x,time_dim=time_dim)
        inner_xx = self.inner(x,x,time_dim=time_dim)
        res = z - (inner_zx/inner_xx).unsqueeze(1)*x
        return res

    def proj_tan0(self, z, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        origin = z.clone()
        origin[:,:] = 0
        origin[:,0] = abs(beta)**0.5
        return self.proj_tan(z, origin, beta, time_dim=time_dim)
    
    def perform_rescaling_beta(self, X, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        norm_X = X * X
        norm_Xtime = norm_X[:,0:time_dim]
        norm_Xspace = norm_X[:,time_dim:]
        res = X / torch.abs( torch.sum(norm_Xspace,dim=1, keepdim=True) - torch.sum(norm_Xtime,dim=1, keepdim=True) ).sqrt().expand_as(X) * beta.abs().sqrt()
        return res

    def mobius_matvec(self, m, x, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        u = self.logmap0(x, beta, time_dim=time_dim)
        mu = u @ m.transpose(-1, -2)
        # mu = F.normalize(mu)
        mu = self.expmap0(mu, beta, time_dim=self.time_dim)
        return mu

    def mobius_add(self, x, y, beta):
        u = self.logmap0(y,beta)
        u = u.repeat(x.shape[0],1)
        v = self.ptransp0(x,u,beta)
        # v = F.normalize(v)
        # print(beta, v.max())
        return self.expmap(x,v,beta)
        
    # def ptransp0(self, x, u, c):
    #     c = c.abs()
    #     K = 1. / c
    #     sqrtK = K ** 0.5
    #     x0 = x.narrow(-1, 0, 1)
    #     d = x.size(-1) - 1
    #     y = x.narrow(-1, 1, d)
    #     y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
    #     y_normalized = y / y_norm
    #     v = torch.ones_like(x)
    #     v[:, 0:1] = - y_norm 
    #     v[:, 1:] = (sqrtK - x0) * y_normalized
    #     alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
    #     res = u - alpha * v
    #     return self.proj_tan(res, x, c)

    def ptransp0(self,x,u,beta):
        origin = x.clone()
        origin[:,:] = 0
        origin[:,0] = abs(beta)**0.5
        return self.ptransp(origin, x, u,beta)

    def ptransp(self,x,y,u, beta,time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_positive = self.inner(x, y, time_dim=time_dim)
        epsilon = 0.0000001
        p = inner_positive < abs(beta) - epsilon
        n = inner_positive > abs(beta) + epsilon
        neutral = (~p) & (~n)
        U = u.clone()
        U[neutral] = u[neutral]
        if True in p:
            U[p] = self.ptransp_n(x[p], y[p], u[p], beta, time_dim=time_dim)
        if True in n:
            print("pt.negative", inner_positive[n].min().item())
            negative_trans = self.ptransp_n(x[n], -y[n], u[n], beta, time_dim=time_dim)
            U[n] = self.ptransp_n(-y[n], y[n], beta, negative_trans) 
        return U
    
    def ptransp_n(self,x,y,u,beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_xy = self.inner(x,y,time_dim=time_dim)
        log_xy = self.logmap_n(x, y, beta, time_dim=time_dim) 
        inner_log_xy = self.inner(log_xy, log_xy, time_dim=time_dim)
        inner = self.inner(u, log_xy, time_dim=time_dim)
        inner_yu = self.inner(y,u,time_dim=time_dim)
        dist = torch.clamp(self.sqdist(x,y,beta,time_dim=time_dim), min=1e-5, max=self.max_norm)
        epsilon = 0.000001
        time_like = inner_log_xy > epsilon
        space_like = inner_log_xy < -epsilon
        null_like = (~time_like) & (~space_like)
        other = (~time_like) & (~space_like) & (~null_like)
        U = u.clone()
        assert U[other].shape[0] == 0
        if True in time_like:
            # print('pt.time_like')
            # U[time_like,:] = torch.clamp((inner[time_like]/dist[time_like]).unsqueeze(1) * (x[time_like]*torch.sinh(dist[time_like]).unsqueeze(1) + (log_xy[time_like]/dist[time_like].unsqueeze(1))*torch.cosh(dist[time_like]).unsqueeze(1)) + (u[time_like] - inner[time_like].unsqueeze(1)*log_xy[time_like]/(dist[time_like]**2).unsqueeze(1)) , max=self.max_norm)
            U[time_like,:] = u[time_like] + (inner_yu[time_like]/(beta.abs()-inner_xy[time_like])).unsqueeze(1)  * (x[time_like]+y[time_like])
            assert not torch.isnan(U[time_like,:] ).any()

        if True in space_like:
            # print('pt.space_like')
            # U[space_like,:] = torch.clamp((inner[space_like]/dist[space_like]).unsqueeze(1)* (x[space_like]*torch.sin(dist[space_like]).unsqueeze(1) - (log_xy[space_like]/dist[space_like].unsqueeze(1))*torch.cos(dist[space_like]).unsqueeze(1)) + (u[space_like] + inner[space_like].unsqueeze(1)*log_xy[space_like]/(dist[space_like]**2).unsqueeze(1)), max=self.max_norm)
            U[space_like,:] = u[space_like] - (inner_yu[space_like]/(-beta.abs()+inner_xy[space_like])).unsqueeze(1)  * (x[space_like]+y[space_like])
            assert not torch.isnan(U[space_like,:] ).any()
            
        if True in null_like:
            # print('null_like')
            # print('pt.null_like')
            # U[null_like,:] = torch.clamp((inner[null_like]/dist[null_like]).unsqueeze(1)*(x[null_like]+log_xy[null_like]/2) + u[null_like], max=self.max_norm)
            U[null_like,:] = u[null_like]
            assert not torch.isnan(U[null_like,:]).any()
        # assert not torch.isnan(U).any()
        U = torch.clamp(U, max=1.0)
        return U

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.proj(x + u)


