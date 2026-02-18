"""Hyperboloid manifold."""

import torch

from .base import Manifold
from ..utils import arcosh, cosh, sinh 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
from .hyperboloid import Hyperboloid

class PseudoHyperboloid(Manifold):
    
    def __init__(self,space_dim=0, time_dim=10, beta=-1):
        super(PseudoHyperboloid, self).__init__()
        self.name = 'PseudoHyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-10
        self.max_norm = 1e6
        self.dim = space_dim + time_dim
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.dim = space_dim + time_dim
       
    def _check_point_on_manifold(self, x, beta, time_dim=None, rtol=1e-08, atol=1e-02):
        inner = self.inner(x, x, time_dim=time_dim)
        ok = torch.allclose(inner, inner.new((beta.abs(),)).fill_(beta.item()), atol=atol, rtol=rtol)
        if not ok:
            return False
        return True

    def _check_vector_on_tangent(self, x, u, time_dim=None,rtol=1e-08, atol=1e-03):
        inner = self.inner(x, u,time_dim=time_dim)
        ok = torch.allclose(inner, inner.new_zeros((1,)), atol=atol, rtol=rtol)
        if not ok:
            return False
        return True

    def _check_vector_on_tangent0(self, x, beta, time_dim=None, atol=1e-5, rtol=1e-5):
        origin = x.clone()
        origin[:,:] = 0
        origin[:,0] = beta.abs()**0.5
        inner = self.inner(origin, x,time_dim=time_dim)
        ok = torch.allclose(inner, inner.new_zeros((1,)), atol=atol, rtol=rtol)
        if not ok:
            return False
        return True
    
    def inner(self, x, y, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        z = x * y
        res = torch.clamp(z[:,time_dim:].sum(dim=-1) - z[:, 0:time_dim].sum(dim=-1), max=50000)
        return res

    def sqdist(self, x, y, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner = self.inner(x, y, time_dim=time_dim)
        epsilon = 0.00001
        K = inner/beta.abs()
        c1 = K < -1.0 - epsilon
        c2 = (K <= -1.0 + epsilon) & (K >= -1.0 - epsilon)
        c3 = (K > -1.0 + epsilon) & (K < 1 - epsilon)
        c4 = K > 1 + epsilon
        c5 = (K>=1-epsilon) & (K<= 1+epsilon)
        other = (~c1) & (~c2) & (~c3) & (~c4) & (~c5)

        dist2 = x[:,0].clone()
        device = y.get_device()

        if True in c4:
            # print("cycle_tree distance")
            d = torch.min(self.cycle_dist(x[c4], beta) + self.sqdist(-x[c4], y[c4], beta), self.cycle_dist(y[c4],beta) + self.sqdist(x[c4], -y[c4], beta))
            dist2[c4] = torch.clamp(d,max=self.max_norm)
        
        if True in c5:
            # print("cycle distance")
            dist2[c5] = (beta.abs()**0.5)*math.pi 
        
        if True in c1:
            # print('dist:hyperbolic_like')
            u = self.logmap_n(x[c1],y[c1],beta)
            d = self.inner(u,u,time_dim=time_dim).abs()
            dist2[c1] = torch.clamp(d,max=self.max_norm)
            # dist2[c1] = (beta.abs()**0.5) * torch.clamp(torch.acosh(inner[c1]/beta),min=self.min_norm, max=self.max_norm) 
        if True in c2:
            u = self.logmap_n(x[c2],y[c2],beta)
            # d = u.norm(dim=1,p=2)
            d = self.inner(u,u,time_dim=time_dim).abs()
            dist2[c2] = torch.clamp(d,max=self.max_norm)
            # print('dist:Euclidean_like', dist2[c2].max().item())
            # dist2[c2] = 0
        if True in c3:
            u = self.logmap_n(x[c3],y[c3],beta)
            d = self.inner(u,u,time_dim=time_dim).abs()
            dist2[c3] = torch.clamp(d, max=self.max_norm)
            # print('dist:shperical_like',dist2[c3].max().item() )
            # dist2[c3] = (beta.abs()**0.5) * torch.clamp(torch.acos(inner[c3]/beta),min=self.min_norm, max=self.max_norm)
        return torch.clamp(dist2, min=0.00001,max=50)

    def cycle_dist(self,x,beta):
        return (beta.abs()**0.5)*math.pi #* x.norm(dim=1,p=2)

    def expmap(self, x, v, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        epsilon = 0.00001
        n = v.shape[0]
        d = v.shape[1]
        inner = self.inner(v, v, time_dim=time_dim)
        norm_product = torch.clamp(inner.abs(),min=self.min_norm).sqrt()
        norm_product = torch.clamp(norm_product, max=self.max_norm).view(norm_product.size(0),-1)

        space_like = inner < -epsilon
        time_like = inner > epsilon
        null_geodesic = (~space_like) & (~time_like)
        other = (~time_like) & (~space_like) & (~null_geodesic)
        U = v.clone()
        abs_beta = 1/(abs(beta) ** 0.5)
        if True in time_like:
            # print('exp:hyperbolic_like')
            beta_product = torch.clamp(abs_beta*norm_product[time_like],max=self.max_norm)
            # print(beta_product.max().item())
            U[time_like,:] = torch.clamp( torch.clamp( x[time_like,:]*torch.clamp(torch.cosh(beta_product),max=self.max_norm),max=self.max_norm)  +  torch.clamp( torch.clamp(v[time_like,:]*torch.sinh(beta_product), max=self.max_norm)/beta_product,  max=self.max_norm),max=self.max_norm)
            assert not torch.isnan( U[time_like,:] ).any()
        if True in space_like:
            # print('exp:spherical_like')
            beta_product = torch.clamp(abs_beta*norm_product[space_like],max=self.max_norm)
            U[space_like,:] = torch.clamp(x[space_like,:]*torch.clamp(torch.cos(beta_product),max=self.max_norm) +  torch.clamp(torch.clamp(v[space_like,:]*torch.sin(beta_product), max=self.max_norm)/beta_product,  max=self.max_norm),max=self.max_norm)
            assert not torch.isnan(  U[space_like,:]).any()
        if True in null_geodesic:
            # print('exp:null_like')
            U[null_geodesic,:] = torch.clamp( x[null_geodesic,:] + v[null_geodesic,:], max=self.max_norm)
            assert not torch.isnan(U[null_geodesic,:]).any()
        assert not torch.isnan(U).any()
        # assert not torch.isnan(self.proj(U)).any()
        return self.proj(U,beta)

    def expmap_0(self,v, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        origin = v.clone()
        origin[:,:] = 0
        origin[:,0] = abs(beta)**0.5
        p = v[:,0] >=0
        n = v[:,0] < 0
        v[:,0] = 0
        U = v.clone()
        
        if True in p:
            U[p] = self.expmap(origin[p], v[p], beta, time_dim=time_dim)
        if True in n:
            U[n] = self.expmap(-origin[n], v[n], beta, time_dim=time_dim)
        return U

    def logmap_0(self, y, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        origin = y.clone()
        origin[:,:] = 0
        origin[:,0] = abs(beta)**0.5
        p = y[:,0] >= 0
        n = y[:,0] < 0
        y[:,0] = 0
        U = y.clone()

        if True in p:
            U[p] = self.logmap(origin[p], y[p], beta, time_dim=time_dim)
            U[p][:,0] = 1.0
        if True in n:
            U[n] = self.logmap(-origin[n], y[n], beta, time_dim=time_dim)
            U[p][:,0] = -1.0
            
        return U

    def to_R_hyperboloid(self, x, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        R = x[:,0:time_dim-1]
        hs = x[:,time_dim:]
        h0 = x[:,time_dim-1].unsqueeze(1)
        hs_norm = torch.sum(hs*hs, dim=1, keepdim=True)
        h0 = (beta.abs() + hs_norm).sqrt()*torch.sign(h0)
        hyperboloid = torch.cat((h0,hs),1)
        U = torch.cat((R,hyperboloid),1)
        return U

    def from_R_hyperboloid(self, x, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        r = x[:,0:time_dim-1]
        hs = x[:,time_dim:]
        h0 = x[:,time_dim-1].unsqueeze(1)
        r_norm = torch.sum(r*r, dim=1, keepdim=True)
        h0_norm = torch.sum(h0*h0, dim=1, keepdim=True)
        h0 = ((h0_norm - r_norm)).abs().sqrt()*torch.sign(h0)
        U = torch.cat((r,h0,hs),1)
        return U

    def logmap0(self, x, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        r_hyperboloid = self.to_R_hyperboloid(x,beta,time_dim)
        r = r_hyperboloid[:,0:time_dim-1]
        hyperboloid = r_hyperboloid[:,time_dim-1:]
        tangent_r = r
        tangent_h = torch.clamp(self.logmap_0(hyperboloid,beta, time_dim=1),max=self.max_norm)
        tangent_rh = torch.cat((tangent_r, tangent_h),1)
        return tangent_rh

    def expmap0(self, x, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        tangent_r = x[:,0:time_dim-1]
        tangent_h = x[:,time_dim-1:]
        x_r = tangent_r
        x_hyperboloid = self.expmap_0(tangent_h, beta, time_dim=1)
        r_hyperboloid = torch.cat((x_r, x_hyperboloid),1)
        U = self.from_R_hyperboloid(r_hyperboloid, beta, time_dim=time_dim)
        U = self.proj(U, beta,time_dim=time_dim)
        return U

    def logmap_n(self, x, y, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        # self.beta = self.beta.cuda().to(y.get_device())
        d = x.shape[1]
        n = x.shape[0]
        inner_positive = self.inner(x,y, time_dim=time_dim)

        inner_positive = torch.clamp(inner_positive, max=self.max_norm)
        abs_beta = abs(beta)
        epsilon = 0.000001
        time_like_positive = inner_positive/abs_beta < -1 - epsilon
        # print(time_like_positive)
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
            U[space_like_positive,:] = torch.clamp(((up/low).repeat(1,d))* torch.clamp((y[space_like_positive,:]-x[space_like_positive,:]*beta_product_positive[space_like_positive].repeat(1,d)),max=self.max_norm),max=self.max_norm)
            assert not torch.isnan(U[space_like_positive,:]).any()
        if True in time_like_positive:
            # print('log:hyperbolic_like')
            up = torch.clamp(torch.acosh(torch.clamp(beta_product_positive[time_like_positive], min=self.min_norm, max=self.max_norm)), max=self.max_norm)
            low = torch.clamp(sqrt_minus_positive[time_like_positive], min=self.min_norm, max=self.max_norm)
            U[time_like_positive,:] = torch.clamp(((up/low).repeat(1,d))*torch.clamp( (y[time_like_positive,:]-x[time_like_positive,:]*beta_product_positive[time_like_positive].repeat(1,d)),max=self.max_norm),max=self.max_norm)
            assert not torch.isnan(U[time_like_positive,:]).any()
        if True in null_geodesic_positive:
            # print('log:null_like')
            U[null_geodesic_positive,:] = torch.clamp(y[null_geodesic_positive,:] - x[null_geodesic_positive,:],max=self.max_norm)
            assert not torch.isnan(U[null_geodesic_positive,:]).any()
        assert not torch.isnan(U).any()
        return U

    def logmap(self, x, y, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_positive = self.inner(x, y, time_dim=time_dim)
        epsilon = 0.00001
        # print(inner_positive.max())
        positive_log_map = inner_positive < abs(beta) - epsilon
        negative_log_map = inner_positive >= abs(beta) + epsilon
        neutral = (~positive_log_map) & (~negative_log_map)
        U = y.clone()
        other = (~positive_log_map) & (~negative_log_map) & (~neutral)
        assert U[other].shape[0] == 0
        if True in positive_log_map:
            U[positive_log_map] = self.logmap_n(x[positive_log_map], y[positive_log_map], beta, time_dim=time_dim)
        if True in negative_log_map:
            print('log:negative')
            # U[negative_log_map] = self.logmap_n(-x[negative_log_map], y[negative_log_map], beta, time_dim=time_dim)
            # U[negative_log_map][:,0] = -1.0
        U[neutral] = y[neutral] - x[neutral]
        U = self.proj_tan(U, x, beta)
        return U

    def proj(self, x, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        if time_dim == self.dim:
            U =  F.normalize(x)*(beta.abs()**0.5)
            return U
        Xtime = F.normalize(x[:,0:time_dim])
        Xspace = x[:,time_dim:].div(beta.abs().sqrt())
        spaceNorm = torch.clamp(torch.sum(Xspace*Xspace, dim=1, keepdim=True),max=self.max_norm)
        if time_dim == 1:
            Xtime = torch.sqrt((spaceNorm).add(1.0)).view(-1,1)
        else:
            Xtime = torch.clamp(torch.clamp(torch.sqrt(spaceNorm.add(1.0)),max=self.max_norm).expand_as(Xtime) * Xtime, max=self.max_norm)
        U =  torch.clamp(torch.cat((Xtime,Xspace),1)*(beta.abs()**0.5), max=self.max_norm)
        return U

    def proj_tan(self, z, x,beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_zx = self.inner(z,x,time_dim=time_dim)
        inner_xx = self.inner(x,x,time_dim=time_dim)
        res = torch.clamp(z - (inner_zx/inner_xx).unsqueeze(1)*x, max=self.max_norm)
        return res

    def proj_tan_0(self, z, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        origin = z.clone()
        origin[:,:] = 0
        origin[:,0] = abs(beta)**0.5
        return self.proj_tan(z, origin,beta,time_dim=time_dim)

    def proj_tan0(self, z, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        origin = z.clone()
        origin[:,:] = 0
        origin[:,time_dim-1] = abs(beta)**0.5

        U = z.clone()
        U[:,0:time_dim-1] = torch.clamp(z[:,0:time_dim-1], max=self.max_norm)
        U[:,time_dim-1:] = torch.clamp(self.proj_tan(z[:,time_dim-1:], origin[:,time_dim-1:], beta, time_dim=1), max=self.max_norm)
        return U
    
    def perform_rescaling_beta(self, X, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        norm_X = X * X
        norm_Xtime = norm_X[:,0:time_dim]
        norm_Xspace = norm_X[:,time_dim:]
        res = torch.clamp( X / torch.clamp( torch.abs( torch.sum(norm_Xspace,dim=1, keepdim=True) - torch.sum(norm_Xtime,dim=1, keepdim=True) ).sqrt().expand_as(X) * beta.abs().sqrt(), max=self.max_norm), max=self.max_norm)
        return res
    
    def mobius_matvec(self, m, x, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        u = self.logmap0(x,beta,time_dim=time_dim)
        mu = torch.clamp(u @ m.transpose(-1, -2), max=self.max_norm)
        if time_dim != self.dim:
            mu = F.normalize(mu)
        mu = self.proj_tan0(mu, beta, time_dim=self.time_dim)
        mu = self.expmap0(mu, beta, time_dim=self.time_dim)
        # mu = self.perform_rescaling_beta(mu, beta, time_dim=self.time_dim)
        return mu

    def mobius_add(self, x, y, beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        origin = x.clone()
        origin[:,:] = 0
        origin[:,0] = abs(beta)**0.5
        y = y.repeat(x.shape[0],1)
        u = self.logmap0(y, beta, time_dim=time_dim)
        assert not torch.isnan(u).any()
        v,p,n = self.ptransp0(x, u, beta,time_dim)
        assert not torch.isnan(v).any()
        U = x.clone()
        if True in p:
            U[p] = self.expmap(x[p], v[p], beta, time_dim=time_dim)
            assert not torch.isnan(U[p]).any()
        if True in n:
            U[n] = -self.expmap(-x[n], v[n], beta, time_dim=time_dim)
            assert not torch.isnan(U[n]).any()

        U = self.proj(U,beta)
        assert not torch.isnan(U).any()
        # U = self.perform_rescaling_beta(U, beta, time_dim=self.time_dim)
        # U = self.proj(self.proj(U,beta),beta)
        # print("exp:",self.inner(U,U).max().item(),  self.inner(U,U).min().item() )
        # assert self._check_point_on_manifold(U,beta)
        return U

    def ptransp0(self,x,u,beta,time_dim):
        origin = x.clone()
        origin[:,:] = 0
        origin[:,time_dim-1] = abs(beta)**0.5
        return self.ptransp(origin, x, u, beta)

    def ptransp(self,x,y,u, beta,time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_positive = self.inner(x, y, time_dim=time_dim)
        epsilon = 0.000001
        p = inner_positive < beta.abs() - epsilon
        n = inner_positive > beta.abs() + epsilon
        neutral = (~p) & (~n)
        U = u.clone()
        U[neutral] = u[neutral]
        if True in p:
            U[p] = self.ptransp_n(x[p], y[p], u[p], beta, time_dim=time_dim)
        if True in n:
            U[n] = self.ptransp_n(x[n], -y[n], -u[n], beta, time_dim=time_dim) 
        return U,p,n
    
    def ptransp_n(self,x,y,u,beta, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_xy = self.inner(x,y,time_dim=time_dim)
        log_xy = self.logmap_n(x, y, beta, time_dim=time_dim) 
        log_yx = self.logmap_n(y,x, beta, time_dim=time_dim) 
        inner_log_xy = torch.clamp(self.inner(log_xy, log_xy, time_dim=time_dim), min=self.min_norm)
        inner = self.inner(u, log_xy, time_dim=time_dim)/beta.abs()
        inner_yu = self.inner(y,u,time_dim=time_dim)
        dist = torch.clamp(self.sqdist(x,y,beta,time_dim=time_dim), min=1e-5, max=self.max_norm)
        # print(inner_log_xy.abs().min(), beta.abs())
        norm = torch.clamp(inner_log_xy.abs().sqrt()/beta.abs().sqrt(), max=self.max_norm)
        epsilon = 0.000001
        time_like = inner_log_xy < -epsilon
        space_like = inner_log_xy > epsilon
        null_like = (~time_like) & (~space_like)
        other = (~time_like) & (~space_like) & (~null_like)
        U = u.clone()
        assert U[other].shape[0] == 0
        norm = norm

        if True in space_like:
            # print('pt.space_like')
            U[space_like,:] = torch.clamp((inner[space_like]/norm[space_like]).unsqueeze(1)*(x[space_like]*torch.sinh(norm[space_like]).unsqueeze(1) + (log_xy[space_like]/norm[space_like].unsqueeze(1))*torch.cosh(norm[space_like]).unsqueeze(1)) + (u[space_like] - inner[space_like].unsqueeze(1)*log_xy[space_like]/(norm[space_like]**2).unsqueeze(1)), max=self.max_norm)
            # U[space_like,:] = torch.clamp(u[space_like] - (inner_yu[space_like]/(-beta.abs()+inner_xy[space_like])).unsqueeze(1)  * (x[space_like]+y[space_like]), max=self.max_norm)
            # print('pt space:',self.inner(y[space_like,:],U[space_like,:]).max().item(),  self.inner(y[space_like,:],U[space_like,:]).min().item()  )
            assert not torch.isnan(U[space_like,:] ).any()
            
        if True in time_like:
            # print('pt.time_like')
            U[time_like,:] = torch.clamp((inner[time_like]/norm[time_like]).unsqueeze(1) * (x[time_like]*torch.sin(norm[time_like]).unsqueeze(1) - (log_xy[time_like]/norm[time_like].unsqueeze(1))*torch.cos(norm[time_like]).unsqueeze(1)) + (u[time_like] + inner[time_like].unsqueeze(1)*log_xy[time_like]/(norm[time_like]**2).unsqueeze(1)) , max=self.max_norm)
            # U[time_like,:] = torch.clamp(u[time_like] + (inner_yu[time_like]/(beta.abs()-inner_xy[time_like])).unsqueeze(1)  * (x[time_like]+y[time_like]), max=self.max_norm)
            # U[time_like,:] = u[time_like] + (inner_yu[time_like]/(beta.abs()-inner_xy[time_like])).unsqueeze(1)  * (x[time_like]+y[time_like])
            # U[time_like,:] = u[time_like] - (inner[time_like]/dist[time_like]).unsqueeze(1)*(log_xy[time_like]+log_yx[time_like])
            # print('pt time:',self.inner(y[time_like,:],U[time_like,:]).max().item(),  self.inner(y[time_like,:],U[time_like,:]).min().item()  )
        if True in null_like:
            # print('null_like')
            # print('pt.null_like')
            U[null_like,:] = torch.clamp((inner[null_like]).unsqueeze(1)*(x[null_like]+log_xy[null_like]/2) + u[null_like], max=self.max_norm)
            # U[null_like,:] = torch.clamp(u[null_like], max=self.max_norm)
            # print('pt light:',self.inner(y[null_like,:],U[null_like,:]).max().item(),  self.inner(y[null_like,:],U[null_like,:]).min().item()  )
            assert not torch.isnan(U[null_like,:]).any()
        # U = u - (inner/dist).unsqueeze(1)*(log_xy+log_yx)
        # print('po:',beta.item(),self.inner(y,y).max().item(),self.inner(y,y).min().item())
        # print('pt:',beta.item(),self.inner(y,U).max().item())
        # assert self._check_vector_on_tangent(y,U)
        # assert not torch.isnan(U).any()
        return U



