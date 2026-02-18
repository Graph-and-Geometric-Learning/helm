import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from ...nn.linear.hnn_layers import HypLinear
from ...utils import *

'''
H2HGCN, module code in progress
'''


class H2HGCN(nn.Module):
    
    def __init__(self, args, logger):
        super(H2HGCN, self).__init__()
        self.debug = False
        self.args = args
        self.logger = logger
        self.set_up_params()
        self.activation = nn.SELU()
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.linear = nn.Linear(
                int(args.feature_dim), int(args.dim),
        )
        nn_init(self.linear, self.args.proj_init)
        self.args.eucl_vars.append(self.linear)	

        if self.args.task == 'nc':
            self.distance = CentroidDistance(args, logger, args.manifold, c=self.curvatures[0])


    def create_params(self):
        """
        create the GNN params for a specific msg type
        """
        msg_weight = []
        layer = self.args.num_layers if not self.args.tie_weight else 1
        for iii in range(layer):
            M = torch.zeros([self.args.dim-1, self.args.dim-1], requires_grad=True)
            init_weight(M, 'orthogonal')
            M = nn.Parameter(M)
            self.args.stie_vars.append(M)
            msg_weight.append(M)
        return nn.ParameterList(msg_weight)

    def set_up_params(self):
        """
        set up the params for all message types
        """
        self.type_of_msg = 1

        for i in range(0, self.type_of_msg):
            setattr(self, "msg_%d_weight" % i, self.create_params())

    def apply_activation(self, node_repr, c):
        """
        apply non-linearity for different manifolds
        """
        if self.args.select_manifold in {"poincare", "euclidean"}:
            return self.activation(node_repr)
        elif self.args.select_manifold == "lorentz":
            return self.args.manifold.poincare_to_lorentz(
                self.activation(self.args.manifold.lorentz_to_poincare(node_repr, k=c)), k=c
            )

    def split_input(self, adj_mat, weight):
        return [adj_mat], [weight]

    def p2k(self, x, c):
        denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
        return 2 * x / denom

    def k2p(self, x, c):
        denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
        return x / denom

    def lorenz_factor(self, x, *, c=1.0, dim=-1, keepdim=False):
        """
            Calculate Lorenz factors
        """
        x_norm = x.pow(2).sum(dim=dim, keepdim=keepdim)
        x_norm = torch.clamp(x_norm, 0, 0.9)
        tmp = 1 / torch.sqrt(1 - c * x_norm)
        return tmp
     
    def from_lorentz_to_poincare(self, x, c):
        """
        Args:
            u: [batch_size, d + 1]
        """
        d = x.size(-1) - 1
        beta_sqrt = c.reciprocal().sqrt()
        return x.narrow(-1, 1, d) * beta_sqrt / (x.narrow(-1, 0, 1) + beta_sqrt)

    def h2p(self, x, c):
        return self.from_lorentz_to_poincare(x, c)

    def from_poincare_to_lorentz(self, x, c, eps=1e-6):
        """
        Args:
            u: [batch_size, d]
        """
        x_norm_square = x.pow(2).sum(-1, keepdim=True)
        beta = c.reciprocal()
        beta_sqrt = beta.sqrt()
        x_space = 2 * beta_sqrt * x
        x_space = x_space / (beta - x_norm_square).clamp_min(eps)
        x_time = (beta + x_space ** 2).sqrt()
        tmp = torch.cat((x_time, x_space), dim=1)
        return  tmp

    def p2h(self, x, c):
        return  self.from_poincare_to_lorentz(x, c)

    def p2k(self, x, c=1.0):
        denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
        return 2 * x / denom

    def k2p(self, x, c=1.0):
        denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
        return x / denom

    def h2k(self, x, c):
        tmp = x.narrow(-1, 1, x.size(-1)-1) / x.narrow(-1, 0, 1)
        return tmp
        
    def k2h(self, x, c):
        x_norm_square = x.pow(2).sum(-1, keepdim=True) 
        x_norm_square = torch.clamp(x_norm_square, max=0.9)
        tmp = torch.ones((x.size(0),1)).cuda().to(self.args.device)
        tmp1 = torch.cat((tmp, x), dim=1)
        tmp2 = 1.0 / torch.sqrt(1.0 - x_norm_square)
        tmp3 = (tmp1 * tmp2)
        return tmp3 


    def hyperbolic_mean(self, y, node_num, max_neighbor, real_node_num, weight, dim=0, c=1.0, ):
        '''
        y [node_num * max_neighbor, dim]
        '''
        x = y[0:real_node_num*max_neighbor, :]
        weight_tmp = weight.view(-1,1)[0:real_node_num*max_neighbor, :]
        x = self.h2k(x)
        
        lamb = self.lorenz_factor(x, c=c, keepdim=True)
        lamb = lamb  * weight_tmp 
        lamb = lamb.view(real_node_num, max_neighbor, -1)

        x = x.view(real_node_num, max_neighbor, -1) 
        k_mean = (torch.sum(lamb * x, dim=1, keepdim=True) / (torch.sum(lamb, dim=1, keepdim=True))).squeeze()
        h_mean = self.k2h(k_mean)

        virtual_mean = torch.cat((torch.tensor([[1.0]]), torch.zeros(1,y.size(-1)-1)), 1).cuda().to(self.args.device)
        tmp = virtual_mean.repeat(node_num-real_node_num, 1)

        mean = torch.cat((h_mean, tmp), 0)
        return mean	

    def test_lor(self, A):
        tmp1 = (A[:,0] * A[:,0]).view(-1)
        tmp2 = A[:,1:]
        tmp2 = torch.diag(tmp2.mm(tmp2.transpose(0,1)))
        return (tmp1 - tmp2)

    def retrieve_params(self, weight, step):
        """
        Args:
            weight: a list of weights
            step: a certain layer
        """
        layer_weight = torch.cat((torch.zeros((self.args.dim-1, 1)).cuda().to(self.args.device), weight[step]), dim=1)
        tmp = torch.zeros((1, self.args.dim)).cuda().to(self.args.device)
        tmp[0,0] = 1
        layer_weight = torch.cat((tmp, layer_weight), dim=0)
        return layer_weight

    def aggregate_msg(self, node_repr, adj_mat, weight, layer_weight, mask, c):
        """
        message passing for a specific message type.
        """
        node_num, max_neighbor = adj_mat.shape[0], adj_mat.shape[1] 
        combined_msg = node_repr.clone()

        tmp = self.test_lor(node_repr)
        msg = torch.mm(node_repr, layer_weight) * mask
        real_node_num = (mask>0).sum()
        
        # select out the neighbors of each node
        neighbors = torch.index_select(msg, 0, adj_mat.view(-1)) 
        combined_msg = self.hyperbolic_mean(neighbors, node_num, max_neighbor, real_node_num, weight, c=c)
        return combined_msg 

    def get_combined_msg(self, step, node_repr, adj_mat, weight, mask, c):
        """
        perform message passing in the tangent space of x'
        """
        gnn_layer = 0 if self.args.tie_weight else step
        combined_msg = None
        for relation in range(0, self.type_of_msg):
            layer_weight = self.retrieve_params(getattr(self, "msg_%d_weight" % relation), gnn_layer)
            aggregated_msg = self.aggregate_msg(node_repr,
                                                adj_mat[relation],
                                                weight[relation],
                                                layer_weight, mask, c)
            combined_msg = aggregated_msg if combined_msg is None else (combined_msg + aggregated_msg)
        return combined_msg


    def encode(self, node_repr, adj_list, weight):
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(self.args)
        node_repr = self.activation(self.linear(node_repr))
        adj_list, weight = self.split_input(adj_list, weight)
        c = torch.Tensor([1.]).cuda().to(self.args.device)

        mask = torch.ones((node_repr.size(0),1)).cuda().to(self.args.device)
        node_repr = self.args.manifold.exp_map_zero(node_repr, c=c, norm_control = True)

        for step in range(self.args.num_layers):
            node_repr = node_repr * mask
            tmp = node_repr
            combined_msg = self.get_combined_msg(step, node_repr, adj_list, weight, mask, c=c)
            combined_msg = (combined_msg) * mask
            node_repr = combined_msg * mask
            node_repr = self.apply_activation(node_repr, c=c) * mask
            real_node_num = (mask>0).sum()
            node_repr = self.args.manifold.normalize(node_repr, c=c)
        if self.args.task == 'nc':
            _, node_centroid_sim = self.distance(node_repr, mask) 
            return node_centroid_sim.squeeze()
        return node_repr