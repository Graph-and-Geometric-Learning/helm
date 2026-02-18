import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ...utils import *
# from utils.pre_utils import nn_init

class CentroidDistance(nn.Module):
    """
    Implement a model that calculates the pairwise distances between node representations
    and centroids
    """
    def __init__(self, args, logger, manifold, c):
        super(CentroidDistance, self).__init__()
        self.args = args
        self.logger = logger
        self.manifold = manifold
        self.debug = False
        self.c = c

        # centroid embedding
        self.centroid_embedding = nn.Embedding(
            args.num_centroid, args.dim,
            sparse=False,
            scale_grad_by_freq=False,
        )
        nn_init(self.centroid_embedding, self.args.proj_init)
        args.eucl_vars.append(self.centroid_embedding)

    def forward(self, node_repr, mask):
        """
        Args:
            node_repr: [node_num, dim] 
            mask: [node_num, 1] 1 denote real node, 0 padded node
        return:
            graph_centroid_dist: [1, num_centroid]
            node_centroid_dist: [1, node_num, num_centroid]
        """
        node_num = node_repr.size(0)

        # broadcast and reshape node_repr to [node_num * num_centroid, dim]
        node_repr =  node_repr.unsqueeze(1).expand(
                                                -1,
                                                self.args.num_centroid,
                                                -1).contiguous().view(-1, self.args.dim)

        # broadcast and reshape centroid embeddings to [node_num * num_centroid, dim]
        centroid_repr = self.manifold.exp_map_zero(self.centroid_embedding(th.arange(self.args.num_centroid).cuda().to(self.args.device)), self.c)
        centroid_repr = centroid_repr.unsqueeze(0).expand(
                                                node_num,
                                                -1,
                                                -1).contiguous().view(-1, self.args.dim) 
        # get distance
        node_centroid_dist = self.manifold.induced_distance(node_repr, centroid_repr, self.c) 
        node_centroid_dist = node_centroid_dist.view(1, node_num, self.args.num_centroid) 
        # average pooling over nodes
        graph_centroid_dist = th.sum(node_centroid_dist, dim=1) / th.sum(mask)
        return graph_centroid_dist, node_centroid_dist

