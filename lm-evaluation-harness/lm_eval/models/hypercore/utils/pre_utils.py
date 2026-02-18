from collections import defaultdict
import os
import pickle
import json
import torch.nn as nn
import torch as th
import torch.optim as optim
import numpy as np
import random
from ..optimizers import RiemannianSGD
import math
import subprocess
import random

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

def th_dot(x, y, keepdim=True):
    return th.sum(x * y, dim=1, keepdim=keepdim)

def pad_sequence(data_list, maxlen, value=0):
    return [row + [value] * (maxlen - len(row)) for row in data_list]

def normalize_weight(adj_mat, weight):
    degree = [1 / math.sqrt(sum(np.abs(w))) for w in weight]
    for dst in range(len(adj_mat)):
        for src_idx in range(len(adj_mat[dst])):
            src = adj_mat[dst][src_idx]
            weight[dst][src_idx] = degree[dst] * weight[dst][src_idx] * degree[src]

def nn_init(nn_module, method='orthogonal'):
    """
    Initialize a Sequential or Module object
    Args:
        nn_module: Sequential or Module
        method: initialization method
    """
    if method == 'none':
        return
    for param_name, _ in nn_module.named_parameters():
        if isinstance(nn_module, nn.Sequential):
            # for a Sequential object, the param_name contains both id and param name
            i, name = param_name.split('.', 1)
            param = getattr(nn_module[int(i)], name)
        else:
            param = getattr(nn_module, param_name)
        if param_name.find('weight') > -1:
            init_weight(param, method)
        elif param_name.find('bias') > -1:
            nn.init.uniform_(param, -1e-4, 1e-4)

def get_params(params_list, vars_list):
    """
    Add parameters in vars_list to param_list
    """
    for i in vars_list:
        if issubclass(i.__class__, nn.Module):
            params_list.extend(list(i.parameters()))
        elif issubclass(i.__class__, nn.Parameter):
            params_list.append(i)
        else:
            print("Encounter unknown objects")
            exit(1)

def categorize_params(args):
    """
    Categorize parameters into hyperbolic ones and euclidean ones
    """
    stiefel_params, euclidean_params = [], []
    get_params(euclidean_params, args.eucl_vars)
    get_params(stiefel_params, args.stie_vars)
    return stiefel_params, euclidean_params

def get_activation(args):
    if args.activation == 'leaky_relu':
        return nn.LeakyReLU(args.leaky_relu)
    elif args.activation == 'rrelu':
        return nn.RReLU()
    elif args.activation == 'relu':
        return nn.ReLU()
    elif args.activation == 'elu':
        return nn.ELU()
    elif args.activation == 'prelu':
        return nn.PReLU()
    elif args.activation == 'selu':
        return nn.SELU()

def init_weight(weight, method):
    """
    Initialize parameters
    Args:
        weight: a Parameter object
        method: initialization method 
    """
    if method == 'orthogonal':
        nn.init.orthogonal_(weight)
    elif method == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(weight)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')


def get_stiefel_optimizer(args, params, lr_stie):
    if args.stiefel_optimizer == 'rsgd':
        optimizer = RiemannianSGD(
            args,
            params,
            lr=lr_stie,
        )
    elif args.stiefel_optimizer == 'ramsgrad':
        optimizer = RiemannianAMSGrad(
            args,
            params,
            lr=lr_stie,
        )
    else:
        print("unsupported hyper optimizer")
        exit(1)        
    return optimizer

def get_lr_scheduler(args, optimizer):
    if args.lr_scheduler == 'exponential':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
    elif args.lr_scheduler == 'cycle':
        return optim.lr_scheduler.CyclicLR(optimizer, 0, max_lr=args.lr, step_size_up=20, cycle_momentum=False)
    elif args.lr_scheduler == 'step':
        return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.step_lr_reduce_freq),
        gamma=float(args.step_lr_gamma)
    )
    elif args.lr_scheduler == 'none':
        return NoneScheduler()

def get_optimizer(args, params, lr):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(params, lr=lr, amsgrad=True, weight_decay=args.weight_decay)
    return optimizer

def set_up_optimizer_scheduler(hyperbolic, args, model, lr, lr_stie, pprint=True):
    stiefel_params, euclidean_params = categorize_params(args)
    # assert(len(list(model.parameters())) == len(stiefel_params) + len(euclidean_params))
    optimizer = get_optimizer(args, euclidean_params, lr)
    lr_scheduler = get_lr_scheduler(args, optimizer)
    if len(stiefel_params) > 0:
        stiefel_optimizer = get_stiefel_optimizer(args, stiefel_params, lr_stie)
        stiefel_lr_scheduler = get_lr_scheduler(args, stiefel_optimizer)
    else:
        stiefel_optimizer, stiefel_lr_scheduler = None, None
    return optimizer, lr_scheduler, stiefel_optimizer, stiefel_lr_scheduler