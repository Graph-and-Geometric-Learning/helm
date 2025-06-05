from geoopt import ManifoldParameter
from .radam import RiemannianAdam
from .rsgd import RiemannianSGD
import torch

def get_param_groups(model, weight_decay=0):
    no_decay = ['bias', 'scale']
    parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if p.requires_grad and not any(
                nd in n
                for nd in no_decay) and not isinstance(p, ManifoldParameter)
        ],
        'weight_decay':
        weight_decay
    }, {
        'params': [
            p for n, p in model.named_parameters() if p.requires_grad and any(
                nd in n
                for nd in no_decay) or isinstance(p, ManifoldParameter)
        ],
        'weight_decay':
        0.0
    }]
    return parameters

def select_optimizers(model, opt='radam', lr=0.01, weight_decay=0., lr_reduce_freq=1000, gamma=0.9):
    optimizer_grouped_parameters = get_param_groups(model, weight_decay)
    optimizer = None
    if opt == 'radam':
        optimizer = RiemannianAdam(params=optimizer_grouped_parameters,
                                   lr=lr,
                                   stabilize=10)
    elif opt.optimizer == 'rsgd':
        optimizer = RiemannianSGD(params=optimizer_grouped_parameters,
                                  lr=lr,
                                  stabilize=10)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=int(
                                                       lr_reduce_freq),
                                                   gamma=float(gamma))
    return optimizer, lr_scheduler

class Optimizer(object):
    '''
    Taken from HypFormer
    '''

    def __init__(self, model, euc_optimizer_type='adam', euc_lr=0.01, euc_weight_decay=0.0, hyp_optimizer_type='radam', hyp_lr=0.01, 
                    hyp_weight_decay=0.0, stabilize=50, amsgrad=False, nesterov_euc=False, nesterov_hyp=False, momentum_euc=0.9, momentum_hyp=0.9):
        # Separate parameters for Euclidean and Hyperbolic parts of the model
        euc_params = [p for n, p in model.named_parameters() if p.requires_grad and not isinstance(p, ManifoldParameter)]  # Euclidean parameters
        hyp_params = [p for n, p in model.named_parameters() if p.requires_grad and isinstance(p, ManifoldParameter)]  # Hyperbolic parameters
        # Initialize Euclidean optimizer
        if euc_optimizer_type == 'adam':
            optimizer_euc = torch.optim.Adam(euc_params, lr=euc_lr, weight_decay=euc_weight_decay, amsgrad=amsgrad)
        elif euc_optimizer_type == 'sgd':
            optimizer_euc = torch.optim.SGD(euc_params, lr=euc_lr, weight_decay=euc_weight_decay, nesterov=nesterov_euc, momentum=momentum_euc)
        elif euc_optimizer_type == 'adamW':
            optimizer_euc = torch.optim.AdamW(euc_params, lr=euc_lr, weight_decay=euc_weight_decay, amsgrad=amsgrad)
        else:
            raise NotImplementedError("Unsupported Euclidean optimizer type")

        # Initialize Hyperbolic optimizer if there are Hyperbolic parameters
        if hyp_params:
            if hyp_optimizer_type == 'radam':
                optimizer_hyp = RiemannianAdam(hyp_params, lr=hyp_lr, stabilize=stabilize, weight_decay=hyp_weight_decay, amsgrad=amsgrad)
            elif hyp_optimizer_type == 'rsgd':
                optimizer_hyp = RiemannianSGD(hyp_params, lr=hyp_lr, stabilize=stabilize, weight_decay=hyp_weight_decay, nesterov=nesterov_hyp, momentum=momentum_hyp)
            else:
                raise NotImplementedError("Unsupported Hyperbolic optimizer type")

            # Store both optimizers
            self.optimizer = [optimizer_euc, optimizer_hyp]
        else:
            # Store only Euclidean optimizer if there are no Hyperbolic parameters
            self.optimizer = [optimizer_euc]

    def step(self):
        # Perform optimization step for each optimizer
        for optimizer in self.optimizer:
            optimizer.step()

    def zero_grad(self):
        # Reset gradients to zero for each optimizer
        for optimizer in self.optimizer:
            optimizer.zero_grad()

class LR_Scheduler(object):
    '''
    Taken from HypFormer
    '''

    def __init__(self, optimizer_euc, euc_lr_reduce_freq=1000, euc_gamma=0.9, optimizer_hyp=None, hyp_lr_reduce_freq=1000, hyp_gamma=0.9):
        euc_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_euc,
                                                step_size=int(
                                                    euc_lr_reduce_freq),
                                                gamma=float(euc_gamma))
        hyp_lr_scheduler = None
        if optimizer_hyp:
            hyp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_hyp,
                                                    step_size=int(
                                                        hyp_lr_reduce_freq),
                                                    gamma=float(hyp_gamma))
        self.lr_scheduler = []
        self.lr_scheduler.append(euc_lr_scheduler)
        if hyp_lr_scheduler:
            self.lr_scheduler.append(hyp_lr_scheduler)

    def step(self):
        for lr_scheduler in self.lr_scheduler:
            lr_scheduler.step()