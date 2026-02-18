import torch
from ..utils.data_utils import load_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='disease_lp')
parser.add_argument('--task', type=str, default='lp')
parser.add_argument('--use_feats', type=int, default=1)
parser.add_argument('--val_prop', type=float, default=0.05)
parser.add_argument('--test_prop', type=float, default=0.1)
parser.add_argument('--split_seed', type=int, default=1234)
parser.add_argument('--normalize_adj', type=int, default=1)
parser.add_argument('--normalize_feats', type=int, default=1)
parser.add_argument('--type', type=str, default='graph')
parser.add_argument('--use_adj_feat', type=bool, default=False)


DATAPATH = 'hypercore/data/'


class Dataset(object):
    def __init__(self):
        self.name = None
        self.data = None
        self.graph = {}
        self.label = None

        self.args = parser.parse_args()
        if self.args.dataset == 'airport':
                self.args.use_adj_feats = True
    
    def set_args(self, use_feats=None, val_prop=None, test_prop=None, split_seed=None, normalize_adj=None, normalize_feats=None):
        if use_feats is not None:
            self.args.use_feats = use_feats
        if val_prop is not None:
            self.args.val_prop = val_prop
        if test_prop is not None:
            self.args.test_prop = test_prop
        if split_seed is not None:
            self.args.split_seed = split_seed
        if normalize_adj is not None:
            self.args.normalize_adj = normalize_adj
        if normalize_feats is not None:
            self.args.normalize_feats = normalize_feats
    
    def load_dataset(self, dataset, task, path=None):
        if path is not None:
            data_path = path
        else:
            data_path = DATAPATH + dataset
        self.name = dataset
        self.args.dataset = dataset
        self.args.task = task
        if self.args.type == 'graph':
            self.data = load_data(self.args, data_path)
        else:
            raise NotImplementedError('Only graph datasets are currently supported with data loader')
        if task == 'nc':
            self.label = self.data['labels']


