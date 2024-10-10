from torch import Tensor
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
import torch
import numpy as np
from typing import Iterable
from torch_sparse import SparseTensor
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import random
from typing import Optional



@torch.jit.script
def to_directed(ei: Tensor):
    ei = torch.cat((ei, ei[[1, 0]]), dim=1)
    mask = (ei[0] > ei[1])
    ei = torch.unique(ei[:, mask], dim=1)
    return ei


def do_edge_split(data, val_ratio=0.05, test_ratio=0.1):
    data = train_test_split_edges(data, val_ratio, test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index
    split_edge['train']['edge_neg'] = data.train_neg_edge_index
    split_edge['valid']['edge'] = data.val_pos_edge_index
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index
    split_edge['test']['edge'] = data.test_pos_edge_index
    split_edge['test']['edge_neg'] = data.test_neg_edge_index
    return split_edge


def dropoutmask(num: int, p: float, device: torch.DeviceObjType):
    return torch.rand((num), dtype=torch.float, device=device) > p


def gcn_norm(edge_index: Tensor,
             num_node: int,
             improved: bool = True,
             add_self_loops: bool = True):
    # copied from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html
    fill_value = 2.0 if improved else 1.0
    edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_node)
        edge_weight = tmp_edge_weight
    deg = scatter_add(edge_weight, edge_index[1], dim=0, dim_size=num_node)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    ei, ea = edge_index, deg_inv_sqrt[
        edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]
    return ei, ea


def eiea2adj(ei: Tensor, ea: Tensor, num_node: int) -> SparseTensor:
    adj = SparseTensor(row=ei[0],
                       col=ei[1],
                       value=ea,
                       sparse_sizes=(num_node, num_node)).coalesce()
    return adj


class tensorDataloader:

    def __init__(self, tensortuple: Iterable[Tensor], batch_size: int,
                 droplast: bool, shuffle: bool, device: torch.DeviceObjType):
        self.tensortuple = tuple(tensor.contiguous().to(device)
                                 for tensor in tensortuple)
        self.device = device
        self.droplast = droplast
        lens = [tensor.shape[0] for tensor in self.tensortuple]
        assert np.all(np.array(lens) == lens[0]), "tensors must have same size"
        self.length = lens[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        ret = self.length / self.batch_size
        if self.droplast:
            ret = np.floor(ret)
        else:
            ret = np.ceil(ret)
        return ret

    def __iter__(self):
        if self.shuffle:
            self.perm = torch.randperm(self.length, device=self.device)
        self.idx = 0
        return self

    def get_batch(self, length: Optional[int]):
        if length is None:
            length = self.batch_size
        if self.idx >= self.length:
            return None
        if self.idx + length > self.length and self.droplast:
            return None
        if self.shuffle:
            slice = self.perm[self.idx:self.idx + length]
            ret = tuple(tensor[slice] for tensor in self.tensortuple)
        else:
            ret = tuple(tensor[self.idx:self.idx + length]
                        for tensor in self.tensortuple)
        self.idx += length
        if len(ret) == 1: ret = ret[0]
        return ret

    def __next__(self):
        batch = self.get_batch(self.batch_size)
        if batch is None:
            raise StopIteration
        return batch


@torch.jit.script
def edge2ds(edge: Tensor, edge_neg: Tensor):
    ei = torch.cat((edge, edge_neg), dim=-1)
    y = torch.empty_like(ei[0], dtype=torch.float)
    y[:edge.shape[1]] = 1.0
    y[edge.shape[1]:] = 0.0
    return ei, y


@torch.jit.script
def idx2mask(idx: Tensor, masklen: int):
    a = torch.zeros((masklen), dtype=torch.bool, device=idx.device)
    a[idx] = True
    return a

@torch.jit.script
def mask2idx(mask):
    a = torch.arange(mask.shape[0], dtype=torch.long, device=mask.device)
    return a[mask]

def set_seed(seed: int):
    print("seed ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

