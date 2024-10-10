from tokenize import Floatnumber
from torch_sparse import SparseTensor
import numpy as np
from typing import Final, Tuple, List, Set
import torch
import random
from torch import Tensor


def sparse2tuplelist(inci: SparseTensor):
    rowptr, col, _ = inci.csr()
    edges = []
    for i in range(len(rowptr) - 1):
        tmp: np.ndarray = col[rowptr[i]:rowptr[i + 1]].numpy()
        edges.append(tuple(sorted(tmp.tolist())))
    return edges


def tuplelist2sparse(edges: List[Tuple], num_node: int):
    rowptr = torch.tensor([0] + [len(_) for _ in edges]).cumsum_(dim=0)
    col = torch.tensor([b for a in edges for b in a])
    return SparseTensor(rowptr=rowptr, col=col, sparse_sizes=(len(edges), num_node))


def neg_sample(num_nodes: int, k: int, base_hyperedges: Set[Tuple],
               tar_hyperedges: List[Tuple]) -> List[Tuple]:
    negatives = []
    for tar in tar_hyperedges:
        cnt = 0
        while cnt < k:
            one_neg = list(tar)
            id_to_change = random.randint(0, len(tar) - 1)
            one_neg[id_to_change] = random.randint(0, num_nodes - 1)
            one_neg = tuple(sorted(one_neg))
            if one_neg not in base_hyperedges:
                negatives.append(one_neg)
                cnt += 1
    return negatives

class tensorDataloader:
    shuffle: Final[bool]
    def __init__(self,
                 dat: Tensor, # (M')
                 batch_size: int,
                 shuffle: bool):
        self.dat = dat
        self.len = self.dat.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        ret = (self.len+self.batch_size-1) // self.batch_size
        return ret

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            self.perm = torch.randperm(self.len)
        return self

    def __next__(self):
        if self.idx >= self.len:
            raise StopIteration
        if self.shuffle == True and self.idx + self.batch_size > self.len:
            raise StopIteration
        #print(self.idx, self.len, flush=True)
        if self.shuffle:
            ret = self.dat[self.perm[self.idx: self.idx+self.batch_size]]
        else:
            ret = self.dat[self.idx: self.idx+self.batch_size]
        self.idx += self.batch_size
        return ret