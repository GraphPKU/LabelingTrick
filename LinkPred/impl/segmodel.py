from .model import OnlyConv
from torch import Tensor
import torch.nn as nn
import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import MessagePassing
class OnlyConv(MessagePassing):
    def __init__(self, mlp: nn.Module, aggr="add", **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.mlp = mlp

    def forward(self, x: Tensor, ei: Tensor) -> Tensor:
        ret = self.mlp(x)
        ret = self.propagate(ei, x=ret)
        return ret

    def message(self, x_j):
        return x_j

class FLabelingNet(nn.Module):

    def __init__(self, convs: nn.ModuleList, f0s: nn.ModuleList,
                 f1s: nn.ModuleList):
        super().__init__()
        self.convs = convs
        self.f0s = f0s
        self.f1s = f1s

    def forward(self, x: Tensor, adj: SparseTensor, pos: Tensor):
        '''
        x (N, F)
        pos(M, 2) 
        emb (M, 2, F)
        '''
        idx = pos.flatten()
        x = x
        for i in range(len(self.convs)):
            x0 = self.f0s[i](x)
            x0[idx] = self.f1s[i](x[idx]) 
            x = self.convs[i](x0, adj)
        x = x[pos]
        return x


class PLabelingNet2Set(nn.Module):

    def __init__(self, convs: nn.ModuleList, f0s: nn.ModuleList,
                 f1s: nn.ModuleList):
        super().__init__()
        self.convs = convs
        self.f0s = f0s
        self.f1s = f1s

    def forward(self, x: Tensor, adj: SparseTensor, pos: Tensor):
        '''
        x (N, F)
        sel_id (B)
        x0 (B, N, F) 
        emb (M, 2, F)
        '''
        idx = pos.t()
        x = x.unsqueeze(0).repeat(2, 1, 1)
        for i in range(len(self.convs)):
            x0 = self.f0s[i](x)
            x0[0, idx[0]] = self.f1s[i](x[0, idx[0]])
            x0[1, idx[1]] = self.f1s[i](x[1, idx[1]])
            x = self.convs[i](x0, adj)
        x = torch.stack([x0[0][idx], x0[1][idx[[1, 0]]]], dim=0).transpose(0, 2)
        return x


class NoLabelingNet(nn.Module):

    def __init__(self, convs: nn.ModuleList):
        super().__init__()
        self.convs = convs

    def forward(self, x: Tensor, adj: SparseTensor, pos: Tensor):
        '''
        x (N, F)
        sel_id (B)
        x0 (B, N, F)
        pos (M, 2) 
        emb (M, 2, F)
        '''
        for i in range(len(self.convs)):
            x = self.convs[i](x, adj)
        x = self.convs[i](x, adj)
        x = x[pos]
        return x


class FullModel(nn.Module):

    def __init__(self, max_x: int, num_node: int, mlp_0: nn.Module, RepreMod: nn.Module,
                 PredMod: nn.Module, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(max_x + 1, emb_dim) if emb_dim > 0 else nn.Identity()
        self.RepreMod = RepreMod
        self.PredMod = PredMod
        self.mlp_0 = mlp_0
        self.num_node = num_node

    def forward(self, x: Tensor, ei: Tensor, pos: Tensor):
        x = self.mlp_0(self.emb(x))
        emb = self.RepreMod(x, ei, pos)
        ret = self.PredMod(emb)
        return ret