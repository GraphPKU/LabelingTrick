from torch import Tensor
import torch
import torch.nn as nn
from torch_sparse import SparseTensor, matmul
from typing import Iterable, Optional


#@torch.jit.script
def convert_pos(pos: Tensor, num_node: int):
    # pos (B, 2)
    sel_id, inverse_ind = torch.unique(pos, return_inverse=True)
    # a->b: a标记中b的位置=inverse_ind[a]*num_node+b
    ret = pos + inverse_ind[:, [1, 0]] * num_node  # ret (B, 2)，分别为a->b, b->a的位置
    return sel_id, ret


def convert_pos2set(pos: Tensor, num_node: int):
    # pos (B, 2)
    sel_id, inverse_ind = torch.unique(pos, return_inverse=True)
    # a->b: a标记中b的位置=inverse_ind[a]*num_node+b
    ret = pos[:, [[0, 1], [1, 0]]] + (inverse_ind * num_node)[:, [[0, 1], [0, 1]]]  # 先pool dim 1再pool dim2
    return sel_id, ret

class OnlyConv(nn.Module):

    def __init__(self, mlp: nn.Module, **kwargs):
        super().__init__()
        self.mlp = mlp

    def forward(self, x: Tensor, adj: SparseTensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        ret = adj @ self.mlp(x)
        return ret


class Convs(nn.Module):

    def __init__(self, output_channels: int, fs: Iterable[nn.Module]) -> None:
        super().__init__()
        self.mods = nn.ModuleList(fs)
        self.conv0 = OnlyConv(output_channels, bias=False)
        self.conv1 = OnlyConv(output_channels, True)

    def forward(self, x: Tensor, adj: SparseTensor):
        for f in self.mods[:-1]:
            x = f(x)
            x = self.conv0(x, adj)
        x = self.mods[-1](x)
        x = self.conv1(x, adj)
        return x


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
        M = pos.shape[0]
        idx0 = torch.arange(M, device=x.device).unsqueeze(-1).repeat((1, 2)).flatten()
        idx1 = pos.flatten()
        for i in range(len(self.convs)):
            if i == 0:
                x0 = self.f0s[i](x).unsqueeze(0).repeat((M, 1, 1))  # (M, N, F)
                x1 = self.f1s[i](x[idx1]) 
                x0[idx0, idx1] = x1
            else:
                x0 = self.f0s[i](x)
                x0[idx0, idx1] = self.f1s[i](x[idx0, idx1])
            x = self.convs[i](x0, adj)
        x = x[idx0, idx1].reshape(M, 2, -1)
        return x


class PLabelingNet(nn.Module):

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
        sel_id, pos = convert_pos(pos, x.shape[0])
        B = sel_id.shape[0]
        N = x.shape[0]
        idx0 = torch.arange(B, device=x.device)
        for i in range(len(self.convs)):
            if i == 0:
                x0 = self.f0s[i](x).unsqueeze(0).repeat((B, 1, 1))  # (B, N, F)
                x1 = self.f1s[i](x[sel_id])  # (B, F)
                x0[idx0, sel_id] = x1
            else:
                x0 = self.f0s[i](x)
                x0[idx0, sel_id] = self.f1s[i](x[idx0, sel_id])
            x = self.convs[i](x0, adj)
        x = x.reshape(B * N, -1)[pos]
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
        sel_id, pos = convert_pos2set(pos, x.shape[0])
        B = sel_id.shape[0]
        N = x.shape[0]
        idx0 = torch.arange(B, device=x.device)
        for i in range(len(self.convs)):
            if i == 0:
                x0 = self.f0s[i](x).unsqueeze(0).repeat((B, 1, 1))  # (B, N, F)
                x1 = self.f1s[i](x[sel_id])  # (B, F)
                x0[idx0, sel_id] = x1
            else:
                x0 = self.f0s[i](x)
                x0[idx0, sel_id] = self.f1s[i](x[idx0, sel_id])
            x = self.convs[i](x0, adj)
        x = x.reshape(B * N, -1)[pos]
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

    def forward(self, x: Tensor, adj: SparseTensor, pos: Tensor):
        x = self.mlp_0(self.emb(x))
        emb = self.RepreMod(x, adj, pos)
        ret = self.PredMod(emb)
        return ret
