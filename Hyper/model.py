from multiprocessing import pool
from turtle import forward
from typing import Callable, Final
import torch
import torch.nn as nn
from torch_sparse import SparseTensor, transpose as sp_transpose
from torch_sparse.matmul import spmm_mean, spmm_max, spmm_sum, spmm_min
from torch_geometric.nn import MessagePassing
from torch import Tensor
from torch_scatter import scatter_mean, scatter_max


class SemiConv(nn.Module):
    use_max: Final[bool]
    use_mean: Final[bool]
    use_sqmean: Final[bool]
    prodpool: Final[bool]

    def __init__(self,
                 hid_dim: int,
                 conv_layer: int,
                 conv_tailln: bool,
                 use_mean: bool = True,
                 use_sqmean: bool = True,
                 use_max: bool = True,
                 prodpool: bool = False,
                 prodjk: bool = False):
        super().__init__()
        self.use_max = use_max
        self.use_mean = use_mean
        self.use_sqmean = use_sqmean
        self.prodpool = prodpool
        mpdim = hid_dim * use_max + hid_dim * use_mean + hid_dim * use_sqmean if not prodpool else hid_dim
        self.linv = MLP(mpdim, hid_dim, conv_layer, True, conv_tailln)
        self.line = MLP(mpdim, hid_dim, conv_layer, True, conv_tailln)
        self.prodjk = prodjk

    def mp(self, inci: SparseTensor, x: Tensor):
        ret = []
        if self.use_mean:
            ret.append(spmm_mean(inci, x))
        if self.use_max:
            ret.append(spmm_max(inci, x)[0])
        if self.use_sqmean:
            ret.append(spmm_mean(inci, torch.square(x)))
        if self.prodpool:
            return torch.stack(ret, dim=-1).prod(dim=-1)
        return torch.cat(ret, dim=-1)

    def forward(self, e2vinci, v2einci, xe, xv):
        xe = xe + self.line(self.mp(v2einci, xv))
        xv = xv + self.linv(self.mp(e2vinci, xe))
        if self.prodjk:
            xe = xe * self.line(self.mp(v2einci, xv))
            xv = xv * self.linv(self.mp(e2vinci, xe))
        else:
            xe = xe + self.line(self.mp(v2einci, xv))
            xv = xv + self.linv(self.mp(e2vinci, xe))
        return xe, xv


class DegFn(nn.Module):

    def __init__(self, hid_dim: int, use_log: bool, use_neg: bool,
                 use_tanh: bool):
        super().__init__()
        self.use_neg = use_neg
        self.use_log = use_log
        self.lin = nn.Sequential(
            nn.Linear(self.use_log * 1 + self.use_neg * 1, hid_dim),
            nn.Tanh() if use_tanh else nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True))

    def forward(self, deg):
        deg = deg + 1
        ret = []
        if self.use_neg:
            ret.append(1 / (deg))
        if self.use_log:
            ret.append(torch.log(deg))
        return self.lin(torch.stack(ret, dim=-1))


class DegInit(nn.Module):

    def __init__(self,
                 hid_dim,
                 use_neg: bool = True,
                 use_log: bool = True,
                 use_tanh: bool = False):
        super().__init__()
        self.degfn = DegFn(hid_dim, use_log, use_neg, use_tanh)

    def forward(self, inci: SparseTensor):
        dege = inci.sum(dim=-1)
        degv = inci.sum(dim=0)
        return self.degfn(dege), self.degfn(degv)


class RandomInit(nn.Module):

    def __init__(self, num_node: int, hid_dim: int, init_dp: float = 0):
        super().__init__()
        self.xv = nn.Embedding(num_node, hid_dim)
        self.dp = nn.Dropout(init_dp) if init_dp > 0.01 else nn.Identity()
        self.lin = nn.Identity()

    def forward(self, inci: SparseTensor):
        xv = self.dp(self.xv.weight)
        xe = spmm_max(inci, self.lin(xv))[0]
        return xe, xv


class HGNN(nn.Module):

    def __init__(self,
                 num_layer: int,
                 hid_dim: int,
                 input_dim: int,
                 init_fn: Callable = None,
                 pool_max: bool = True,
                 pool_mean: bool = True,
                 **kwargs):
        super().__init__()
        self.init_fn = init_fn
        self.initlinv = nn.Sequential(nn.Linear(input_dim, hid_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hid_dim, hid_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hid_dim, hid_dim))
        self.initline = nn.Sequential(nn.Linear(input_dim, hid_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hid_dim, hid_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hid_dim, hid_dim))
        self.mps = nn.ModuleList(
            [SemiConv(hid_dim, **kwargs) for _ in range(num_layer)])
        pool_dim = hid_dim * pool_max + hid_dim * pool_mean
        self.pool_max = pool_max
        self.pool_mean = pool_mean
        self.outputlin = nn.Linear(pool_dim, 1)

    def forward(self,
                inci: SparseTensor,
                tar_inci: SparseTensor,
                xe: Tensor = None,
                xv: Tensor = None,
                inciT: SparseTensor = None):
        if xe is None:
            xe, xv = self.init_fn(inci)
        num_node, num_edge = xv.shape[-2], xe.shape[-2]
        if inciT is None:
            colptr, row, _ = inci.csc()
            inciT = SparseTensor(rowptr=colptr,
                                 col=row,
                                 sparse_sizes=(num_node, num_edge),
                                 is_sorted=True).device_as(xe,
                                                           non_blocking=True)
        xe = self.initline(xe)
        xv = self.initlinv(xv)
        for i in range(len(self.mps)):
            self.mps[i](inciT, inci, xe, xv)
        pool_out = []
        if self.pool_max:
            pool_out.append(spmm_max(tar_inci, xv)[0])
        if self.pool_mean:
            pool_out.append(spmm_mean(tar_inci, xv))
        pool_out = torch.cat(pool_out, dim=-1)
        return self.outputlin(pool_out)


class MLP(nn.Module):

    def __init__(self,
                 indim: int,
                 hid_dim: int,
                 num_layer: int,
                 tailact: bool,
                 tailln: bool = False,
                 outdim: int = None):
        super().__init__()
        if outdim is None:
            outdim = hid_dim
        if num_layer == 0:
            assert indim == hid_dim
            self.mod = nn.Identity()
        else:
            dima = indim
            dimb = hid_dim
            mods = []
            for _ in range(num_layer - 1):
                mods.append(nn.Linear(dima, dimb))
                mods.append(nn.ReLU(inplace=True))
                dima = hid_dim
            dimb = outdim
            mods.append(nn.Linear(dima, dimb))
            if tailln:
                mods.append(nn.LayerNorm(dimb))
            if tailact:
                mods.append(nn.ReLU(inplace=True))
            self.mod = nn.Sequential(*tuple(mods))

    def forward(self, x):
        return self.mod(x)


class PZOHGNN(nn.Module):

    def __init__(self,
                 num_layer: int,
                 hid_dim: int,
                 init_fn: Callable = None,
                 pool1_max: bool = True,
                 pool1_mean: bool = True,
                 pool2_max: bool = True,
                 pool2_mean: bool = True,
                 layerwise_label: bool = False,
                 init_layer: int = 3,
                 init_tailact: bool = False,
                 init_tailln: bool = False,
                 conv_layer: int = 1,
                 conv_tailln: bool = True,
                 pool1_layer: int = 1,
                 pool1_tailln: bool = False,
                 pool2_layer: int = 2,
                 use_degmask: bool = False,
                 **kwargs):
        super().__init__()
        self.init_fn = init_fn
        self.initlinv = MLP(hid_dim, hid_dim, init_layer, init_tailact,
                            init_tailln)
        self.initline = MLP(hid_dim, hid_dim, init_layer, init_tailact,
                            init_tailln)
        self.mps = nn.ModuleList([
            SemiConv(hid_dim,
                     conv_layer=conv_layer,
                     conv_tailln=conv_tailln,
                     **kwargs) for _ in range(num_layer)
        ])
        pool1_dim = hid_dim * pool1_max + hid_dim * pool1_mean
        pool2_dim = hid_dim * pool2_max + hid_dim * pool2_mean
        self.pool1_max = pool1_max
        self.pool1_mean = pool1_mean
        self.outputlin1 = MLP(pool1_dim, hid_dim, pool1_layer, True,
                              pool1_tailln)
        self.pool2_max = pool2_max
        self.pool2_mean = pool2_mean
        self.outputlin2 = MLP(pool2_dim, hid_dim, pool2_layer, False, False, 1)
        self.degmask = DegFn(pool2_dim, True, True,
                             True) if use_degmask else None
        self.zolabel = MLP(hid_dim, hid_dim, 1, True, False)
        self.layerwise_label = layerwise_label
        if layerwise_label:
            self.zolabels = nn.ModuleList([
                MLP(hid_dim, hid_dim, 1, True, False) for _ in range(num_layer)
            ])

    def deal_tar_inci(self, tar_inci: SparseTensor):
        _, nodes, _ = tar_inci.csr()
        _, num_tar_node = tar_inci.sparse_sizes()
        batch = tar_inci.coo()[0]
        tar_inci = tar_inci[batch]
        #print(tar_inci.sparse_sizes(), num_tar_node, nodes.max())
        row, col, _ = tar_inci.coo()
        col += num_tar_node * nodes[row]
        return batch, SparseTensor(
            row=row,
            col=col,
            sparse_sizes=(batch.shape[0],
                          num_tar_node * num_tar_node)).device_as(batch)

    def forward(self,
                inci: SparseTensor,
                tar_inci: SparseTensor,
                xe: Tensor = None,
                xv: Tensor = None,
                inciT: SparseTensor = None):
        _, tar_nodes, _ = tar_inci.coo()
        tar_nodes = torch.unique(tar_nodes)
        n_tar_node = tar_nodes.shape[0]
        if xe is None:
            xe, xv = self.init_fn(inci)  #xe (M, d), xv(N, d)
        xe = xe.unsqueeze(0).expand(n_tar_node, -1, -1)
        xv = xv.unsqueeze(0).repeat(n_tar_node, 1, 1)
        idx = torch.arange(n_tar_node, device=xe.device)
        xv[idx, tar_nodes] = self.zolabel(xv[idx, tar_nodes])
        num_node, num_edge = xv.shape[-2], xe.shape[-2]
        if inciT is None:
            colptr, row, _ = inci.csc()
            inciT = SparseTensor(rowptr=colptr,
                                 col=row,
                                 sparse_sizes=(num_node, num_edge),
                                 is_sorted=True).device_as(xe,
                                                           non_blocking=True)
        xe = self.initline(xe)
        xv = self.initlinv(xv)
        for i in range(len(self.mps)):
            self.mps[i](inciT, inci, xe, xv)
            if self.layerwise_label:
                xv = xv.clone()
                xv[idx, tar_nodes] = self.zolabels[i](xv[idx, tar_nodes])
        xv = xv[:, tar_nodes].flatten(0, 1)
        tar_inci = tar_inci[:, tar_nodes]
        tar_deg = tar_inci.sum(dim=-1)
        #print(tar_inci.sparse_sizes())
        batch, tar_inci = self.deal_tar_inci(tar_inci)
        #print(xv.shape, tar_inci.sparse_sizes())
        pool1_out = []
        if self.pool1_max:
            pool1_out.append(spmm_max(tar_inci, xv)[0])
        if self.pool1_mean:
            pool1_out.append(spmm_mean(tar_inci, xv))
        pool1_out = torch.cat(pool1_out, dim=-1)
        pool1_out = self.outputlin1(pool1_out)
        pool2_out = []
        if self.pool2_max:
            pool2_out.append(scatter_max(pool1_out, batch, dim=0)[0])
        if self.pool2_mean:
            pool2_out.append(scatter_mean(pool1_out, batch, dim=0))
        pool2_out = torch.cat(pool2_out, dim=-1)
        if self.degmask is not None:
            pool2_out = pool2_out * self.degmask(tar_deg)
        return self.outputlin2(pool2_out)


class MLPNN(nn.Module):

    def __init__(self,
                 num_node: int,
                 hid_dim: int,
                 dp0: float = 0.1,
                 scalefreq: bool = True,
                 num_layer: int = 2,
                 share_lin: bool = True,
                 layerwise_label: bool = False,
                 share_label: bool = True,
                 use_act: bool = False,
                 no_label: bool = False,
                 **kwargs):
        super().__init__()
        self.vemb = nn.Embedding(num_node,
                                 hid_dim,
                                 scale_grad_by_freq=scalefreq)
        self.post = nn.Sequential(nn.LayerNorm(hid_dim),
                                  nn.Dropout(p=dp0, inplace=True))
        self.out = nn.Linear(hid_dim, 1)
        self.lins = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim),
                          nn.ReLU(inplace=True) if use_act else nn.Identity())
            for _ in range(num_layer if not share_lin else 1)
        ])
        self.zolabels = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim),
                          nn.ReLU(inplace=True) if use_act else nn.Identity())
            if not no_label else nn.Identity()
            for _ in range(num_layer if not share_label else 1)
        ])
        self.num_layer = num_layer
        self.share_lin = share_lin
        self.layerwise_label = layerwise_label
        self.share_label = share_label

    def deal_tar_inci(self, tar_inci: SparseTensor):
        _, nodes, _ = tar_inci.csr()
        _, num_tar_node = tar_inci.sparse_sizes()
        batch = tar_inci.coo()[0]
        tar_inci = tar_inci[batch]
        row, col, _ = tar_inci.coo()
        col += num_tar_node * nodes[row]
        return batch, SparseTensor(
            row=row,
            col=col,
            sparse_sizes=(batch.shape[0],
                          num_tar_node * num_tar_node)).device_as(batch)

    def forward(self,
                inci: SparseTensor,
                tar_inci: SparseTensor,
                xe: Tensor = None,
                xv: Tensor = None,
                inciT: SparseTensor = None):
        xv = self.vemb.weight
        
        if inciT is None:
            colptr, row, _ = inci.csc()
            num_edge, num_node = inci.sparse_sizes()
            inciT = SparseTensor(rowptr=colptr,
                                 col=row,
                                 sparse_sizes=(num_node, num_edge),
                                 is_sorted=True).device_as(xv,
                                                           non_blocking=True)
        inci = inciT @ inci
        '''
        inci = inci.to_torch_sparse_coo_tensor()
        inci = inci.t() @ inci
        inci = SparseTensor.from_torch_sparse_coo_tensor(inci, has_value=False)
        '''
        _, tar_nodes, _ = tar_inci.coo()
        tar_nodes = torch.unique(tar_nodes)
        n_tar_node = tar_nodes.shape[0]

        xv = self.post(xv)
        xv = xv.unsqueeze(0).repeat(n_tar_node, 1, 1)

        idx = torch.arange(n_tar_node, device=xv.device)

        for i in range(self.num_layer):
            if i == 0 or self.layerwise_label:
                xv[idx,
                   tar_nodes] = self.zolabels[0 if self.share_label else i](
                       xv[idx, tar_nodes])
            #xe = spmm_max(inci, self.lins[0 if self.share_lin else i](xv))[0]
            #xv = xv + spmm_max(inciT, xe)[0]
            xv = xv + spmm_max(inci, self.lins[0 if self.share_lin else i](xv))[0]

        xv = xv[:, tar_nodes].flatten(0, 1)
        tar_inci = tar_inci[:, tar_nodes]
        batch, tar_inci = self.deal_tar_inci(tar_inci)
        xe = spmm_max(tar_inci, xv)[0]
        xe = scatter_max(xe, batch, dim=0)[0]
        return self.out(xe)


class MLPNN_label(nn.Module):

    def __init__(self,
                 num_node: int,
                 hid_dim: int,
                 dp0: float = 0.1,
                 scalefreq: bool = True,
                 num_layer: int = 2,
                 share_lin: bool = True,
                 layerwise_label: bool = False,
                 share_label: bool = True,
                 use_act: bool = False,
                 no_label: bool = False,
                 **kwargs):
        super().__init__()
        self.vemb = nn.Embedding(num_node,
                                 hid_dim,
                                 scale_grad_by_freq=scalefreq)
        self.post = nn.Sequential(nn.LayerNorm(hid_dim),
                                  nn.Dropout(p=dp0, inplace=True))
        self.out = nn.Linear(hid_dim, 1)
        self.lins = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim),
                          nn.ReLU(inplace=True) if use_act else nn.Identity())
            for _ in range(num_layer if not share_lin else 1)
        ])
        self.zolabels = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim),
                          nn.ReLU(inplace=True) if use_act else nn.Identity())
            if not no_label else nn.Identity()
            for _ in range(num_layer if not share_label else 1)
        ])
        self.num_layer = num_layer
        self.share_lin = share_lin
        self.layerwise_label = layerwise_label
        self.share_label = share_label

    def deal_tar_inci(self, tar_inci: SparseTensor):
        _, nodes, _ = tar_inci.csr()
        _, num_tar_node = tar_inci.sparse_sizes()
        batch = tar_inci.coo()[0]
        tar_inci = tar_inci[batch]
        row, col, _ = tar_inci.coo()
        col += num_tar_node * nodes[row]
        return batch, SparseTensor(
            row=row,
            col=col,
            sparse_sizes=(batch.shape[0],
                          num_tar_node * num_tar_node)).device_as(batch)

    def forward(self,
                inci: SparseTensor,
                tar_inci: SparseTensor,
                xe: Tensor = None,
                xv: Tensor = None,
                inciT: SparseTensor = None):
        
        xv = self.vemb.weight
        
        if inciT is None:
            colptr, row, _ = inci.csc()
            num_edge, num_node = inci.sparse_sizes()
            inciT = SparseTensor(rowptr=colptr,
                                 col=row,
                                 sparse_sizes=(num_node, num_edge),
                                 is_sorted=True).device_as(xv,
                                                           non_blocking=True)
        inci = inciT @ inci
        '''
        inci = inci.to_torch_sparse_coo_tensor()
        inci = inci.t() @ inci
        inci = SparseTensor.from_torch_sparse_coo_tensor(inci, has_value=False)
        '''
        num_tar = tar_inci.sparse_sizes()[0]
        tar_subg, tar_nodes, _ = tar_inci.coo()

        xv = self.post(xv)

        xv = xv.unsqueeze(0).repeat(num_tar, 1, 1)
        for i in range(self.num_layer):
            if i == 0 or self.layerwise_label:
                xv[tar_subg,
                   tar_nodes] = self.zolabels[0 if self.share_label else i](
                       xv[tar_subg, tar_nodes])
            xv = xv + spmm_max(inci, self.lins[0 if self.share_lin else i](xv))[0]
        xv = xv[tar_subg, tar_nodes]
        xe = scatter_max(xv, tar_subg, dim=0, dim_size=num_tar)[0]
        return self.out(xe)
