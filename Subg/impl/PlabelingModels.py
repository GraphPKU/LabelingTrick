from typing import Final
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import GraphSizeNorm
from torch_geometric.nn.glob import global_max_pool, global_add_pool, global_mean_pool
from .utils import pad2batch
from .models import Seq, MLP
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_sum
from torch import Tensor



def buildAdj(edge_index, edge_weight, n_node: int):
    '''
        Calculating the normalized adjacency matrix.
        Args:
            n_node: number of nodes in graph.
            aggr: the aggregation method, can be "mean", "sum" or "gcn".
    '''
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       value=edge_weight,
                       sparse_sizes=(n_node,
                                     n_node)).to_device(edge_index.device)
    return adj


class GLASSConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for GLASS.
    We use different parameters to transform the features of node with different labels individually, and mix them.
    Args:
        aggr: the aggregation method.
        z_ratio: the ratio to mix the transformed features.
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean",
                 z_ratio=0.8,
                 dropout=0.2):
        super().__init__()
        self.trans_fns = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels)
        ])
        self.comb_fns = nn.ModuleList([
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Linear(in_channels + out_channels, out_channels)
        ])
        self.adj = None
        self.activation = activation
        assert aggr in ["mean", "max", "sum"]
        self.aggr = aggr
        self.gn = nn.LayerNorm(out_channels)
        self.z_ratio = z_ratio
        self.dropout = dropout

    def forward(self, x_, edge_index, edge_weight, node2subg):
        if self.adj is None:
            n_node = x_.shape[-2]
            self.adj = buildAdj(edge_index, edge_weight, n_node)
        # transform node features with different parameters individually.
        x1 = self.activation(self.trans_fns[1](x_))
        x0 = self.activation(self.trans_fns[0](x_))
        # mix transformed feature

        row, col, _ = node2subg.coo()

        x = self.z_ratio * x0 + (1 - self.z_ratio) * x1
        x[row,
          col] = self.z_ratio * x1[row, col] + (1 - self.z_ratio) * x0[row,
                                                                       col]
        # pass messages.
        if self.aggr == "max":
            x = spmm_max(self.adj, x)[0]
        elif self.aggr == "mean":
            x = spmm_mean(self.adj, x)
        elif self.aggr == "sum":
            x = spmm_sum(self.adj, x)
        else:
            raise NotImplementedError
        x = self.gn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat((x, x_), dim=-1)
        # transform node features with different parameters individually.
        x1 = self.comb_fns[1](x)
        x0 = self.comb_fns[0](x)
        # mix transformed feature.
        x = self.z_ratio * x0 + (1 - self.z_ratio) * x1
        x[row,
          col] = self.z_ratio * x1[row, col] + (1 - self.z_ratio) * x0[row,
                                                                       col]
        return x


class EmbZGConv(nn.Module):
    '''
    combination of some GLASSConv layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features.
        conv: the message passing layer we use.
        gn: whether to use nn.LayerNorm.
    '''

    def __init__(self,
                 hidden_channels,
                 output_channels,
                 num_layers,
                 max_deg,
                 dropout=0,
                 activation=nn.ReLU(inplace=True),
                 conv=GLASSConv,
                 gn=True,
                 scalefreq=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1,
                                      hidden_channels,
                                      scale_grad_by_freq=scalefreq)
        self.post_emb = nn.Sequential(nn.LayerNorm(hidden_channels),
                                      nn.Dropout(dropout, inplace=True))
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=hidden_channels,
                     activation=activation,
                     gn=gn,
                     **kwargs))

        self.convs.append(
            conv(in_channels=hidden_channels,
                 out_channels=output_channels,
                 activation=activation,
                 gn=gn,
                 **kwargs))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight, node2subg):
        # z is the node label.
        # convert integer input to vector node features.
        num_subg, num_node = node2subg.sparse_sizes()
        x = self.input_emb(x).reshape(num_node, -1)
        x = self.post_emb(x)
        x = x.unsqueeze_(0).repeat(num_subg, 1, 1)
        # pass messages at each layer.
        for layer, conv in enumerate(self.convs):
            x0 = x
            x = conv(x, edge_index, edge_weight, node2subg)
            x = x + x0
        return x


class SimpleConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for GLASS.
    We use different parameters to transform the features of node with different labels individually, and mix them.
    Args:
        aggr: the aggregation method.
        z_ratio: the ratio to mix the transformed features.
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean",
                 z_ratio=0.8,
                 dropout=0.2,
                 gn=True):
        super().__init__()
        self.trans_fns = nn.ModuleList([
            nn.Sequential(nn.Linear(in_channels, out_channels),
                          nn.LayerNorm(out_channels) if gn else nn.Identity(),
                          nn.Dropout(dropout, inplace=True), activation),
            nn.Sequential(nn.Linear(in_channels, out_channels),
                          nn.LayerNorm(out_channels) if gn else nn.Identity(),
                          nn.Dropout(dropout, inplace=True), activation)
        ])
        self.adj = None

        assert aggr in ["mean", "max", "sum"]
        self.aggr = aggr

        self.z_ratio = z_ratio
        self.lin_out = nn.Sequential(nn.Linear(out_channels, out_channels),
                                     activation)

    def forward(self, x_, edge_index, edge_weight, node2subg):
        if self.adj is None:
            n_node = x_.shape[-2]
            self.adj = buildAdj(edge_index, edge_weight, n_node)
        # transform node features with different parameters individually.
        x1 = self.trans_fns[1](x_)
        x0 = self.trans_fns[0](x_)
        # mix transformed feature
        row, col, _ = node2subg.coo()

        x = self.z_ratio * x0 + (1 - self.z_ratio) * x1
        x[row,
          col] = self.z_ratio * x1[row, col] + (1 - self.z_ratio) * x0[row,
                                                                       col]
        # pass messages.
        if self.aggr == "max":
            x = spmm_max(self.adj, x)[0]
        elif self.aggr == "mean":
            x = spmm_mean(self.adj, x)
        elif self.aggr == "sum":
            x = spmm_sum(self.adj, x)
        else:
            raise NotImplementedError
        x = self.lin_out(x)
        return x


class GLASS(nn.Module):
    '''
    GLASS model: combine message passing layers and mlps and pooling layers.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''

    def __init__(self, conv: EmbZGConv, pred: nn.Module, pool: str):
        super().__init__()
        self.conv = conv
        self.pred = pred
        self.pool = pool
        assert pool in ["max", "mean", "sum"]

    def NodeEmb(self, x, edge_index, edge_weight, node2subg):
        emb = self.conv(x, edge_index, edge_weight, node2subg)
        return emb

    def Pool(self, emb, node2subg):
        batch, subgnodes, _ = node2subg.coo()
        emb = emb[batch, subgnodes]
        if self.pool == "max":
            emb = global_max_pool(emb, batch)
        elif self.pool == "mean":
            emb = global_mean_pool(emb, batch)
        elif self.pool == "sum":
            emb = global_add_pool(emb, batch)
        else:
            raise NotImplementedError
        return emb

    def forward(self, x, edge_index, edge_weight, node2subg):
        emb = self.NodeEmb(x, edge_index, edge_weight, node2subg)
        emb = self.Pool(emb, node2subg)
        return self.pred(emb)


class PGLASS(nn.Module):
    '''
    GLASS model: combine message passing layers and mlps and pooling layers.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''

    def __init__(self, conv: EmbZGConv, pred: nn.Module, pool: str):
        super().__init__()
        self.conv = conv
        self.pred = pred
        self.pool = pool
        assert pool in ["max", "mean", "sum"]

    def NodeEmb(self, x, edge_index, edge_weight, node2tarnode):
        emb = self.conv(x, edge_index, edge_weight, node2tarnode)
        return emb

    def Pool(self, emb: Tensor, node2subg):
        if self.pool == "max":
            emb = spmm_max(node2subg, emb.transpose(0, 1))[0]
            emb = spmm_max(node2subg, emb.transpose(0, 1))[0]
        elif self.pool == "mean":
            emb = spmm_mean(node2subg, emb.transpose(0, 1))
            emb = spmm_mean(node2subg, emb.transpose(0, 1))
        elif self.pool == "sum":
            emb = spmm_sum(node2subg, emb.transpose(0, 1))
            emb = spmm_sum(node2subg, emb.transpose(0, 1))
        else:
            raise NotImplementedError
        emb = torch.diagonal(emb).t_()
        return emb

    def forward(self, x, edge_index, edge_weight, node2subg):
        batch, subgnodes, _ = node2subg.coo()
        tar_nodes, subgnodes = torch.unique(subgnodes, return_inverse=True)
        node2tarnodes = SparseTensor(row=torch.arange(tar_nodes.shape[0],
                                                      device=subgnodes.device),
                                     col=tar_nodes,
                                     sparse_sizes=(tar_nodes.shape[0],
                                                   x.shape[0])).device_as(x)
        emb = self.NodeEmb(x, edge_index, edge_weight, node2tarnodes)
        emb = emb[:, tar_nodes]
        node2subg = SparseTensor(row=batch,
                                 col=subgnodes).to_device(emb.device)
        emb = self.Pool(emb, node2subg)
        return self.pred(emb)


from torch_sparse import sample_adj


class OneHeadPGLASS(nn.Module):
    '''
    GLASS model: combine message passing layers and mlps and pooling layers.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''
    subsetsize: Final[int]

    def __init__(self, conv: EmbZGConv, pred: nn.Module, pool: str,
                 subsetsize: int):
        super().__init__()
        self.conv = conv
        self.pred = pred
        self.pool = pool
        assert pool in ["max", "mean", "sum"]
        self.subsetsize = subsetsize

    def NodeEmb(self, x, edge_index, edge_weight, node2subg):
        emb = self.conv(x, edge_index, edge_weight, node2subg)
        return emb

    def Pool(self, emb, node2subg):
        batch, subgnodes, _ = node2subg.coo()
        emb = emb[batch, subgnodes]
        if self.pool == "max":
            emb = global_max_pool(emb, batch)
        elif self.pool == "mean":
            emb = global_mean_pool(emb, batch)
        elif self.pool == "sum":
            emb = global_add_pool(emb, batch)
        else:
            raise NotImplementedError
        return emb

    def forward(self, x, edge_index, edge_weight, node2subg: SparseTensor):
        num_subg, num_node = node2subg.sparse_sizes()
        node2subgsubset = sample_adj(
            node2subg.cpu(),
            torch.arange(node2subg.sparse_sizes()[0]),
            self.subsetsize, False)#[0].to_device(x.device)
        row, col, _ = node2subgsubset[0].coo()
        col = node2subgsubset[1][col]
        node2subgsubset = SparseTensor(row, col=col, sparse_sizes=(num_subg, num_node)).to_device(x.device)
        emb = self.NodeEmb(x, edge_index, edge_weight, node2subgsubset)
        emb = self.Pool(emb, node2subg)
        return self.pred(emb)




class noGLASS(nn.Module):
    '''
    GLASS model: combine message passing layers and mlps and pooling layers.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''

    def __init__(self, conv: EmbZGConv, pred: nn.Module, pool: str):
        super().__init__()
        self.conv = conv
        self.pred = pred
        self.pool = pool
        assert pool in ["max", "mean", "sum"]

    def NodeEmb(self, x, edge_index, edge_weight, node2subg):
        emb = self.conv(x, edge_index, edge_weight, node2subg)
        return emb

    def Pool(self, emb, node2subg):
        batch, subgnodes, _ = node2subg.coo()
        emb = emb[batch, subgnodes]
        if self.pool == "max":
            emb = global_max_pool(emb, batch)
        elif self.pool == "mean":
            emb = global_mean_pool(emb, batch)
        elif self.pool == "sum":
            emb = global_add_pool(emb, batch)
        else:
            raise NotImplementedError
        return emb

    def forward(self, x, edge_index, edge_weight, node2subg: SparseTensor):
        emb = self.NodeEmb(x, edge_index, edge_weight, SparseTensor(row=edge_index[0][:0], col=edge_index[0][:0], sparse_sizes=node2subg.sparse_sizes()))
        emb = self.Pool(emb, node2subg)
        return self.pred(emb)