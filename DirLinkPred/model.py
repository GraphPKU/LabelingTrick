from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric_signed_directed.nn.directed.complex_relu import complex_relu_layer
from torch_geometric_signed_directed.nn.directed.MagNetConv import MagNetConv


class MagNet_link_prediction(nn.Module):

    def __init__(self,
                 num_features: int,
                 hidden: int = 2,
                 q: float = 0.25,
                 K: int = 2,
                 label_dim: int = 2,
                 activation: bool = True,
                 trainable_q: bool = False,
                 layer: int = 2,
                 dropout: float = 0.5,
                 normalization: str = 'sym',
                 cached: bool = False):
        super(MagNet_link_prediction, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(
            MagNetConv(in_channels=num_features,
                       out_channels=hidden,
                       K=K,
                       q=q,
                       trainable_q=trainable_q,
                       normalization=normalization,
                       cached=cached))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(
                MagNetConv(in_channels=hidden,
                           out_channels=hidden,
                           K=K,
                           q=q,
                           trainable_q=trainable_q,
                           normalization=normalization,
                           cached=cached))

        self.Chebs = chebs
        self.linear = nn.Linear(hidden * 4, label_dim)
        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear.reset_parameters()

    def forward(
            self,
            real: torch.FloatTensor,
            imag: torch.FloatTensor,
            edge_index: torch.LongTensor,
            query_edges: torch.LongTensor,
            edge_weight: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x = torch.cat((real[query_edges[:, 0]], real[query_edges[:, 1]],
                       imag[query_edges[:, 0]], imag[query_edges[:, 1]]),
                      dim=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


class MagNet_link_prediction_labelingtrick(nn.Module):

    def __init__(self,
                 num_features: int,
                 hidden: int = 2,
                 q: float = 0.25,
                 K: int = 2,
                 label_dim: int = 2,
                 activation: bool = True,
                 trainable_q: bool = False,
                 layer: int = 2,
                 dropout: float = 0.5,
                 normalization: str = 'sym',
                 cached: bool = False):
        super().__init__()

        chebs = nn.ModuleList()
        self.inputlin_real = nn.Sequential(nn.Linear(num_features, hidden),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hidden, hidden))
        self.inputlin_imag = nn.Sequential(nn.Linear(num_features, hidden),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hidden, hidden))
        chebs.append(
            MagNetConv(in_channels=hidden,
                       out_channels=hidden,
                       K=K,
                       q=q,
                       trainable_q=trainable_q,
                       normalization=normalization,
                       cached=cached))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(
                MagNetConv(in_channels=hidden,
                           out_channels=hidden,
                           K=K,
                           q=q,
                           trainable_q=trainable_q,
                           normalization=normalization,
                           cached=cached))

        self.Chebs = chebs
        self.linear = nn.Linear(hidden * 4, label_dim)
        self.dropout = dropout
        self.labelemb = nn.Embedding(3, hidden)
        self.register_buffer("one", torch.tensor([1], dtype=torch.long))
        self.register_buffer("two", torch.tensor([2], dtype=torch.long))

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear.reset_parameters()

    def forward(
            self,
            real: torch.FloatTensor,
            imag: torch.FloatTensor,
            edge_index: torch.LongTensor,
            query_edges: torch.LongTensor,
            edge_weight: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        #print(real.shape, imag.shape)
        real = self.inputlin_real(real)
        imag = self.inputlin_imag(imag)
        num_tar = query_edges.shape[0]
        idx = torch.arange(num_tar, device=real.device)
        real = real.unsqueeze(0).repeat(num_tar, 1, 1)
        imag = imag.unsqueeze(0).repeat(num_tar, 1, 1)
        #print(real.shape, imag.shape)
        head, tail = query_edges[:, 0], query_edges[:, 1]
        #print(real[idx, head].shape, self.labelemb(self.one).shape)
        real[idx, head] *= self.labelemb(self.one)
        real[idx, tail] *= self.labelemb(self.two)
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            real[idx, head] *= self.labelemb(self.one)
            real[idx, tail] *= self.labelemb(self.two)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x = torch.cat(
            (real[idx, head], real[idx, tail], imag[idx, head], imag[idx,
                                                                     tail]),
            dim=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


from torch_geometric.nn import GINConv


class ResBlock(nn.Module):

    def __init__(self, mod) -> None:
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return x + self.mod(x)


class RGIN(nn.Module):

    def __init__(self, hid_dim, ln: bool, res: bool, cat_merge: bool,
                 **kwargs) -> None:
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                             nn.LayerNorm(hid_dim) if ln else nn.Identity(),
                             nn.ReLU(inplace=True))
        mlp2 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                             nn.LayerNorm(hid_dim) if ln else nn.Identity(),
                             nn.ReLU(inplace=True))
        self.gin1 = GINConv(ResBlock(mlp1) if res else mlp1)
        self.gin2 = GINConv(ResBlock(mlp2) if res else mlp2)
        self.lin = nn.Sequential(nn.Linear(2 * hid_dim, 2 * hid_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2 * hid_dim, hid_dim),
                                 nn.ReLU(inplace=True))
        self.cat_merge = cat_merge

    def forward(self, x, ei):
        ret1 = self.gin1(x, ei)
        ret2 = self.gin2(x, ei[[1, 0]])
        if self.cat_merge:
            return self.lin(torch.cat((ret1, ret2), dim=-1))
        else:
            return ret1 * ret2


def findedge(tar_edge, src_edge):
    assert tar_edge.shape[0] == 2
    assert src_edge.shape[0] == 2
    n_node = max(torch.max(tar_edge).item(), torch.max(src_edge).item()) + 10
    tar_hash = tar_edge[0] * n_node + tar_edge[1]
    src_hash = src_edge[0] * n_node + src_edge[1]
    #print(tar_edge.shape, tar_hash.shape, src_edge.shape, src_hash.shape)
    tar_hash = torch.sort(tar_hash)[0]
    #print(tar_hash)
    idx = torch.searchsorted(tar_hash[:-1], src_hash)
    ret = (tar_hash[idx] == src_hash)
    #print(ret.all())
    return ret.any()


class GIN_link_prediction_labelingtrick(nn.Module):

    def __init__(self,
                 num_features: int,
                 hidden: int = 2,
                 label_dim: int = 2,
                 layer=1,
                 label: str = "dizo",
                 label_fn: str = "lin",
                 layerwise_label: bool = True,
                 multpred: bool = True,
                 **kwargs):
        super().__init__()
        assert label in ["dizo", "zo", "no", "onehead"]
        self.multpred = multpred
        chebs = nn.ModuleList()
        self.inputlin_real = nn.Sequential(nn.Linear(num_features, hidden),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hidden, hidden))

        for _ in range(0, layer):
            chebs.append(RGIN(hidden, **kwargs))

        self.Chebs = chebs
        self.linear = nn.Linear(hidden if multpred else 2 * hidden, label_dim)
        self.no_label = label == "no"
        assert label_fn in ["lin", "emb"]
        self.label_fn = label_fn
        self.label_lin = nn.ModuleList(
            [nn.Linear(hidden, hidden
                       ), nn.Linear(hidden, hidden) if label=="dizo" else nn.Identity()] if label in
            ["dizo", "onehead"] else [nn.Linear(hidden, hidden)])
        self.labelemb = nn.Embedding(3, hidden)
        self.register_buffer("one", torch.tensor([1], dtype=torch.long))
        self.register_buffer(
            "two",
            torch.tensor([2] if label == "dizo" else [1], dtype=torch.long))
        self.layerwise_label = layerwise_label

    def reset_parameters(self):
        pass

    def forward(
            self,
            real: torch.FloatTensor,
            imag: torch.FloatTensor,
            edge_index: torch.LongTensor,
            query_edges: torch.LongTensor,
            edge_weight: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        real = self.inputlin_real(real)
        # print("real", real.shape)
        num_tar = query_edges.shape[0]
        idx = torch.arange(num_tar, device=real.device)
        real = real.unsqueeze(0).repeat(num_tar, 1, 1)
        head, tail = query_edges[:, 0], query_edges[:, 1]
        if not self.no_label:
            if self.label_fn == "emb":
                real[idx, head] *= self.labelemb(self.one)
                real[idx, tail] *= self.labelemb(self.two)
            else:
                real[idx, head] = self.label_lin[0](real[idx, head])
                real[idx, tail] = self.label_lin[-1](real[idx, tail])
        for cheb in self.Chebs:
            treal = cheb(real, edge_index).clone()
            if not self.no_label and self.layerwise_label:
                if self.label_fn == "emb":
                    treal[idx, head] *= self.labelemb(self.one)
                    treal[idx, tail] *= self.labelemb(self.two)
                else:
                    treal[idx, head] = self.label_lin[0](treal[idx, head])
                    treal[idx, tail] = self.label_lin[-1](treal[idx, tail])
            real = real + treal
        if self.multpred:
            x = real[idx, head] * real[idx, tail]
        else:
            x = torch.cat((real[idx, head], real[idx, tail]), dim=-1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x