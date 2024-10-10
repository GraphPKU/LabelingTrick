# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU, 
                      Sequential, LayerNorm as BN, Dropout, Identity)
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, SAGEConv, GINConv, 
                                global_sort_pool, global_add_pool, global_mean_pool)
import pdb


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset, 
                 use_feature=False, node_embedding=None, dropout=0.5):
        super(GCN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        return x


class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset=None, 
                 use_feature=False, node_embedding=None, dropout=0.5):
        super(SAGE, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(SAGEConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        return x


# An end-to-end deep learning architecture for graph classification, AAAI-18.
class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None, 
                 dynamic_train=False, GNN=GCNConv, use_feature=False, 
                 node_embedding=None):
        super(DGCNN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                if dynamic_train:
                    sampled_train = train_dataset[:1000]
                else:
                    sampled_train = train_dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs.append(GNN(initial_channels, hidden_channels))
        for i in range(0, num_layers-1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, 
                 jk=True, train_eps=False):
        super(GIN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.jk = jk

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.conv1 = GINConv(
            Sequential(
                Linear(initial_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ),
            train_eps=train_eps)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ),
                    train_eps=train_eps))

        self.dropout = dropout
        if self.jk:
            self.lin1 = Linear(num_layers * hidden_channels, hidden_channels)
        else:
            self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.jk:
            x = global_mean_pool(torch.cat(xs, dim=1), batch)
        else:
            x = global_mean_pool(xs[-1], batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x

from torch_geometric.nn import MessagePassing
class myGINConv(MessagePassing):
    def __init__(self, mlp, aggr = "add"):
        super().__init__(aggr)
        self.mlp = mlp

    def forward(self, x, edge_index):
        """"""
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x)
        return self.mlp(out)


    def message(self, x_j):
        return x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.mlp})'

class mySAGEConv(SAGEConv):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

from functools import partial

class PlabelingZOGIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, 
                 jk=True, no_label=False, train_eps=False, aggr = "add", center_pool: bool= False, conv_dropout: bool=False):
        super().__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.jk = jk
        self.center_pool = center_pool
        self.no_label = no_label

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        conv_fn = lambda a, b: GINConv(Sequential(
                Linear(a, b),
                ReLU(inplace=True),
                Linear(b, b),
                ReLU(inplace=True),
                BN(b)
            ))
        self.conv1 = conv_fn(initial_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                conv_fn(hidden_channels, hidden_channels))
        self.lin0 = Sequential(Linear(num_layers * hidden_channels if self.jk else hidden_channels, hidden_channels), Dropout(p=dropout, inplace=True), ReLU(inplace=True))
        self.dropout = dropout
        self.lin = Sequential(Linear(hidden_channels, hidden_channels),
        Dropout(p=dropout, inplace=True), ReLU(inplace=True), Linear(hidden_channels, 1))

    def forward(self, z:torch.Tensor, edge_index, batch, x=None, edge_weight=None, node_id=None):
        if self.no_label:
            z.fill_(0)
        z_emb = self.z_embedding((torch.tensor([[1], [2]], dtype=z.dtype, device=z.device) == z.unsqueeze(0)).to(torch.long))
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float).unsqueeze(0).expand(2, -1, -1)], -1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb.unsqueeze(0).expand(2, -1, -1)], -1)
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.jk:
            x = torch.cat(xs, dim=-1)
        else:
            x = xs[-1]
        x = self.lin0(x).sum(dim=0)
        if self.center_pool:
            x = x[z>0]
            N = x.shape[0]
            assert N % 2 == 0
            x = x.reshape(N//2, 2, -1)
            x = x.prod(dim=1).sum(dim=-1)
        else:
            x = global_mean_pool(x, batch)
            x = self.lin(x)
        return x


class PlabelingDRNLGIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, 
                 jk=True, no_label=False, train_eps=False, aggr = "add", center_pool: bool= False, conv_dropout: bool=False):
        super().__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.jk = jk
        self.center_pool = center_pool
        self.no_label = no_label

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        conv_fn = lambda a, b: GINConv(Sequential(
                Linear(a, b),
                ReLU(inplace=True),
                Linear(b, b),
                ReLU(inplace=True),
                BN(b)
            ))
        self.conv1 = conv_fn(initial_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                conv_fn(hidden_channels, hidden_channels))
        self.lin0 = Sequential(Linear(num_layers * hidden_channels if self.jk else hidden_channels, hidden_channels), Dropout(p=dropout, inplace=True), ReLU(inplace=True))
        self.dropout = dropout
        self.lin = Sequential(Linear(hidden_channels, hidden_channels),
        Dropout(p=dropout, inplace=True), ReLU(inplace=True), Linear(hidden_channels, 1))

    def forward(self, z:torch.Tensor, edge_index, batch, x=None, edge_weight=None, node_id=None):
        if self.no_label:
            z.fill_(0)
        mask1 = (z==1)
        mask0 = torch.cat((mask1[:-1], torch.zeros(1, dtype=mask1.dtype, device=mask1.device)))
        z1 = z.clone()
        z1[mask1] = 0
        z1[mask0] = 1
        z_emb = self.z_embedding(torch.stack((z1, z)))
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float).unsqueeze(0).expand(2, -1, -1)], -1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb.unsqueeze(0).expand(2, -1, -1)], -1)
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.jk:
            x = torch.cat(xs, dim=-1)
        else:
            x = xs[-1]
        x = self.lin0(x).sum(dim=0)
        if self.center_pool:
            x = x[z>0]
            N = x.shape[0]
            assert N % 2 == 0
            x = x.reshape(N//2, 2, -1)
            x = x.prod(dim=1).sum(dim=-1)
        else:
            x = global_mean_pool(x, batch)
            x = self.lin(x)
        return x


class PlabelingDEGIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, 
                 jk=True, no_label=False, train_eps=False, aggr = "add", center_pool: bool= False, conv_dropout: bool=False):
        super().__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.jk = jk
        self.center_pool = center_pool
        self.no_label = no_label

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        conv_fn = lambda a, b: GINConv(Sequential(
                Linear(a, b),
                ReLU(inplace=True),
                Linear(b, b),
                ReLU(inplace=True),
                BN(b)
            ))
        self.conv1 = conv_fn(initial_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                conv_fn(hidden_channels, hidden_channels))
        self.lin0 = Sequential(Linear(num_layers * hidden_channels if self.jk else hidden_channels, hidden_channels), Dropout(p=dropout, inplace=True), ReLU(inplace=True))
        self.dropout = dropout
        self.lin = Sequential(Linear(hidden_channels, hidden_channels),
        Dropout(p=dropout, inplace=True), ReLU(inplace=True), Linear(hidden_channels, 1))

    def forward(self, z:torch.Tensor, edge_index, batch, x=None, edge_weight=None, node_id=None):
        if self.no_label:
            z.fill_(0)
        z_emb = self.z_embedding(z.t())
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float).unsqueeze(0).expand(2, -1, -1)], -1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb.unsqueeze(0).expand(2, -1, -1)], -1)
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.jk:
            x = torch.cat(xs, dim=-1)
        else:
            x = xs[-1]
        x = self.lin0(x).sum(dim=0)
        if self.center_pool:
            x = x[z>0]
            N = x.shape[0]
            assert N % 2 == 0
            x = x.reshape(N//2, 2, -1)
            x = x.prod(dim=1).sum(dim=-1)
        else:
            x = global_mean_pool(x, batch)
            x = self.lin(x)
        return x


class PSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset=None, 
                 use_feature=False, node_embedding=None, dropout=0.5):
        super().__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.dropout = dropout
        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(SAGEConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.lin0 = Sequential(Linear(hidden_channels, hidden_channels), Dropout(p=dropout, inplace=True), ReLU(inplace=True))

        self.lin = Sequential(Linear(hidden_channels, hidden_channels), Dropout(p=dropout, inplace=True), ReLU(inplace=True), Linear(hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding((torch.tensor([[1], [2]], dtype=z.dtype, device=z.device) == z.unsqueeze(0)).to(torch.long))
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float).unsqueeze(0).expand(2, -1, -1)], -1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb.unsqueeze(0).expand(2, -1, -1)], -1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.lin0(x).sum(dim=0)
        if True:  # center pooling
            x_src = x[z == 1]
            x_dst = x[z == 2]
            x = (x_src * x_dst)
            x = self.lin(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        return x
