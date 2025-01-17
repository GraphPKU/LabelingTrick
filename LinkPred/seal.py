# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import os, sys
from shutil import copy
import random
import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import to_undirected
import optuna
from Dataset import load_dataset
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import train_test_split_edges
import warnings
from scipy.sparse import SparseEfficiencyWarning
from torch_geometric.utils import add_self_loops, negative_sampling
from scipy.sparse.csgraph import shortest_path
from torch import Tensor
from torch.nn.functional import logsigmoid

warnings.simplefilter('ignore', SparseEfficiencyWarning)


@torch.jit.script
def loss_fn(pred1: Tensor, pred0: Tensor):
    return -logsigmoid(torch.cat((pred1, -pred0))).mean()


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            neg_edge = negative_sampling(add_self_loops(edge_index)[0],
                                         num_nodes=num_nodes,
                                         num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0,
                                       num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[
            perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack(
            [source.repeat_interleave(neg_per_target),
             target_neg.view(-1)])
    return pos_edge, neg_edge


def construct_pyg_graph(node_ids,
                        adj,
                        dists,
                        node_features,
                        y,
                        node_label='drnl'):
    def drnl_node_labeling(adj, src, dst):
        # Double Radius Node Labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)

        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst,
                                 directed=False,
                                 unweighted=True,
                                 indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src,
                                 directed=False,
                                 unweighted=True,
                                 indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = dist // 2, dist % 2

        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1.
        z[dst] = 1.
        z[torch.isnan(z)] = 0.

        return z.to(torch.long)

    def di_drnl_node_labeling(adj, src, dst):
        # Double Radius Node Labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)

        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst,
                                 directed=False,
                                 unweighted=True,
                                 indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src,
                                 directed=False,
                                 unweighted=True,
                                 indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = dist // 2, dist % 2

        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 0.
        z[dst] = 1.
        z[torch.isnan(z)] = 0.

        return z.to(torch.long)

    def de_node_labeling(adj, src, dst, max_dist=3):
        # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More
        # Powerful Neural Networks for Graph Representation Learning."
        src, dst = (dst, src) if src > dst else (src, dst)

        dist = shortest_path(adj,
                             directed=False,
                             unweighted=True,
                             indices=[src, dst])
        dist = torch.from_numpy(dist)

        dist[dist > max_dist] = max_dist
        dist[torch.isnan(dist)] = max_dist + 1

        return dist.to(torch.long).t()

    def dide_node_labeling(adj, src, dst, max_dist=3):
        # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More
        # Powerful Neural Networks for Graph Representation Learning."
        src, dst = (dst, src) if src > dst else (src, dst)

        dist1 = shortest_path(adj,
                             directed=False,
                             unweighted=True,
                             indices=[src])
        dist2 = shortest_path(adj,
                             directed=False,
                             unweighted=True,
                             indices=[dst])
        dist1 = torch.from_numpy(dist1).flatten()
        dist2 = torch.from_numpy(dist2).flatten()
        dist1[dist1 > max_dist] = max_dist
        dist2[dist2 > max_dist] = max_dist
        dist1[torch.isnan(dist1)] = max_dist + 1
        dist2[torch.isnan(dist2)] = max_dist + 1
        dist = torch.stack((dist1, dist2), dim=-1)
        return dist.to(torch.long)

    def de_plus_node_labeling(adj, src, dst, max_dist=100):
        # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
        # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
        src, dst = (dst, src) if src > dst else (src, dst)

        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst,
                                 directed=False,
                                 unweighted=True,
                                 indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src,
                                 directed=False,
                                 unweighted=True,
                                 indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
        dist[dist > max_dist] = max_dist
        dist[torch.isnan(dist)] = max_dist + 1

        return dist.to(torch.long)

    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == 'didrnl':  # DRNL
        z = di_drnl_node_labeling(adj, 0, 1)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists) == 0).to(torch.long)
    elif node_label == 'dizo':  # zero-one labeling trick
        z = torch.zeros_like(node_ids)
        z[0] = 1
        z[1] = 2
    elif node_label == 'dide':  # distance encoding
        z = dide_node_labeling(adj, 0, 1)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z > 100] = 100  # limit the maximum label to 100
    elif node_label == 'no':  # this is technically not a valid labeling trick
        z = torch.zeros_like(node_ids)
    else:
        raise NotImplementedError
    data = Data(node_features,
                edge_index,
                edge_weight=edge_weight,
                y=y,
                z=z,
                node_id=node_ids,
                num_nodes=num_nodes)
    return data


def k_hop_subgraph(src,
                   dst,
                   num_hops,
                   A,
                   sample_ratio=1.0,
                   max_nodes_per_hop=None,
                   node_features=None,
                   y=1,
                   directed=False,
                   A_csc=None):
    def neighbors(fringe, A, outgoing=True):
        if outgoing:
            res = set(A[list(fringe)].indices)
        else:
            res = set(A[:, list(fringe)].indices)
        return res

    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops + 1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = sorted(fringe)
            fringe = random.sample(fringe, int(sample_ratio * len(fringe)))
            fringe = set(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = sorted(fringe)
                fringe = random.sample(fringe, max_nodes_per_hop)
                fringe = set(fringe)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]
    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0
    if node_features is not None:
        node_features = node_features[nodes]
    return nodes, subgraph, dists, node_features, y


def extract_enclosing_subgraphs(link_index,
                                A,
                                x,
                                y,
                                num_hops,
                                node_label='drnl',
                                ratio_per_hop=1.0,
                                max_nodes_per_hop=None,
                                directed=False,
                                A_csc=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in link_index.t().tolist():
        tmp = k_hop_subgraph(src,
                             dst,
                             num_hops,
                             A,
                             ratio_per_hop,
                             max_nodes_per_hop,
                             node_features=x,
                             y=y,
                             directed=directed,
                             A_csc=A_csc)
        data = construct_pyg_graph(*tmp, node_label)
        data_list.append(data)

    return data_list

class SEALDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data,
                 split_edge,
                 num_hops,
                 percent=100,
                 split='train',
                 use_coalesce=False,
                 node_label='drnl',
                 ratio_per_hop=1.0,
                 max_nodes_per_hop=None,
                 directed=False):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data_{}'.format(self.split, self.node_label)
        else:
            name = 'SEAL_{}_data_{}_{}'.format(self.split, self.node_label,
                                               self.percent)
        name += '.pt'
        return [name]

    def process(self):
        print("begin precompute", flush=True)
        pos_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes))

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None

        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(pos_edge, A, self.data.x, 1,
                                               self.num_hops, self.node_label,
                                               self.ratio_per_hop,
                                               self.max_nodes_per_hop,
                                               self.directed, A_csc)
        neg_list = extract_enclosing_subgraphs(neg_edge, A, self.data.x, 0,
                                               self.num_hops, self.node_label,
                                               self.ratio_per_hop,
                                               self.max_nodes_per_hop,
                                               self.directed, A_csc)

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        print("end precompute", flush=True)
        del pos_list, neg_list


class SEALDynamicDataset(Dataset):
    def __init__(self,
                 root,
                 data,
                 split_edge,
                 num_hops,
                 percent=100,
                 split='train',
                 use_coalesce=False,
                 node_label='drnl',
                 ratio_per_hop=1.0,
                 max_nodes_per_hop=None,
                 directed=False,
                 **kwargs):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDynamicDataset, self).__init__(root)

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes))
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None

    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        tmp = k_hop_subgraph(src,
                             dst,
                             self.num_hops,
                             self.A,
                             self.ratio_per_hop,
                             self.max_nodes_per_hop,
                             node_features=self.data.x,
                             y=y,
                             directed=self.directed,
                             A_csc=self.A_csc)
        data = construct_pyg_graph(*tmp, self.node_label)

        return data


def train():
    model.train()

    total_loss = 0
    pbar = train_loader
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight,
                       node_id)
        mask = (data.y > 0.5).flatten()
        loss = loss_fn(logits[mask], logits[torch.logical_not(mask)])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)


@torch.no_grad()
def test():
    model.eval()

    y_pred, y_true = [], []
    for data in val_loader:
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight,
                       node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true == 1]
    neg_val_pred = val_pred[val_true == 0]
    print("val_loss:", loss_fn(pos_val_pred, neg_val_pred))

    y_pred, y_true = [], []
    for data in test_loader:
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight,
                       node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true == 1]
    neg_test_pred = test_pred[test_true == 0]
    print("test_loss:", loss_fn(pos_test_pred, neg_test_pred))
    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred,
                                neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred,
                               neg_test_pred)
    elif args.eval_metric == 'auc':
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results


@torch.no_grad()
def test_multiple_models(models):
    for m in models:
        m.eval()
    y_pred, y_true = [[] for _ in range(len(models))
                      ], [[] for _ in range(len(models))]
    for data in test_loader:
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, m in enumerate(models):
            logits = m(data.z, data.edge_index, data.batch, x, edge_weight,
                       node_id)
            y_pred[i].append(logits.view(-1).cpu())
            y_true[i].append(data.y.view(-1).cpu().to(torch.float))
    test_pred = [torch.cat(y_pred[i]) for i in range(len(models))]
    test_true = [torch.cat(y_true[i]) for i in range(len(models))]
    pos_test_pred = [
        test_pred[i][test_true[i] == 1] for i in range(len(models))
    ]
    neg_test_pred = [
        test_pred[i][test_true[i] == 0] for i in range(len(models))
    ]

    Results = []
    for i in range(len(models)):
        if args.eval_metric == 'hits':
            Results.append(
                evaluate_hits_test(pos_test_pred[i], neg_test_pred[i]))
        elif args.eval_metric == 'mrr':
            Results.append(
                evaluate_mrr_test(pos_test_pred[i], neg_test_pred[i]))
        elif args.eval_metric == 'auc':
            Results.append(
                evaluate_auc(test_pred[i], test_pred[i]))
    return Results


def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results

def evaluate_hits_test(pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = test_hits
    return results

def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)

    return results

def evaluate_mrr_test(pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = test_mrr

    return results

def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results


# Data settings
parser = argparse.ArgumentParser(description='OGBL (SEAL)')
parser.add_argument('--dataset', type=str, default='ogbl-collab')
parser.add_argument(
    '--fast_split',
    action='store_true',
    help="for large custom datasets (not OGB), do a fast data split")
# GNN settings
parser.add_argument('--model', type=str, default='DGCNN')
parser.add_argument('--no_label', action="store_true")
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
# Subgraph extraction settings
parser.add_argument('--num_hops', type=int, default=1)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label',
                    type=str,
                    default='drnl',
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature',
                    action='store_true',
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_edge_weight',
                    action='store_true',
                    help="whether to consider edge weight in GNN")
# Training settings
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--train_percent', type=float, default=100)
parser.add_argument('--val_percent', type=float, default=100)
parser.add_argument('--test_percent', type=float, default=100)
parser.add_argument('--dynamic_train',
                    action='store_true',
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--dynamic_val', action='store_true')
parser.add_argument('--dynamic_test', action='store_true')
parser.add_argument(
    '--num_workers',
    type=int,
    default=16,
    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument(
    '--train_node_embedding',
    action='store_true',
    help="also train free-parameter node embeddings together with GNN")
parser.add_argument(
    '--pretrained_node_embedding',
    type=str,
    default=None,
    help="load pretrained node embeddings as additional node features")
# Testing settings
parser.add_argument('--use_valedges_as_input', action='store_true')
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--data_appendix',
                    type=str,
                    default='',
                    help="an appendix to the data directory")
parser.add_argument('--save_appendix',
                    type=str,
                    default='',
                    help="an appendix to the save directory")
parser.add_argument('--keep_old',
                    action='store_true',
                    help="do not overwrite old files in the save directory")
parser.add_argument('--continue_from',
                    type=int,
                    default=None,
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--only_test',
                    action='store_true',
                    help="only test without training")
parser.add_argument('--save_run', type=int, default=1)
parser.add_argument('--save_epoch', type=int, default=10)
parser.add_argument('--test_multiple_models',
                    action='store_true',
                    help="test multiple models together")
parser.add_argument('--use_heuristic',
                    type=str,
                    default=None,
                    help="test a link prediction heuristic (CN or AA)")
args = parser.parse_args()

def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset[0]
    random.seed(234)
    torch.manual_seed(234)

    data = train_test_split_edges(data, val_ratio, test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


print(args)
if args.dataset[:4] == "ogbl":
    dataset = PygLinkPropPredDataset(name=args.dataset, root="data")
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    evaluator = Evaluator(name=args.dataset)
elif args.dataset in [
        "USAir", "NS", "PB", "Yeast", "Router", "Wikipedia", "PB", "Power",
        "Celegans", "Ecoli", "arxiv"
]:
    data = load_dataset(args.dataset)
    split_edge = do_edge_split([data.clone()])

if args.dataset.startswith('ogbl-citation'):
    args.eval_metric = 'mrr'
    directed = True
elif args.dataset.startswith('ogbl'):
    args.eval_metric = 'hits'
    directed = False
else:
    args.eval_metric = "auc"
    directed = False
if args.use_valedges_as_input:
    val_edge_index = split_edge['valid']['edge'].t()
    if not directed:
        val_edge_index = to_undirected(val_edge_index)
    data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
    val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
    data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.use_heuristic:
    # Test link prediction heuristics.
    num_nodes = data.num_nodes
    if 'edge_weight' in data:
        edge_weight = data.edge_weight.view(-1)
    else:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])),
                       shape=(num_nodes, num_nodes))

    pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge,
                                                   data.edge_index,
                                                   data.num_nodes)
    pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge,
                                                     data.edge_index,
                                                     data.num_nodes)
    pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge)
    neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge)
    pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge)
    neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge)

    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred,
                                neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred,
                               neg_test_pred)
    elif args.eval_metric == 'auc':
        val_pred = torch.cat([pos_val_pred, neg_val_pred])
        val_true = torch.cat([
            torch.ones(pos_val_pred.size(0), dtype=int),
            torch.zeros(neg_val_pred.size(0), dtype=int)
        ])
        test_pred = torch.cat([pos_test_pred, neg_test_pred])
        test_true = torch.cat([
            torch.ones(pos_test_pred.size(0), dtype=int),
            torch.zeros(neg_test_pred.size(0), dtype=int)
        ])
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    print(results)
    exit()

# SEAL.
path = args.dataset + '_seal{}'.format(args.data_appendix)
use_coalesce = True if args.dataset == 'ogbl-collab' else False
if not args.dynamic_train and not args.dynamic_val and not args.dynamic_test:
    args.num_workers = 0

dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALDataset'
train_dataset = eval(dataset_class)(
    path,
    data,
    split_edge,
    num_hops=args.num_hops,
    percent=args.train_percent,
    split='train',
    use_coalesce=use_coalesce,
    node_label=args.node_label,
    ratio_per_hop=args.ratio_per_hop,
    max_nodes_per_hop=args.max_nodes_per_hop,
    directed=directed,
)

dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
val_dataset = eval(dataset_class)(
    path,
    data,
    split_edge,
    num_hops=args.num_hops,
    percent=args.val_percent,
    split='valid',
    use_coalesce=use_coalesce,
    node_label=args.node_label,
    ratio_per_hop=args.ratio_per_hop,
    max_nodes_per_hop=args.max_nodes_per_hop,
    directed=directed,
)
dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
test_dataset = eval(dataset_class)(
    path,
    data,
    split_edge,
    num_hops=args.num_hops,
    percent=args.test_percent,
    split='test',
    use_coalesce=use_coalesce,
    node_label=args.node_label,
    ratio_per_hop=args.ratio_per_hop,
    max_nodes_per_hop=args.max_nodes_per_hop,
    directed=directed,
)

max_z = 1000  # set a large max_z so that every z has embeddings to look up

train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers)
val_loader = DataLoader(val_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         num_workers=args.num_workers)

if args.train_node_embedding:
    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
elif args.pretrained_node_embedding:
    weight = torch.load(args.pretrained_node_embedding)
    emb = torch.nn.Embedding.from_pretrained(weight)
    emb.weight.requires_grad = False
else:
    emb = None


for run in range(4, args.runs):
    from seal_models import *
    if args.model == 'DGCNN':
        model = DGCNN(args.hidden_channels,
                      args.num_layers,
                      max_z,
                      args.sortpool_k,
                      train_dataset,
                      args.dynamic_train,
                      use_feature=args.use_feature,
                      node_embedding=emb).to(device)
    elif args.model == 'SAGE':
        model = SAGE(args.hidden_channels,
                     args.num_layers,
                     max_z,
                     train_dataset,
                     args.use_feature,
                     node_embedding=emb).to(device)
    elif args.model == 'GCN':
        model = GCN(args.hidden_channels,
                    args.num_layers,
                    max_z,
                    train_dataset,
                    args.use_feature,
                    node_embedding=emb).to(device)
    elif args.model == 'GIN':
        model = GIN(args.hidden_channels,
                    args.num_layers,
                    max_z,
                    train_dataset,
                    args.use_feature,
                    node_embedding=emb).to(device)
    elif args.model == 'PGIN':
        if args.node_label == "didrnl":
            model = PlabelingDRNLGIN(args.hidden_channels,
                                args.num_layers,
                                max_z,
                                train_dataset,
                                args.use_feature,
                                node_embedding=emb,
                                no_label=args.no_label).to(device)
        elif args.node_label == "dizo":
            model = PlabelingZOGIN(args.hidden_channels,
                                args.num_layers,
                                max_z,
                                train_dataset,
                                args.use_feature,
                                node_embedding=emb,
                                no_label=args.no_label).to(device)
        elif args.node_label == "dide":
            model = PlabelingDEGIN(args.hidden_channels,
                                args.num_layers,
                                max_z,
                                train_dataset,
                                args.use_feature,
                                node_embedding=emb,
                                no_label=args.no_label).to(device)
        else:
            raise NotImplementedError
    elif args.model == 'PGIN_dp':
        model = PlabelingZOGIN(args.hidden_channels,
                               args.num_layers,
                               max_z,
                               train_dataset,
                               args.use_feature,
                               node_embedding=emb,
                               conv_dropout=True).to(device)
    elif args.model == 'PGIN_center':
        model = PlabelingZOGIN(args.hidden_channels,
                               args.num_layers,
                               max_z,
                               train_dataset,
                               args.use_feature,
                               node_embedding=emb,
                               center_pool=True).to(device)
    elif args.model == 'PSAGE':
        model = PSAGE(args.hidden_channels,
                      args.num_layers,
                      max_z,
                      train_dataset,
                      args.use_feature,
                      node_embedding=emb).to(device)
    start_epoch = 1
    try:
        os.mkdir(f"model/{args.dataset}/")
    except FileExistsError:
        pass
    model_name = os.path.join(
        f"model/{args.dataset}/",
        f'{args.model}_{args.node_label}_run{args.save_run}_model_checkpoint{args.save_epoch}.pth'
    )
    if args.only_test:
        model.load_state_dict(torch.load(model_name, map_location="cpu"))
        results = test()
        print(results)
        exit()
    if args.continue_from is not None:
        model.load_state_dict(torch.load(model_name, map_location="cpu"))
        print("load_from", model_name, flush=True)
        start_epoch = args.save_run + 1
    print(model)
    model = model.to(device)
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    if args.model == 'DGCNN':
        print(f'SortPooling k is set to {model.k}')

    # Training starts
    print("begin train", flush=True)
    tar_K = {"ogbl-ppa": 100, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": "mrr"}.get(args.dataset, "auc")
    best_val = 0
    tst = 0
    best_epoch = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss = train()

        if epoch % args.eval_steps == 0:
            model_name = os.path.join(
                f"model/{args.dataset}",
                f'{args.model}_{args.node_label}_run{run + 1}_model_checkpoint{epoch}.pth')
            torch.save(model.state_dict(), model_name)
            results = test()
            if epoch % args.log_steps == 0:
                for key, result in results.items():
                    valid_res, test_res = result
                    if str(tar_K) in key:
                        if valid_res > best_val:
                            best_val = valid_res
                            tst = test_res
                            best_epoch = epoch
                    to_print = (
                        f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                        f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                        f'Test: {100 * test_res:.2f}%')
                    print(key)
                    print(to_print, flush=True)
    print(f"best: epoch *{best_epoch}* val ${best_val}$ tst  #{tst}#")

print(f'Total number of parameters is {total_params}')
