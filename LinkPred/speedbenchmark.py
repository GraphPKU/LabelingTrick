import time
from numpy import arange
from impl import Utils
from torch_geometric.utils import to_undirected, add_self_loops, coalesce
from torch_geometric.utils import k_hop_subgraph, is_undirected, degree
import torch
from importlib import reload
from impl import SegUtils
from ogb.linkproppred import PygLinkPropPredDataset
from torch_sparse import SparseTensor
from torch_geometric.data import Data as pygData
import torch_sparse
from impl import SegUtils
from torch import Tensor, BoolTensor

@torch.jit.script
def my_k_hop_subgraph(node_idx: Tensor, num_hops: int, edge_index: Tensor, num_nodes: int):
    '''
    copied from pyg
    '''
    row, col = edge_index[0], edge_index[1]

    node_mask: BoolTensor = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask: BoolTensor = row.new_empty(row.size(0), dtype=torch.bool)
    node_idx = node_idx
    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(0)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.unique(torch.cat(subsets), return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(0)
    node_mask[subset] = True

    edge_index = edge_index[:, node_mask[row] & node_mask[col]]

    node_idx = row.new_full((num_nodes, ), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx[edge_index]

    return subset, edge_index, inv

@torch.jit.script
def work(pos_cuda: Tensor, boosted_ei: Tensor, boosted_deg: Tensor, num_node: int, num_hop: int):
        xs = []
        eis = []
        eas = []
        poss = []
        for j in range(pos_cuda.shape[0]):
            node_idx, segei, segpos = my_k_hop_subgraph(pos_cuda[j], num_hop, boosted_ei, num_node)
            t_deg = boosted_deg[node_idx].clone()  ## 不应修改boosted_deg
            tx = t_deg - 1
            segea = 1 / torch.sqrt(
                    (t_deg[segei[0]] * t_deg[segei[1]]).to(torch.float)) 
            xs.append(tx.cpu())
            eis.append(segei.cpu())
            eas.append(segea.cpu())
            poss.append(segpos.cpu())
        xlens = torch.cumsum(torch.tensor([0]+[x.shape[0] for x in xs]), dim=0)
        eilens = torch.cumsum(torch.tensor([0] + [ei.shape[1] for ei in eis]), dim=0)
        ealens = eilens
        poslens = torch.arange(pos_cuda.shape[0]+1)
        x = torch.cat(xs)
        ei = torch.cat(eis, dim=1)
        ea = torch.cat(eas)
        pos = torch.cat((torch.stack(poss), xlens[1:].unsqueeze(-1)), dim=-1)
        return x, ei, ea,pos, xlens, eilens, ealens, poslens


def work_nojit(pos_cuda: Tensor, boosted_ei: Tensor, boosted_deg: Tensor, num_node: int, num_hop: int):
        ret = []
        for j in range(pos_cuda.shape[0]):
            node_idx, segei, segpos = my_k_hop_subgraph(pos_cuda[j], num_hop, boosted_ei, num_node)
            t_deg = boosted_deg[node_idx].clone()  ## 不应修改boosted_deg
            tx = t_deg - 1
            segea = 1 / torch.sqrt(
                    (t_deg[segei[0]] * t_deg[segei[1]]).to(torch.float))
            ret.append(pygData(tx, segei, segea, None, segpos)) 
        return ret


dataset = PygLinkPropPredDataset(f"ogbl-collab", "data")
data = dataset[0]
device = torch.device("cuda")
edge_index = data.edge_index.to(device)
boosted_ei = add_self_loops(edge_index)[0]
num_node = dataset[0].num_nodes
split_edge = dataset.get_edge_split()
adj_t = SparseTensor.from_edge_index(to_undirected(edge_index),
                                     sparse_sizes=(num_node, num_node))
adj_t = torch_sparse.set_diag(adj_t)  #原来没有set_diag
boosted_deg = adj_t.sum(dim=1)
pos = split_edge["valid"]["edge_neg"][:2048]
torch.cuda.synchronize(device)
t1 = time.time()
adj_t_cuda = adj_t.cuda()
boosted_deg_cuda = boosted_deg.cuda()
pos_cuda = pos.cuda()

num_hop = 2

torch.cuda.synchronize(device)
t1 = time.time()
y = None
for i in range(100):
    dal = work_nojit(pos_cuda, boosted_ei, boosted_deg_cuda, num_node, num_hop)
    SegUtils.SubGDataset(dal)
torch.cuda.synchronize(device)
t2 = time.time()
print(t2-t1, flush=True)


torch.cuda.synchronize(device)
t1 = time.time()
y = None
for i in range(100):
    work(pos_cuda, boosted_ei, boosted_deg_cuda, num_node, num_hop)
torch.cuda.synchronize(device)
t2 = time.time()
print(t2-t1, flush=True)



boosted_ei_cpu = boosted_ei.cpu()
torch.cuda.synchronize(device)
t1 = time.time()
y = None
for i in range(100):
    work(pos, boosted_ei_cpu, boosted_deg, num_node, num_hop)
torch.cuda.synchronize(device)
t2 = time.time()
print(t2-t1, flush=True)

for i in range(100):
    SegUtils.get_subg(None, num_node, adj_t_cuda, edge_index, boosted_deg_cuda,
    num_hop, pos_cuda, None)
torch.cuda.synchronize(device)
t2 = time.time()
print(t2-t1, flush=True)

torch.cuda.synchronize(device)
t1 = time.time()
for i in range(100):
    SegUtils.get_subg2(None, num_node, adj_t_cuda, edge_index, boosted_deg_cuda,
    num_hop, pos_cuda, None)
torch.cuda.synchronize(device)
t2 = time.time()