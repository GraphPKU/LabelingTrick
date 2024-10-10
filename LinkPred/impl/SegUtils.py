from torch import Tensor
import torch.nn as nn
import torch_sparse
from .Utils import idx2mask
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data as pygData
from torch_geometric.data import InMemoryDataset as pygDataset
from typing import List, Optional

class SubGDataset(pygDataset):

    def __init__(self, datalist: List[pygData]):
        super().__init__()
        self.data, self.slices = self.collate(datalist)
        self.data.pos = torch.cat((self.data.pos, torch.diff(self.slices["x"]).reshape(-1, 1)), dim=-1)
    
    def set_y(self, y: Tensor):
        self.data.y = y
        self.slices["y"] = torch.arange(y.shape[0]+1)
    

@torch.jit.script
def pos2subgroot(pos: Tensor):
    idx0 = torch.arange(pos.shape[0],
                        device=pos.device).unsqueeze(-1).repeat(1,
                                                                2).flatten()
    idx1 = pos.flatten()
    return torch.stack((idx0, idx1), dim=0)


@torch.jit.script
def pos2segpos(pos: Tensor, num_node: int):
    num_subg = pos.shape[0]
    offset = torch.arange(num_subg, device=pos.device, dtype=torch.long)
    pos = pos + offset.unsqueeze(1) * num_node
    return pos


#@torch.jit.script
def rootedsubg(edge_index: Tensor, num_node: int, num_subg: int,
               subgroot: Tensor, hop: int):
    '''
    subg of shape (2, E), pos[0] means subg number ,pos[1] meams node number
    '''
    adj_t = SparseTensor.from_edge_index(
        edge_index,
        sparse_sizes=(num_node, num_node),
    ).device_as(subgroot)
    a = torch.zeros((num_node, num_subg),
                    dtype=torch.float,
                    device=subgroot.device)
    a[subgroot[1], subgroot[0]] = 1
    for _ in range(hop):
        a += adj_t @ a
    return (a > 0.5).t()


@torch.jit.script
def get_offset(nums: Tensor, oneoffset: int):
    cumnum = torch.cumsum(nums, dim=0)
    total_num = cumnum[-1]
    ret = torch.zeros(total_num, device=nums.device, dtype=torch.long)
    ret[cumnum[cumnum < total_num]] = oneoffset
    return torch.cumsum(ret, 0)


@torch.jit.script
def expand(src: Tensor, dim: int, rate: int):
    src = src.unsqueeze(dim)
    shapelist = list(src.shape)
    shapelist[dim] = rate
    src = src.expand(shapelist)
    shapelist.pop(dim)
    shapelist[dim] *= rate
    src = src.reshape(shapelist)
    return src


@torch.jit.script
def subgmasks2g(subgmasks: Tensor, ei: Tensor):
    '''
    subgmasks (num_subg, |V|)
    ei (2, |E|)
    ea (|E|)
    '''
    edgemask = subgmasks[:, ei[0]] & subgmasks[:, ei[1]]
    num_subg_edges = torch.sum(edgemask, dim=1)
    num_subg, num_node = subgmasks.shape[0], subgmasks.shape[1]
    '''
    edgeidx = mask2idx(edgemask.flatten()) % ei.shape[1]
    retei = ei[:, edgeidx]
    '''
    retei = torch.cat([ei[:, edgemask[i]] for i in range(num_subg)], dim=-1)
    offset = get_offset(num_subg_edges, num_node)
    retei += offset.unsqueeze(0)
    return retei


@torch.jit.script
def subgmasks2g_single(subgmasks: Tensor, ei: Tensor, pos: Tensor):
    '''
    subgmasks (|V|)
    ei (2, |E|)
    ea (|E|)
    pos (2)
    '''
    segei = ei[:, subgmasks[ei[0]] & subgmasks[ei[1]]]
    segei = torch.cat([segei.flatten(), pos])
    idx_rev, segei = torch.unique(segei, return_inverse=True)
    segpos = segei[-2:]
    segei = segei[:-2].reshape(2, -1)
    segei = torch.cat(
        (segei,
         torch.arange(idx_rev.shape[0], device=segei.device,
                      dtype=segei.dtype).unsqueeze(0).expand(2, -1)),
        dim=1)
    return segei, segpos, idx_rev

@torch.jit.script
def subgmasks2g_single2(subgnodes: Tensor, adj_t: SparseTensor, pos: Tensor):
    '''
    subgnode (|v|)
    ei (2, |E|)
    ea (|E|)
    pos (2)
    '''
    pos = torch.cat((pos, subgnodes))
    idx_rev, segpos = torch.unique(pos, return_inverse=True)
    segpos = segpos[:2]
    tadj = torch_sparse.index_select(
        torch_sparse.index_select(adj_t, 0, idx_rev), 1, idx_rev)
    row, col, _ = tadj.coo()  #??
    segei = torch.stack((row, col), dim=0)
    return segei, segpos, idx_rev


@torch.jit.script
def renumberg(subgmasks: Tensor, segei: Tensor, otherindex: Tensor):
    n_subg, n_node = subgmasks.shape[0], subgmasks.shape[1]
    n_subg_nodes = torch.sum(subgmasks, dim=1)
    offset = get_offset(n_subg_nodes, n_node)
    # selfcircle防止出现有个结点没有边，不出现在ei，进而不出现在子图中
    selfcirc = torch.arange(n_node, dtype=torch.long, device=subgmasks.device)
    selfcirc = expand(selfcirc, 0, n_subg)
    selfcirc = selfcirc[subgmasks.flatten()]
    selfcirc += offset
    selfcirc = selfcirc.unsqueeze(0).expand(2, -1)
    segei = torch.cat([segei, selfcirc], dim=1)
    othershape = otherindex.shape
    ei_len = segei.numel()
    raw_idx = torch.cat((segei.flatten(), otherindex.flatten()), dim=0)
    idx_rev, transformed_idx = torch.unique(raw_idx, return_inverse=True)
    segei = transformed_idx[:ei_len].reshape(2, -1)
    otherindex = transformed_idx[ei_len:].reshape(othershape)
    # 可以删除最后的几条边进而删除selfcirc，但是没必要
    return segei, idx_rev, otherindex


'''
num_subg = pos.shape[0]
offset = num_node * torch.arange(num_subg, dtype=torch.long, device=pos.device)
offset = offset.reshape(-1, 1).expand(-1, 2)
pos += offset
'''


@torch.jit.script
def delete_cluster(segnum_node: int, pos: Tensor, segei: Tensor):
    '''
    delete all edges in a cluster
    pos (dE, 2) directed edge
    segei (2, |E|)
    '''
    pos = pos.flatten()
    mask0 = idx2mask(pos, segnum_node)
    mask0 = mask0[segei]
    mask = mask0[0] & mask0[1]
    return segei[:, torch.logical_not(mask)]



def rootedsubg_adj(adj_t: SparseTensor, num_node: int, num_subg: int, subgroot: torch.Tensor,
                   hop: int):
    '''
    subg of shape (2, E), pos[0] means subg number ,pos[1] meams node number
    '''
    '''
    # spspmm based implementation, has bug
    a = SparseTensor(row=subgroot[1],
                     col=subgroot[0],
                     sparse_sizes=(num_node, num_subg)).device_as(subgroot)
    for _ in range(hop):
        a = adj_t @ a
    rowptr, col, _ = a.csc()
    return rowptr, col
    '''
    a = torch.zeros((num_node, num_subg),
                    dtype=torch.float,
                    device=subgroot.device)
    a[subgroot[1], subgroot[0]] = 1
    for _ in range(hop):
        a = adj_t @ a
    a = SparseTensor.from_dense(a.t(), has_value=False)
    rowptr, col, _ = a.csr()
    return rowptr, col

@torch.jit.script
def rootedsubg_single(adj_t: SparseTensor, pos: torch.Tensor, hop: int):
    '''
    subg of shape (2, E), pos[0] means subg number ,pos[1] meams node number
    '''
    ret=pos
    for _ in range(hop):
        ret = torch.unique(torch_sparse.index_select(adj_t, 0, ret).csr()[1])
    return ret

def get_subg(x: torch.Tensor, num_node: int, adj_t: SparseTensor, ei: torch.Tensor, boosted_deg: torch.Tensor, hop: int, pos: torch.Tensor, y: Optional[torch.Tensor]):
    subgroot = pos2subgroot(pos)
    subgptr, subg = rootedsubg_adj(adj_t, num_node, pos.shape[0], subgroot,
                                   hop)
    datalist: List[pygData] = []
    for i in range(pos.shape[0]):
        #segei, segpos, idx_rev = SegUtils.subgmasks2g_single(subgmask[i], ei, pos[i])
        segei, segpos, idx_rev = subgmasks2g_single2(
            subg[subgptr[i]:subgptr[i + 1]], adj_t, pos[i])
        t_deg = boosted_deg[idx_rev].clone()  ## 不应修改boosted_deg
        if x is None:
            tx = t_deg - 1
        else:
            tx = x[idx_rev]
        segea = 1 / torch.sqrt(
            (t_deg[segei[0]] * t_deg[segei[1]]).to(torch.float)) 
        datalist.append(
            pygData(x=tx,
                    edge_index=segei,
                    edge_attr=segea,
                    pos=torch.cat((segpos,
                                   torch.tensor([tx.shape[0]], device=segpos.device))).unsqueeze(0),
                    y=y[i] if y is not None else None).cpu())
    return datalist


def get_subg2(x: torch.Tensor, num_node: int, adj_t: SparseTensor, ei: torch.Tensor, boosted_deg: torch.Tensor, hop: int, pos: torch.Tensor, y: Optional[torch.Tensor]):
    datalist: List[pygData] = []
    for i in range(pos.shape[0]):
        segei, segpos, idx_rev = subgmasks2g_single2(
            rootedsubg_single(adj_t, pos[i], hop), adj_t, pos[i])
        t_deg = boosted_deg[idx_rev].clone()  ## 不应修改boosted_deg
        if x is None:
            tx = t_deg - 1
        else:
            tx = x[idx_rev]
        segea = 1 / torch.sqrt(
            (t_deg[segei[0]] * t_deg[segei[1]]).to(torch.float)) 
        datalist.append(
            pygData(tx,
                    segei,
                    segea,
                    y[i] if y is not None else None,
                    torch.cat((segpos, torch.tensor([tx.shape[0]], device=segpos.device))
                    ).unsqueeze(0)).cpu())
    return datalist

def get_subg_removeedge(x: torch.Tensor, num_node: int, adj_t: SparseTensor, ei:torch.Tensor, boosted_deg: torch.Tensor, hop: int, pos: torch.Tensor, y: Optional[torch.Tensor]):
    subgroot = pos2subgroot(pos)
    subgptr, subg = rootedsubg_adj(adj_t, num_node, pos.shape[0], subgroot,
                                   hop)
    datalist: List[pygData] = []
    for i in range(pos.shape[0]):
        segei, segpos, idx_rev = subgmasks2g_single2(
            subg[subgptr[i]:subgptr[i + 1]], adj_t, pos[i])
        t_deg = boosted_deg[idx_rev].clone()  ## 不应修改boosted_deg
        t_deg[segpos.flatten()] -= 2  # change deg
        if x is None:
            tx = t_deg - 1
        else:
            tx = x[idx_rev]
        segei = delete_cluster(idx_rev.shape[0], segpos, segei)
        segea = 1 / torch.sqrt(
            (t_deg[segei[0]] * t_deg[segei[1]]).to(torch.float)).cpu() 
        datalist.append(
            pygData(x=tx,
                    edge_index=segei,
                    edge_attr=segea,
                    pos=torch.cat((segpos,
                                   torch.tensor([tx.shape[0]], device=segpos.device))).unsqueeze(0),
                    y=y[i] if y is not None else None).cpu())
    return datalist


def precomputersubgs(x: Optional[torch.Tensor], num_node: int, y: Optional[torch.tensor],
                     adj_t: SparseTensor, ei: torch.Tensor, boosted_deg: torch.Tensor,
                     pos: torch.Tensor, remove_edge: bool, batch_size: int,
                     hop: int):
    adj_t_cuda = adj_t.cuda()
    x_cuda = x.cuda() if x is not None else None
    ei_cuda = ei  #ei.cuda()
    pos_cuda = pos.cuda()
    boosted_deg_cuda = boosted_deg.cuda()
    size = (pos.shape[0] + batch_size - 1) // batch_size
    datalist = []
    if remove_edge:
        for i in range(size):
            datalist += get_subg_removeedge(
                x_cuda, num_node, adj_t_cuda, ei_cuda, boosted_deg_cuda, hop,
                pos_cuda[i * batch_size:i * batch_size + batch_size], y)
    else:
        for i in range(size):
            datalist += get_subg(
                x_cuda, num_node, adj_t_cuda, ei_cuda, boosted_deg_cuda, hop,
                pos_cuda[i * batch_size:i * batch_size + batch_size], y)
    return datalist