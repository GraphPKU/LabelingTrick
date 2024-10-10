from functools import partial
from math import ceil
import torch_geometric
from torch_geometric.utils import add_self_loops, negative_sampling, degree, to_undirected
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import numpy as np
from impl import EdgePool, SegUtils
from impl import Utils
from impl.Utils import tensorDataloader, set_seed
import argparse
import torch
from torch.nn.functional import logsigmoid
import torch_geometric.nn.models as Gmodels
from torch.optim import Adam
import optuna
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.loader import DataLoader as pygDataloader
from torch_geometric.data import Data as pygData
from torch import Tensor, BoolTensor
from typing import Optional

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--test", action="store_true")
parser.add_argument("--nolabel", action="store_true")
parser.add_argument("--flabel", action="store_true")
parser.add_argument("--no_feature", action="store_true")
parser.add_argument("--repeat_batch", type=int, default=10)
parser.add_argument("--exp_label", type=str, default="")
parser.add_argument("--num_worker", type=int, default=0)
args = parser.parse_args()

dataset = PygLinkPropPredDataset(f"ogbl-{args.dataset}", "data")
data = dataset[0]
x = None if args.no_feature else data.x
edge_index = data.edge_index
device = torch.device("cuda")
x = x.to(torch.float) if x is not None else None
num_node = dataset[0].num_nodes
split_edge = None
# copied from SEAL-OGB
evaluator = Evaluator(name=f"ogbl-{args.dataset}")
evaluator.K = {"ppa": 100, "collab": 50, "ddi": 20}[args.dataset]
deg = degree(edge_index[0], num_node)


@torch.jit.script
def loss_fn(pred1: Tensor, pred0: Tensor):
    return -logsigmoid(torch.cat((pred1, -pred0))).mean()


def evaluate_hit(pred1, pred0):
    pred1 = pred1.squeeze()
    pred0 = pred0.squeeze()
    from impl.metric import auroc, ap
    print(f"loss: {loss_fn(pred1, pred0).item():.4f}")
    print("auroc: {:.4f}".format(
        auroc(torch.cat((pred1, pred0)),
              torch.cat((torch.ones_like(pred1), torch.zeros_like(pred0))))))
    print("ap: {:.4f}".format(
        ap(torch.cat((pred1, pred0)),
           torch.cat((torch.ones_like(pred1), torch.zeros_like(pred0))))))
    ret = evaluator.eval({
        'y_pred_pos': pred1,
        'y_pred_neg': pred0,
    })
    print(ret)
    return ret[f'hits@{evaluator.K}']


def split():
    global split_edge, edge_index

    def get_train_neg_edges(split_edge, num_nodes):
        if 'edge' in split_edge['train']:
            pos_edge = split_edge["train"]['edge'].t()
            neg_edge = negative_sampling(
                add_self_loops(to_undirected(pos_edge))[0],
                num_nodes=num_nodes,
                num_neg_samples=pos_edge.shape[1]).t()  #[E, 2]
        elif 'source_node' in split_edge['train']:
            source = split_edge["train"]['source_node']
            target_neg = torch.randint(0,
                                       num_nodes, [source.shape[0]],
                                       dtype=torch.long)
            neg_edge = torch.stack((source, target_neg), dim=1)  #[E, 2]
        else:
            raise NotImplementedError
        return neg_edge

    split_edge = dataset.get_edge_split()
    split_edge["train"]["edge_neg"] = get_train_neg_edges(split_edge, num_node)
    if args.dataset == "citation2":
        split_edge["train"]["edge"] = torch.stack(
            (split_edge["train"]["source_node"],
             split_edge["train"]["target_node"]),
            dim=1)
        split_edge["valid"]["edge"] = torch.stack(
            (split_edge["valid"]["source_node"],
             split_edge["valid"]["target_node"]),
            dim=1)
        split_edge["valid"]["edge_neg"] = torch.stack(
            (split_edge["valid"]["source_node"],
             split_edge["valid"]["target_node_neg"]),
            dim=1)
        split_edge["test"]["edge"] = torch.stack(
            (split_edge["test"]["source_node"],
             split_edge["test"]["target_node"]),
            dim=1)
        split_edge["test"]["edge_neg"] = torch.stack(
            (split_edge["test"]["source_node"],
             split_edge["test"]["target_node_neg"]),
            dim=1)
    for idx in ["train", "valid", "test"]:
        for term in ["edge", "edge_neg"]:
            assert split_edge[idx][term].shape[1] == 2, "wrong pos shape"


def collatepos(pos: torch.Tensor):
    '''
    pos (batch_size, 3), the third column is subgraph size
    '''
    tpos = pos[:, :2].clone()
    tpos[1:] += torch.cumsum(pos[:-1, 2], dim=0).unsqueeze(-1)
    return tpos


@torch.jit.script
def my_k_hop_subgraph(node_idx: Tensor, num_hops: int, edge_index: Tensor,
                      biedge_index: Tensor, num_nodes: int):
    '''
    copied from pyg
    '''
    row, col = biedge_index[0], biedge_index[1]

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

    edge_index = edge_index[:, node_mask[edge_index[0]]
                            & node_mask[edge_index[1]]]

    node_idx = row.new_full((num_nodes, ), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    return subset, edge_index, inv


def get_subg(x: Optional[Tensor], pos: Tensor, ei: Tensor, biei: Tensor,
             num_node: int, num_hop: int):
    ret = []
    for j in range(pos.shape[0]):
        node_idx, segei, segpos = my_k_hop_subgraph(pos[j], num_hop, ei, biei,
                                                    num_node)
        segei = SegUtils.delete_cluster(node_idx.shape[0], segpos, segei)
        ret.append(
            pygData(x[node_idx] if x is not None else node_idx, segei, None,
                    None, segpos.unsqueeze(0)).cpu())
    return ret


def get_subg_del(x: Optional[Tensor], pos: Tensor, ei: Tensor, biei: Tensor,
                 num_node: int, num_hop: int):
    ret = []
    for j in range(pos.shape[0]):
        node_idx, segei, segpos = my_k_hop_subgraph(pos[j], num_hop, ei, biei,
                                                    num_node)
        segei = SegUtils.delete_cluster(node_idx.shape[0], segpos, segei)
        tx = node_idx
        if x is not None:
            tx = x[node_idx]
            tx[pos] -= 2
        ret.append(pygData(tx, segei, None, None, segpos.unsqueeze(0)).cpu())
    return ret


def precsubgs(x: Optional[Tensor], num_node: int, hop: int, ei: Tensor,
              biei: Optional[Tensor], pos: Tensor, remove_edge: bool):
    x_cuda = x.cuda() if x is not None else None
    ei_cuda = ei.cuda()
    biei_cuda = biei.cuda() if biei is not None else ei_cuda
    pos_cuda = pos.cuda()
    if remove_edge:
        return get_subg_del(x_cuda, pos_cuda, ei_cuda, biei_cuda, num_node,
                            hop)
    else:
        return get_subg(x_cuda, pos_cuda, ei_cuda, biei_cuda, num_node, hop)


from torch_geometric.data import Batch


def batch2pred(x_cuda: Tensor, batch: Batch, mod: nn.Module):
    batch_cuda = batch.batch.cuda(non_blocking=True)
    tei = batch.edge_index.cuda(non_blocking=True)
    tpos = collatepos(batch.pos.cuda())
    tx = batch.x.cuda(non_blocking=True)
    if x_cuda is not None:
        tx = x_cuda[tx]
    z = Utils.idx2mask(tpos.flatten(), tx.shape[0]).to(torch.long)
    pred = mod(z, tei, batch_cuda, tx)
    return pred


@torch.no_grad()
def test_pre(mod, x: Tensor, ei: Tensor, biei: Tensor, hop: int,
             pos: torch.Tensor, batchsize: int):
    mod.eval()
    datalist = precsubgs(None, num_node, hop, ei, biei, pos, True)
    dataloader1 = pygDataloader(SegUtils.SubGDataset(datalist),
                                batchsize,
                                shuffle=False,
                                num_workers=args.num_worker,
                                persistent_workers=args.num_worker > 0)
    preds = []
    x_cuda = x.cuda(non_blocking=True) if x is not None else None
    for batch in dataloader1:
        pred = batch2pred(x_cuda, batch, mod)
        preds.append(pred.cpu())
    return torch.cat(preds)


@torch.no_grad()
def test_full(mod: nn.Module, x: Tensor, ei: Tensor, biei: Optional[Tensor],
              hop: int, pos_positive: Tensor, pos_negative: Tensor,
              full_batchsize: int, batchsize: int):
    mod.eval()
    num_positive, num_negative = pos_positive.shape[0], pos_negative.shape[0]
    pos = torch.cat((pos_positive, pos_negative))
    pred = torch.cat([
        test_pre(mod, x, ei, biei, hop, pos[i:i + full_batchsize], batchsize)
        for i in range(0, pos.shape[0], full_batchsize)
    ])
    return evaluate_hit(pred[:num_positive], pred[num_positive:])


def train(mod: nn.Module, opt: torch.optim.Optimizer, x: Tensor, ei: Tensor,
          biei: Optional[Tensor], hop: int, repeat: int,
          pos_positive: torch.Tensor, pos_negative: torch.Tensor,
          batchsize: int):
    num_positive, num_negative = pos_positive.shape[0], pos_negative.shape[0]
    datalist = precsubgs(None, num_node, hop, ei, biei,
                         torch.cat((pos_positive, pos_negative)), True)
    tdataset = SegUtils.SubGDataset(datalist)
    tdataset.set_y(
        torch.cat((torch.ones(num_positive, dtype=torch.bool),
                   torch.zeros(num_negative, dtype=torch.bool))))
    tdataloader = pygDataloader(tdataset,
                                batchsize,
                                shuffle=True,
                                num_workers=args.num_worker,
                                persistent_workers=args.num_worker > 0)
    mod.train()
    losslist = []
    x_cuda = x.cuda(non_blocking=True) if x is not None else None
    for _ in range(repeat):
        for batch in tdataloader:
            opt.zero_grad()
            mask = batch.y.cuda(non_blocking=True)
            pred = batch2pred(x_cuda, batch, mod)
            loss = loss_fn(pred[mask], pred[torch.logical_not(mask)])
            loss.backward()
            opt.step()
            losslist.append(loss.item())
    return np.average(losslist)


def buildModel(hid_dim: int, num_mplayer: int, labelmod0: str, labelmod1: str,
               **kwargs):
    import torch.nn.functional as F
    from torch_geometric.nn.glob import global_mean_pool

    class GIN(torch.nn.Module):

        def __init__(self,
                     hidden_channels,
                     num_layers,
                     dim_feature,
                     max_z=100,
                     use_feature=False,
                     dropout=0.5,
                     jk=True):
            super(GIN, self).__init__()
            self.use_feature = use_feature
            self.z_embedding = nn.Embedding(max_z, hidden_channels)
            self.jk = jk

            initial_channels = hidden_channels
            if self.use_feature:
                initial_channels += dim_feature
            self.conv1 = GINConv(
                nn.Sequential(nn.Linear(initial_channels, hidden_channels),
                              nn.Dropout(p=dropout, inplace=True),
                              nn.ReLU(inplace=True),
                              nn.Linear(hidden_channels, hidden_channels),
                              nn.LayerNorm(hidden_channels),
                              nn.Dropout(p=dropout, inplace=True),
                              nn.ReLU(inplace=True)
                              ))
            self.convs = torch.nn.ModuleList()
            for i in range(num_layers - 1):
                self.convs.append(
                    GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_channels, hidden_channels),
                            nn.Dropout(p=dropout, inplace=True),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_channels, hidden_channels),
                            nn.LayerNorm(hidden_channels),
                            nn.Dropout(p=dropout, inplace=True),
                            nn.ReLU(inplace=True)
                            )))

            self.lin = nn.Sequential(
                nn.Linear(
                    num_layers *
                    hidden_channels if self.jk else hidden_channels,
                    hidden_channels), nn.Dropout(dropout, inplace=True),
                nn.ReLU(inplace=True), nn.Linear(hidden_channels, 1))

        def forward(self, z, edge_index, batch, x=None):
            z_emb = self.z_embedding(z)
            if self.use_feature:
                x = torch.cat([z_emb, x.to(torch.float)], 1)
            else:
                x = z_emb
            x = self.conv1(x, edge_index)
            xs = [x]
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                xs += [x]
            if self.jk:
                x = global_mean_pool(torch.cat(xs, dim=1), batch)
            else:
                x = global_mean_pool(xs[-1], batch)
            x = self.lin(x)
            return x

    return GIN(256, 3, x.shape[1], 100, False, dropout=0.5, jk=True)


def work(num_step: int = 100000,
         max_early_stop: int = 4000,
         full_batch_size: int = 1024,
         batch_size: int = 128,
         lr: float = 1e-4,
         wd: float = 0,
         test_interval: int = 400,
         do_test: int = False,
         hop: int = 1,
         **kwargs):
    trn_posidx_dl = iter(
        tensorDataloader([split_edge["train"]["edge"]], full_batch_size, True,
                         True, torch.device("cpu")))
    trn_negpos_dl = iter(
        tensorDataloader([split_edge["train"]["edge_neg"]], full_batch_size,
                         True, True, torch.device("cpu")))
    mod = buildModel(**kwargs).to(device)
    opt = Adam(mod.parameters(), lr, weight_decay=wd)
    best_val_score = 0.0
    tst_score = 0.0
    early_stop = 0
    best_iter = 0
    for iteration in range(num_step):
        pos_positive = trn_posidx_dl.get_batch(full_batch_size)
        if pos_positive is None:
            trn_posidx_dl = iter(trn_posidx_dl)
            pos_positive = trn_posidx_dl.get_batch(full_batch_size)
            assert pos_positive is not None, "something wrong with tensorDataloader"
        pos_negative = trn_negpos_dl.get_batch(full_batch_size)
        if pos_negative is None:
            trn_negpos_dl = iter(trn_negpos_dl)
            pos_negative = trn_negpos_dl.get_batch(full_batch_size)
            assert pos_negative is not None, "something wrong with tensorDataloader"
        loss = train(mod, opt, x, edge_index, None, hop, args.repeat_batch,
                     pos_positive, pos_negative, batch_size)
        print(f"iteration {iteration} trn : {loss:.4f} ", flush=True)
        torch.save(mod.state_dict(),
                   f"model/{args.dataset}.{args.exp_label}.{iteration}.pt")
        if iteration % test_interval == 0 and iteration > 0:
            val_score = test_full(mod, x, edge_index, None, hop,
                                  split_edge["valid"]["edge"],
                                  split_edge["valid"]["edge_neg"],
                                  full_batch_size, batch_size)
            print(f"val {val_score:.4f}")
            early_stop += 1
            if val_score > best_val_score:
                best_iter = iteration
                early_stop = 0
                best_val_score = val_score
        if early_stop > max_early_stop:
            break
    print("best_iter: ", best_iter)
    tst_score = None
    if do_test:
        print("begin test")
        mod.load_state_dict(
            torch.load(f"model/{args.dataset}.{args.exp_label}.{best_iter}.pt",
                       map_location=torch.device("cpu")))
        mod = mod.cuda()
        tst_score = test_full(mod, x, edge_index, None, hop,
                              split_edge["test"]["edge"],
                              split_edge["test"]["edge_neg"], full_batch_size,
                              batch_size)
        print("tst score ", tst_score)
    return best_val_score, tst_score


fixed_params = {"labelmod0": "Layerwise", "labelmod1": "Linear", "jk": "last"}
'''
{
    "labelmod0": "Layerwise",
    "pool": "nn",
    "labelmod1": "Linear",
    "num_mplayer": 3
}
'''


def search(trial: optuna.Trial):
    num_mplayer = trial.suggest_int("num_mplayer", 1, 3)
    hid_dim = trial.suggest_int("hid_dim", 32, 64, step=32)
    dp1 = trial.suggest_float("dp1", 0.0, 0.9, step=0.1)
    dp2 = trial.suggest_float("dp2", 0.0, 0.9, step=0.1)
    dp0 = trial.suggest_float("dp0", 0.0, 0.9, step=0.1)
    batch_size = 256  #trial.suggest_int("batch_size", 128, 2048, 128)
    ln1 = trial.suggest_categorical("ln1", [True, False])
    ln2 = trial.suggest_categorical("ln2", [True, False])
    return work(**fixed_params,
                num_mplayer=num_mplayer,
                hid_dim=hid_dim,
                batch_size=batch_size,
                dp1=dp1,
                dp2=dp2,
                dp0=dp0,
                ln1=ln1,
                ln2=ln2,
                num_step=200,
                max_early_stop=20,
                test_interval=5)[0]


print(args)
if args.test:
    from besthp import besthp
    p = fixed_params
    p.update(besthp[args.dataset])
    p["full_batch_size"] = 4096
    p["batch_size"] = 256
    p["num_mplayer"] = 1
    print(p)
    #p["batch_size"] = 256 # 256 for collab, 32 for ddi, 16 for ppa
    summary = []
    split()
    interval = ceil(
        ceil(split_edge["valid"]["edge"].shape[0] / p["full_batch_size"]) /
        args.repeat_batch) * 2
    print("test interval", interval)
    for rep in range(10):
        set_seed(rep)
        ret = work(**p,
                   do_test=True,
                   test_interval=interval,
                   max_early_stop=10,
                   num_step=200 * interval)
        summary.append(ret[1])
    summary = np.array(summary)
    print("p: ", p)
    print("s: ", summary)
    print("ss: ", summary.mean(axis=0), summary.std(axis=0))
else:
    split()
    study = optuna.create_study(f"sqlite:///out/{args.dataset}.db",
                                study_name=args.dataset,
                                direction="maximize",
                                load_if_exists=True)
    study.optimize(search, 200)
