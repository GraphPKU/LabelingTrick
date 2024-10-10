from math import ceil
from torch_geometric.utils import add_self_loops, negative_sampling, degree, to_undirected
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import numpy as np
from torch_sparse import SparseTensor
import torch_sparse
from impl import EdgePool, SegUtils
from impl.Utils import tensorDataloader, set_seed
import argparse
import torch
from torch.nn.functional import logsigmoid
import torch_geometric.nn.models as Gmodels
from impl import segmodel, ModUtils, model
from torch.optim import Adam
import optuna
import torch.nn as nn
from copy import deepcopy
from torch_geometric.loader import DataLoader as pygDataloader


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--test", action="store_true")
parser.add_argument("--nolabel", action="store_true")
parser.add_argument("--flabel", action="store_true")
parser.add_argument("--no_feature", action="store_true")
parser.add_argument("--repeat_batch", type=int, default=10)
parser.add_argument("--exp_label", type=str, default="")
parser.add_argument("--num_worker", type=int, default=1)
args = parser.parse_args()

dataset = PygLinkPropPredDataset(f"ogbl-{args.dataset}", "data")
data = dataset[0]
x = None if args.no_feature else data.x
edge_index = data.edge_index
device = torch.device("cuda")
x = x.to(torch.float) if x is not None else None
num_node = dataset[0].num_nodes
split_edge = None
adj_t = SparseTensor.from_edge_index(to_undirected(edge_index),
                                     sparse_sizes=(num_node, num_node))
adj_t = torch_sparse.set_diag(adj_t)  #原来没有set_diag
adj_t = adj_t.coalesce()
boosted_deg = adj_t.sum(dim=1)
# copied from SEAL-OGB
evaluator = Evaluator(name=f"ogbl-{args.dataset}")
evaluator.K = {"ppa": 100, "collab": 50, "ddi": 20}[args.dataset]
def evaluate_hit(pred1, pred0):
    return evaluator.eval({
        'y_pred_pos': pred1,
        'y_pred_neg': pred0,
    })[f'hits@{evaluator.K}']


def split():
    global split_edge, edge_index

    def get_train_neg_edges(split_edge, num_nodes):
        if 'edge' in split_edge['train']:
            pos_edge = split_edge["train"]['edge'].t()
            neg_edge = negative_sampling(
                add_self_loops(pos_edge)[0],
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
    pos[1:, :2] += torch.cumsum(pos[:-1, 2], dim=0).unsqueeze(-1)
    return pos[:, :2]


def train(mod: nn.Module, opt: Adam, x: torch.Tensor, ei: torch.Tensor,
          boosted_deg: torch.Tensor, hop: int, repeat: int,
          pos_positive: torch.Tensor, pos_negative: torch.Tensor,
          batchsize: int, pre_batchsize: int):
    num_positive, num_negative = pos_positive.shape[0], pos_negative.shape[0]
    datalist = SegUtils.precomputersubgs(x, num_node, torch.ones(num_positive, dtype=torch.bool),
                                ei, boosted_deg, pos_positive, True,
                                pre_batchsize, hop)
    datalist += SegUtils.precomputersubgs(x, num_node, torch.zeros(num_negative,
                                                dtype=torch.bool), ei,
                                 boosted_deg, pos_negative, False,
                                 pre_batchsize, hop)
    dataloader = pygDataloader(SegUtils.SubGDataset(datalist),
                               batchsize,
                               shuffle=True,
                               num_workers=args.num_worker,
                               persistent_workers=args.num_worker>0,
                               pin_memory=True)
    mod.train()
    losslist = []
    for rep in range(repeat):
        for batch in dataloader:
            opt.zero_grad()
            adj = SparseTensor(row=batch.edge_index[0],
                               col=batch.edge_index[1],
                               value=batch.edge_attr,
                               sparse_sizes=(batch.x.shape[0],
                                             batch.x.shape[0])).cuda()
            pos = collatepos(batch.pos.cuda())
            pred = mod(batch.x.cuda(), adj, pos)
            mask = batch.y.to(device)
            loss = -torch.cat((logsigmoid(
                pred[mask]), logsigmoid(-pred[torch.logical_not(mask)])),
                              dim=0).mean()
            loss.backward()
            opt.step()
            losslist.append(loss.item())
    del dataloader, datalist
    return np.average(losslist)


@torch.no_grad()
def test_pre(mod, x, ei, boosted_deg, hop: int, pos: torch.Tensor,
             batchsize: int, pre_batchsize: int):
    datalist = SegUtils.precomputersubgs(x, num_node, None, ei, boosted_deg, pos, False,
                                pre_batchsize, hop)
    dataloader1 = pygDataloader(SegUtils.SubGDataset(datalist),
                                batchsize,
                                shuffle=False,
                                num_workers=args.num_worker,
                                persistent_workers=args.num_worker>0,
                                pin_memory=True)
    preds = []
    for batch in dataloader1:
        adj = SparseTensor(row=batch.edge_index[0],
                           col=batch.edge_index[1],
                           value=batch.edge_attr,
                           sparse_sizes=(batch.x.shape[0],
                                         batch.x.shape[0])).cuda()
        pos = collatepos(batch.pos.cuda())
        preds.append(mod(batch.x.cuda(), adj, pos))
    del dataloader1, datalist
    return torch.cat(preds).cpu()


def test_full(mod, x, ei, boosted_deg, hop: int, pos_positive: torch.Tensor,
              pos_negative: torch.Tensor, full_batchsize: int, batchsize: int,
              pre_batchsize: int):
    pred0 = torch.cat([
        test_pre(mod, x, ei, boosted_deg, hop,
                 pos_negative[i:i + full_batchsize], batchsize, pre_batchsize)
        for i in range(0, pos_negative.shape[0], full_batchsize)
    ])
    pred1 = torch.cat([
        test_pre(mod, x, ei, boosted_deg, hop,
                 pos_positive[i:i + full_batchsize], batchsize, pre_batchsize)
        for i in range(0, pos_positive.shape[0], full_batchsize)
    ])
    return evaluate_hit(pred1, pred0)


def buildModel(hid_dim: int, num_mplayer: int, labelmod0: str, labelmod1: str,
               **kwargs):
    if args.nolabel:
        labelmod0 = "None"
    if args.flabel:
        labelmod0 = "Full"
    assert labelmod0 in ["None", "Bottom", "Layerwise", "Full"]
    assert labelmod1 in ["Bias", "Linear"]
    emb_dim = hid_dim if x is None else -1
    PredMod = EdgePool.TwoSetPool(hid_dim, kwargs["dp2"], kwargs["ln2"])
    head_act = nn.Identity() if emb_dim == hid_dim else nn.Sequential(
        nn.Linear(x.shape[1], hid_dim),
        nn.Dropout(p=kwargs["dp0"], inplace=True), nn.ReLU(inplace=True))
    if labelmod0 == "None":
        PredMod = EdgePool.EdgeInnerProduct(hid_dim, "nn", kwargs["ln2"])
        convs = nn.ModuleList([
            model.OnlyConv(mlp=nn.Sequential(
                ModUtils.BiasMod(hid_dim) if labelmod1 ==
                "Bias" else nn.Linear(hid_dim, hid_dim),
                nn.Dropout(kwargs["dp1"], inplace=True), nn.ReLU(
                    inplace=True), nn.Linear(hid_dim, hid_dim),
                nn.LayerNorm(hid_dim, elementwise_affine=False
                             ) if kwargs["ln1"] else nn.Identity(),
                nn.Dropout(kwargs["dp1"], inplace=True), nn.ReLU(
                    inplace=True))) for i in range(num_mplayer)
        ])
        RepreMod = segmodel.NoLabelingNet(convs)
    elif labelmod0 == "Bottom":
        convs = nn.ModuleList([
            Gmodels.GCN(hid_dim,
                        hid_dim,
                        num_mplayer,
                        hid_dim,
                        dropout=kwargs["dp1"],
                        act=nn.ReLU(inplace=True),
                        norm=nn.LayerNorm(hid_dim) if kwargs["ln1"] else None,
                        jk=kwargs["jk"])
        ])
        if labelmod1 == "Bias":
            f0s = nn.ModuleList([ModUtils.BiasMod(hid_dim)])
            f1s = nn.ModuleList([ModUtils.BiasMod(hid_dim)])
        elif labelmod1 == "Linear":
            f0s = nn.ModuleList([nn.Linear(hid_dim, hid_dim)])
            f1s = nn.ModuleList([nn.Linear(hid_dim, hid_dim)])
        RepreMod = segmodel.PLabelingNet2Set(convs, f0s, f1s)
    elif labelmod0 == "Layerwise":
        convs = nn.ModuleList([
            model.OnlyConv(mlp=nn.Sequential(
                nn.LayerNorm(hid_dim, elementwise_affine=False
                             ) if kwargs["ln1"] else nn.Identity(),
                nn.Dropout(kwargs["dp1"], inplace=True), nn.ReLU(
                    inplace=True))) for i in range(num_mplayer)
        ])
        if labelmod1 == "Bias":
            f0s = nn.ModuleList(
                [ModUtils.BiasMod(hid_dim) for i in range(num_mplayer)])
            f1s = nn.ModuleList(
                [ModUtils.BiasMod(hid_dim) for i in range(num_mplayer)])
        elif labelmod1 == "Linear":
            f0s = nn.ModuleList(
                [nn.Linear(hid_dim, hid_dim) for i in range(num_mplayer)])
            f1s = nn.ModuleList(
                [nn.Linear(hid_dim, hid_dim) for i in range(num_mplayer)])
        RepreMod = segmodel.PLabelingNet2Set(convs, f0s, f1s)
    elif labelmod0 == "Full":
        convs = nn.ModuleList([
            model.OnlyConv(mlp=nn.Sequential(
                nn.Dropout(kwargs["dp1"], inplace=True), nn.ReLU(
                    inplace=True), nn.Linear(hid_dim, hid_dim),
                nn.LayerNorm(hid_dim, elementwise_affine=False
                             ) if kwargs["ln1"] else nn.Identity(),
                nn.Dropout(kwargs["dp1"], inplace=True), nn.ReLU(
                    inplace=True))) for i in range(num_mplayer)
        ])
        if labelmod1 == "Bias":
            f0s = nn.ModuleList(
                [ModUtils.BiasMod(hid_dim) for i in range(num_mplayer)])
            f1s = nn.ModuleList(
                [ModUtils.BiasMod(hid_dim) for i in range(num_mplayer)])
        elif labelmod1 == "Linear":
            f0s = nn.ModuleList(
                [nn.Linear(hid_dim, hid_dim) for i in range(num_mplayer)])
            f1s = nn.ModuleList(
                [nn.Linear(hid_dim, hid_dim) for i in range(num_mplayer)])
        RepreMod = segmodel.FLabelingNet(convs, f0s, f1s)
    return segmodel.FullModel(torch.max(boosted_deg), num_node, head_act,
                              RepreMod, PredMod, emb_dim)


def work(num_step: int = 100000,
         max_early_stop: int = 4000,
         full_batch_size: int = 1024,
         batch_size: int = 128,
         pre_batch_size: int = 128,
         lr: float = 3e-4,
         wd: float = 0,
         test_interval: int = 400,
         do_test: int = False,
         **kwargs):
    hop = kwargs["num_mplayer"]
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
    best_mod = None
    for iteration in range(num_step):
        if iteration % test_interval == 0 and iteration > 0:
            val_score = test_full(mod, x, edge_index, boosted_deg, hop,
                                  split_edge["valid"]["edge"],
                                  split_edge["valid"]["edge_neg"],
                                  full_batch_size, batch_size, pre_batch_size)
            print(f"val {val_score:.4f}")
            early_stop += 1
            if val_score > best_val_score:
                early_stop = 0
                best_val_score = val_score
                best_mod = deepcopy(mod).cpu()
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
        loss = train(mod, opt, x, edge_index, boosted_deg, hop,
                     args.repeat_batch, pos_positive, pos_negative, batch_size,
                     pre_batch_size)
        print(f"iteration {iteration} trn : {loss:.4f} ", flush=True)
        torch.save(mod, f"model/{args.dataset}.{args.exp_label}.{iteration}.pt")
        if early_stop > max_early_stop:
            break
    print("", flush=True)
    tst_score = None
    if do_test:
        print("begin test")
        torch.save(best_mod, f"model/{args.dataset}.{args.exp_label}.best.pt")
        tst_score = test_full(mod, x, edge_index, boosted_deg, hop,
                              split_edge["valid"]["edge"],
                              split_edge["valid"]["edge_neg"], full_batch_size,
                              batch_size, pre_batch_size)
        print("tst score ", tst_score)
    return best_val_score, tst_score


fixed_params = {
    "labelmod0": "Layerwise",
    "pool": "nn",
    "jk": "last",
    "labelmod1": "Linear",
    "num_mplayer": 3
}


def search(trial: optuna.Trial):
    num_mplayer = trial.suggest_int("num_mplayer", 1, 3)
    hid_dim = trial.suggest_int("hid_dim", 32, 32, step=32)
    dp1 = trial.suggest_float("dp1", 0.0, 0.9, step=0.1)
    dp2 = trial.suggest_float("dp2", 0.0, 0.9, step=0.1)
    dp0 = trial.suggest_float("dp0", 0.0, 0.9, step=0.1)
    batch_size = trial.suggest_int("batch_size", 128, 2048, 128)
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
                num_step=2000,
                max_early_stop=200,
                test_interval=20)[0]

print(args)
if args.test:
    from besthp import besthp
    p = fixed_params
    p.update(besthp[args.dataset])
    p["full_batch_size"] = 2048
    p["pre_batch_size"] = 128
    p["batch_size"] = 128
    print(p)
    #p["batch_size"] = 256 # 256 for collab, 32 for ddi, 16 for ppa
    summary = []
    split()
    interval = ceil(
        ceil(split_edge["valid"]["edge"].shape[0] / p["full_batch_size"]) /
        args.repeat_batch)
    print("test interval", interval)
    for rep in range(10):
        set_seed(rep)
        ret = work(**p,
                   do_test=True,
                   test_interval=interval,
                   max_early_stop=3.01 * interval,
                   num_step=50 * interval)
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
