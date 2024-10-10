import functools
from torch_geometric.utils import is_undirected, add_self_loops, negative_sampling, degree
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import numpy as np
from torch_sparse import SparseTensor
from impl import EdgePool, Utils, SegUtils
from impl.Utils import to_directed, edge2ds, tensorDataloader, set_seed
import argparse
import torch
from torch.nn.functional import logsigmoid
import torch_geometric.nn.models as Gmodels
from impl import segmodel, ModUtils, model
from torch.optim import Adam
import optuna
import torch.nn as nn
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--test", action="store_true")
parser.add_argument("--nolabel", action="store_true")
parser.add_argument("--flabel", action="store_true")
args = parser.parse_args()

dataset = PygLinkPropPredDataset(f"ogbl-{args.dataset}", "data")
data = dataset[0]
x = data.x
edge_index = data.edge_index
device = torch.device("cuda")
edge_index = edge_index.to(device)
x = x.to(device).to(torch.float) if x is not None else None
num_node = x.shape[0] if x is not None else torch.max(edge_index).item() + 1
split_edge = None
is_directed = not is_undirected(edge_index)
dss = None

evaluator = Evaluator(name=f"ogbl-{args.dataset}")


# copied from SEAL-OGB
def evaluate_hits(pred, y, K):
    pred = pred.flatten()
    mask = y > 0.5
    pos_val_pred = pred[mask]
    neg_val_pred = pred[torch.logical_not(mask)]
    evaluator.K = K
    valid_hits = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })[f'hits@{K}']

    return valid_hits

score_fn = functools.partial(evaluate_hits,
                             K={
                                 "ppa": 100,
                                 "collab": 50,
                                 "ddi": 20
                             }[args.dataset])


def split():
    global split_edge, dss, edge_index

    def get_train_neg_edges(split_edge, num_nodes):
        if 'edge' in split_edge['train']:
            pos_edge = split_edge["train"]['edge'].t()
            neg_edge = negative_sampling(
                add_self_loops(pos_edge)[0],
                num_nodes=num_nodes,
                num_neg_samples=pos_edge.shape[1]).t()  #[2, E]
        elif 'source_node' in split_edge['train']:
            source = split_edge["train"]['source_node']
            target_neg = torch.randint(0,
                                       num_nodes, [source.shape[0]],
                                       dtype=torch.long)
            neg_edge = torch.stack((source, target_neg), dim=1)  #[2, E]
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
    # split_edges of shape (E, 2)
    if not is_directed:
        edge_index = torch.cat(
            (split_edge["train"]["edge"], split_edge["train"]["edge"][:,
                                                                      [1, 0]]),
            dim=-1).reshape(-1, 2).t()
    for stage in ["train", "valid", "test"]:
        for term in ["edge", "edge_neg"]:
            split_edge[stage][term] = split_edge[stage][term].t()
    dss = [
        edge2ds(split_edge[stage]["edge"], split_edge[stage]["edge_neg"])
        for stage in ["valid", "test"]
    ]


def train(mod, x, ei, boosted_deg, hop:int, opt, pos_positive, pos_negative):
    mod.train()
    opt.zero_grad()
    pos = torch.cat((pos_positive, pos_negative), dim=0)
    num_positive = pos_positive.shape[0]
    subgroot = SegUtils.pos2subgroot(pos)
    subgmask = SegUtils.rootedsubg(ei, num_node, pos.shape[0], subgroot, hop)
    segei = SegUtils.subgmasks2g(subgmask, ei)
    segei, idx_rev, segpos = SegUtils.renumberg(subgmask, segei, SegUtils.pos2segpos(pos, num_node))
    seg_num_node = idx_rev.shape[0]
    node_idx = idx_rev % num_node
    t_deg = boosted_deg[node_idx] ## 不应修改boosted_deg
    d_deg = degree(segpos[:num_positive].flatten(), seg_num_node, torch.long)#仅考虑undirected情况
    t_deg -= d_deg
    if x is None:
        x = t_deg - 1
    else:
        x = x[node_idx]
    segei = SegUtils.delete_cluster(seg_num_node, segpos[: num_positive], segei)
    segea = (t_deg[segei[0]] * t_deg[segei[1]]).to(torch.float)
    segea = 1/torch.sqrt(segea)
    adj = SparseTensor(row=segei[0], col=segei[1], value=segea, sparse_sizes=(seg_num_node, seg_num_node))
    pred = mod(x, adj, segpos)
    loss_positive = -logsigmoid(pred[:num_positive])
    loss_negative = -logsigmoid(-pred[num_positive:])
    loss = torch.cat((loss_positive, loss_negative), dim=0).mean()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def test(mod, x, ei, boosted_deg, hop: int, dl: Utils.tensorDataloader, score_fns, limit: int = -1):
    mod.eval()
    preds = []
    ys = []
    for i, batch in enumerate(dl):
        if i >= limit and limit >= 0:
            break
        pos, y = batch
        subgroot = SegUtils.pos2subgroot(pos)
        subgmask = SegUtils.rootedsubg(ei, num_node, pos.shape[0], subgroot, hop)
        segei = SegUtils.subgmasks2g(subgmask, ei)
        segei, idx_rev, segpos = SegUtils.renumberg(subgmask, segei, SegUtils.pos2segpos(pos, num_node))
        seg_num_node = idx_rev.shape[0]
        node_idx = idx_rev % num_node
        t_deg = boosted_deg[node_idx]
        segea = (t_deg[segei[0]] * t_deg[segei[1]]).to(torch.float)
        segea = 1/torch.sqrt(segea)
        adj  = SparseTensor(row=segei[0], col=segei[1], value=segea, sparse_sizes=(seg_num_node, seg_num_node))
        preds.append(mod(t_deg-1 if x is None else x[node_idx] , adj, segpos))
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return tuple((score_fn(pred, y) for score_fn in score_fns))


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
    tx = degree(edge_index[0], num_node,
                dtype=torch.long) if x is None else None
    max_x = -1 if tx is None else torch.max(tx).item()
    return segmodel.FullModel(max_x, num_node, head_act, RepreMod, PredMod,
                           emb_dim)


def work(num_step: int = 100000,
         max_early_stop: int = 4000,
         batch_size: int = 128,
         lr: float = 3e-4,
         wd: float = 0,
         test_interval: int = 400,
         do_test: int = False,
         **kwargs):
    hop = kwargs["num_mplayer"]
    val_dl, tst_dl = (tensorDataloader((pair[0].t(), pair[1]), batch_size * 2,
                                       False, True, device)
                      for id, pair in enumerate(dss))
    trn_posidx_dl = iter(
        tensorDataloader([torch.arange(split_edge["train"]["edge"].shape[1])],
                         batch_size, True, True, device))
    trn_negpos_dl = iter(
        tensorDataloader([split_edge["train"]["edge_neg"].t()], batch_size,
                         True, True, device))
    mod = buildModel(**kwargs).to(device)
    opt = Adam(mod.parameters(), lr, weight_decay=wd)
    best_val_score = 0.0
    tst_score = 0.0
    early_stop = 0
    full_deg = degree(edge_index[0], num_node,
                    torch.long).to(device)
    boosted_deg = full_deg.to(device) + 1
    full_ei = edge_index.to(device)
    full_x = x
    split_edge["train"]["edge"] = split_edge["train"]["edge"].to(device)
    best_mod = None
    for iteration in range(num_step):
        if iteration % test_interval == 0 and iteration > 0:
            val_score = test(mod, full_x, full_ei, boosted_deg, hop, val_dl, [score_fn],
                             test_interval)[0]
            print(f"val {val_score:.4f}")
            early_stop += 1
            if val_score > best_val_score:
                torch.save(mod, f"model/{args.dataset}.{iteration}.pt")
                early_stop = 0
                best_val_score = val_score
                best_mod = deepcopy(mod).cpu()
        idx = trn_posidx_dl.get_batch(batch_size)
        if idx is None:
            trn_posidx_dl = iter(trn_posidx_dl)
            idx = trn_posidx_dl.get_batch(batch_size)
            assert idx is not None, "something wrong with tensorDataloader"
        mask = Utils.idx2mask(idx, split_edge["train"]["edge"].shape[1])
        pos_positive = split_edge["train"]["edge"][:, mask].t()
        pos_negative = trn_negpos_dl.get_batch(batch_size)
        if pos_negative is None:
            trn_negpos_dl = iter(trn_negpos_dl)
            pos_negative = trn_negpos_dl.get_batch(batch_size)
            assert pos_negative is not None, "something wrong with tensorDataloader"
        loss = train(mod, full_x, full_ei, boosted_deg, hop, opt, pos_positive, pos_negative)
        print(f"iteration {iteration} trn : {loss:.4f} ")
        if early_stop > max_early_stop:
            break
    print("", flush=True)
    tst_score = None
    if do_test:
        print("begin test")
        torch.save(best_mod, f"model/{args.dataset}.pt")
        tst_score = test(best_mod.to(device), full_x, full_ei, boosted_deg, hop, tst_dl,
                         [score_fn])[0]
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


if args.test:
    from besthp import besthp
    p = fixed_params
    p.update(besthp[args.dataset])
    print(p)
    #p["batch_size"] = 256 # 256 for collab, 32 for ddi, 16 for ppa
    summary = []
    split()
    interval = dss[0][0].shape[1]//2//p["batch_size"]
    print("test interval", interval)
    for rep in range(10):
        set_seed(rep)
        ret = work(**p, do_test=True, test_interval=interval, max_early_stop=3.01*interval, num_step=50*interval)
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
