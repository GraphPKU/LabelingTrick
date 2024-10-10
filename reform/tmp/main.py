import numpy as np
import random
from Dataset import load_dataset
from impl import EdgePool, Utils
from impl.Utils import do_edge_split, to_directed, edge2ds, tensorDataloader
import argparse
import torch
from torch.nn.functional import logsigmoid
import torch_geometric.nn.models as Gmodels
from impl import model, metric, ModUtils
from torch.optim import Adam
import optuna
import torch.nn as nn
from torch_scatter import scatter_add
from time import time
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--test", action="store_true")
parser.add_argument("--nolabel", action="store_true")
parser.add_argument("--flabel", action="store_true")
parser.add_argument("--twoset", action="store_true")
args = parser.parse_args()

dat = load_dataset(args.dataset)
x = dat.x
num_node = torch.max(dat.edge_index).item() + 1
device = torch.device("cuda")
full_ei = dat.edge_index.clone()
split_edge, ei, dss = None, None, None


def set_seed(seed: int):
    print("seed ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split():
    global ei, split_edge, dss
    dat.edge_index = full_ei
    split_edge = do_edge_split(dat)
    ei = split_edge["train"]["edge"].clone()
    split_edge["train"]["edge"] = to_directed(ei).to(device)
    split_edge["train"]["edge_neg"] = to_directed(
        split_edge["train"]
        ["edge_neg"])[:, :split_edge["train"]["edge"].shape[1]].to(device)
    dss = [
        edge2ds(split_edge[stage]["edge"], split_edge[stage]["edge_neg"])
        for stage in ["train", "valid", "test"]
    ]


def train(mod, x, adj, opt, pos_positive, pos_negative):
    mod.train()
    opt.zero_grad()
    pos = torch.cat((pos_positive, pos_negative), dim=0)
    num_positive = pos_positive.shape[0]
    pred = mod(x, pos, adj).flatten()
    loss_positive = -logsigmoid(pred[:num_positive])
    loss_negative = -logsigmoid(-pred[num_positive:])
    loss = torch.cat((loss_positive, loss_negative), dim=0).mean()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def test(mod, x, adj, dl: Utils.tensorDataloader, score_fns):
    mod.eval()
    preds = []
    ys = []
    for batch in dl:
        preds.append(mod(x, *batch[:-1], adj))
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return tuple((score_fn(pred, y) for score_fn in score_fns))


def buildModel(hid_dim: int, num_mplayer: int, labelmod0: str, labelmod1: str,
               **kwargs):
    assert labelmod0 in ["None", "Bottom", "Layerwise", "Full"]
    assert labelmod1 in ["Bias", "Linear"]
    emb_dim = hid_dim
    tx = scatter_add(torch.ones_like(ei[0]), ei[0],
                     dim_size=num_node) if x is None else x
    max_x = torch.max(tx).item()
    if args.twoset:
        PredMod = nn.Sequential(
            nn.Dropout(p=kwargs["dp2"], inplace=True),
            EdgePool.EdgeInnerProduct(hid_dim, "nn", kwargs["ln2"], hid_dim),
            EdgePool.EdgeInnerProduct(hid_dim, kwargs["pool"], kwargs["ln2"]))
    else:
        PredMod = nn.Sequential(
            nn.Dropout(p=kwargs["dp2"], inplace=True),
            EdgePool.EdgeInnerProduct(hid_dim, kwargs["pool"], kwargs["ln2"]))
    head_act = nn.Identity()
    if labelmod0 == "None":
        convs = nn.ModuleList([
            model.OnlyConv(mlp=nn.Sequential(
                ModUtils.BiasMod(hid_dim) if labelmod1=="Bias" else nn.Linear(hid_dim, hid_dim),
                nn.Dropout(kwargs["dp1"], inplace=True), nn.ReLU(
                    inplace=True), nn.Linear(hid_dim, hid_dim),
                nn.LayerNorm(hid_dim, elementwise_affine=False
                             ) if kwargs["ln1"] else nn.Identity(),
                nn.Dropout(kwargs["dp1"], inplace=True), nn.ReLU(
                    inplace=True))) for i in range(num_mplayer)
        ])
        RepreMod = model.NoLabelingNet(convs)
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
        RepreMod = model.PLabelingNet2Set(convs, f0s, f1s) if args.twoset else model.PLabelingNet(convs, f0s, f1s)
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
        RepreMod = model.PLabelingNet2Set(convs, f0s, f1s) if args.twoset else model.PLabelingNet(convs, f0s, f1s)
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
        RepreMod = model.FLabelingNet(convs, f0s, f1s)
    return model.FullModel(max_x, num_node, head_act, RepreMod, PredMod, hid_dim)


def work(num_step: int = 2000,
         max_early_stop: int = 500,
         batch_size: int = 128,
         lr: float = 3e-4,
         wd: float = 0,
         test_interval: int = 50,
         **kwargs):
    trn_dl, val_dl, tst_dl = (tensorDataloader(
        (pair[0].t(), pair[1]), batch_size, False, False, device)
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
    full_adj = Utils.eiea2adj(*Utils.gcn_norm(ei, num_node),
                              num_node).to_device(device)
    full_x = scatter_add(torch.ones_like(ei[0]), ei[0],
                         dim_size=num_node).to(device) if x is None else x
    for iteration in range(num_step):
        if iteration % test_interval == 0:
            val_score = test(mod, full_x, full_adj, val_dl, [metric.auroc])[0]
            print(f"val {val_score:.4f}")
            early_stop += 1
            if val_score > best_val_score:
                early_stop = 0
                best_val_score = val_score
                tst_score = test(mod, full_x, full_adj, tst_dl, [metric.auroc])[0]
                print(f" tst {tst_score:.4f}")
        idx = trn_posidx_dl.get_batch(batch_size)
        if idx is None:
            trn_posidx_dl = iter(trn_posidx_dl)
            idx = trn_posidx_dl.get_batch(batch_size)
            assert idx is not None, "something wrong with tensorDataloader"
        mask = Utils.idx2mask(idx, split_edge["train"]["edge"].shape[1])
        pos_positive = split_edge["train"]["edge"][:, mask].t()
        edge_t = split_edge["train"]["edge"][:, torch.logical_not(mask)]
        edge_t = torch.cat((edge_t[[1, 0]], edge_t), dim=-1)
        degree_t = scatter_add(torch.ones_like(edge_t[0]),
                               edge_t[0],
                               dim_size=num_node) if x is None else x
        adj_t = Utils.eiea2adj(*Utils.gcn_norm(edge_t, num_node),
                               num_node).to_device(device)
        pos_negative = trn_negpos_dl.get_batch(batch_size)
        if pos_negative is None:
            trn_negpos_dl = iter(trn_negpos_dl)
            pos_negative = trn_negpos_dl.get_batch(batch_size)
            assert pos_negative is not None, "something wrong with tensorDataloader"
        loss = train(mod, degree_t, adj_t, opt, pos_positive, pos_negative)
        print(f"iteration {iteration} trn : {loss:.4f} ")
        if early_stop > max_early_stop:
            break
    print("", flush=True)
    return best_val_score, tst_score


fixed_params = {
    "num_mplayer": 3,
    "labelmod0": "Layerwise",
    "pool": "nn",
    "jk": "last",
    "labelmod1": "Linear",
}


def search(trial: optuna.Trial):
    hid_dim = trial.suggest_int("hid_dim", 32, 128, step=32)
    dp1 = trial.suggest_float("dp1", 0.0, 0.9, step=0.1)
    dp2 = trial.suggest_float("dp2", 0.0, 0.9, step=0.1)
    batch_size = trial.suggest_int("batch_size", 16, 128, 16)
    ln1 = trial.suggest_categorical("ln1", [True, False])
    ln2 = trial.suggest_categorical("ln2", [True, False])
    split()
    return work(**fixed_params,
                hid_dim=hid_dim,
                batch_size=batch_size,
                dp1=dp1,
                dp2=dp2,
                ln1=ln1,
                ln2=ln2)[0]


if args.test:
    from besthp import besthp
    p = besthp[args.dataset]
    if args.nolabel:
        fixed_params["labelmod0"] = "None"
    if args.flabel:
        fixed_params["labelmod0"] = "Full"
    summary = []
    for rep in range(10):
        set_seed(rep)
        split()
        ret = work(**p, **fixed_params, test_interval=50, max_early_stop=500)
        summary.append(ret[1])
    summary = np.array(summary)
    print("p: ", p)
    print("s: ", summary)
    print("ss: ", summary.mean(axis=0), summary.std(axis=0))
else:
    study = optuna.create_study(f"sqlite:///out/{args.dataset}.db",
                                study_name=args.dataset,
                                direction="maximize",
                                load_if_exists=True)
    study.optimize(search, 200)
