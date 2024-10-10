import enum
import numpy as np
import random
from Dataset import load_dataset
from impl import EdgePool, Utils
from impl.Utils import do_edge_split, to_directed, edge2ds, tensorDataloader
import argparse
import torch
from torch.nn.functional import logsigmoid
import torch_geometric.nn.models as Gmodels
from impl import model, metric, ModUtils, ppamodel
from torch.optim import Adam
import optuna
import torch.nn as nn
from torch_scatter import scatter_add
from time import time
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
'''
from nbfnet/tasks.py
'''






def split_ogb(dataset):
    global split_edge
    split_edge = dataset.get_edge_split()
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
            try:
                split_edge[idx][term] = split_edge[idx][term].t_().to(device)
            except KeyError:
                pass


def set_seed(seed: int):
    print("seed ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split():
    global split_edge
    split_edge = do_edge_split(dat)
    ei = split_edge["train"]["edge"].clone()
    split_edge["train"]["edge"] = to_directed(ei).to(device)
    split_edge["valid"]["edge"] = split_edge["valid"]["edge"].to(device)
    split_edge["valid"]["edge_neg"] = split_edge["valid"]["edge_neg"].to(device)
    split_edge["test"]["edge"] = split_edge["test"]["edge"].to(device)
    split_edge["test"]["edge_neg"] = split_edge["test"]["edge_neg"].to(device)



def compute_ranking(pos_pred, pred, mask):
    ranking = torch.sum((pos_pred.unsqueeze(-1) <= pred) & mask, dim=-1) + 1
    return ranking


import myDataloader


@torch.no_grad()
def test(mod, adj, dl: myDataloader.testDataloader, score_fns):
    mod.eval()
    preds = []
    ys = []
    for taredge, y in dl:
        head, inv = torch.unique_consecutive(taredge[0], return_inverse=True)
        preds.append(mod(adj, head)[inv, taredge[1]])
        ys.append(y)
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return tuple((score_fn(pred, y) for score_fn in score_fns))


from impl.labelutils import train_negsample
import torch.nn.functional as F


def train(mod, adj, opt, tar_edge, num_neg: int, temperature: float):
    mod.train()
    opt.zero_grad()
    pred = mod(adj, tar_edge[0]).squeeze(-1)  #(B, N)
    pos_pred = torch.gather(pred, dim=1,
                            index=tar_edge[1].unsqueeze(-1)).flatten()
    negbatch = train_negsample(adj, tar_edge, num_neg, num_node)
    neg_pred = torch.gather(pred, dim=1, index=negbatch)
    loss_positive = -logsigmoid(pos_pred)
    loss_negative = -logsigmoid(-neg_pred)
    if temperature > 0:
        with torch.no_grad():
            neg_weight = F.softmax(neg_pred / temperature, dim=-1)
    else:
        neg_weight = 1 / num_neg
    loss = loss_positive.mean() + (loss_negative *
                                   neg_weight).sum(dim=-1).mean()
    loss.backward()
    opt.step()
    return loss.item()


def buildModel(hid_dim: int, num_layer: int, **kwargs)->nn.Module:
    return  ppamodel.PlabelNet(num_node, hid_dim, num_layer, **kwargs)#model.PlabelNetOneHead(num_node, hid_dim, num_layer, **kwargs)


def catdata(split_edge):
    edge = torch.cat((split_edge["edge"], split_edge["edge_neg"]), dim=-1)
    y = torch.zeros_like(edge[0], dtype=torch.float)
    y[:split_edge["edge"].shape[1]] = 1
    return edge, y


from myDataloader import trainDataloader, testDataloader


def work(num_epoch: int = 20,
         max_early_stop: int = 5,
         batch_size: int = 16,
         lr: float = 3e-4,
         wd: float = 0,
         test_interval: int = 1,
         num_neg: int = 16,
         temperature: float = 0.5,
         log_step: int = 100,
         sub_sample: int = None,#1000,
         **kwargs):

    trn_dl = trainDataloader(split_edge["train"]["edge"], batch_size)
    val_dl = testDataloader(*catdata(split_edge["valid"]), batch_size)
    tst_dl = testDataloader(*catdata(split_edge["test"]), batch_size)
    print("trn ", len(trn_dl), " val ", len(val_dl), " tst ", len(tst_dl))
    if sub_sample is None:
        sub_sample = len(val_dl) + len(tst_dl)

    mod = buildModel(**kwargs).to(device)
    opt = Adam(mod.parameters(), lr, weight_decay=wd)
    best_val_score = 0.0
    tst_score = 0.0
    early_stop = 0
    smooth_loss = None
    t0 = time()
    adj = trn_dl.ei
    for epoch in range(num_epoch):
        losslist = []
        for step, batch in enumerate(trn_dl):
            tar_edge, tadj = batch
            loss = train(mod, tadj, opt, tar_edge, num_neg, temperature)
            smooth_loss = loss if smooth_loss is None else 0.02*loss + 0.98*smooth_loss
            losslist.append(loss)
            if step% log_step==0:
                torch.save(mod.state_dict(), f"model/{args.dataset}.{epoch}.{step}.pt")
                print(f"{int(time()-t0)} loss {losslist[-1]} smooth {smooth_loss}", flush=True)
            if step >= sub_sample:
                break
        print(f"epoch {epoch} trn : {np.average(losslist):.4f} ")
        if epoch % test_interval == 0:
            val_score = test(mod, adj, val_dl, score_fns)[0]
            print(f"val {val_score:.4f}")
            early_stop += 1
            if val_score > best_val_score:
                early_stop = 0
                best_val_score = val_score
                tst_score = test(mod, adj, tst_dl, score_fns)[0]
                print(f"tst {tst_score:.4f}")
        if early_stop > max_early_stop:
            break
    print("", flush=True)
    return best_val_score, tst_score


import argparse
import optuna


def opt(trial: optuna.Trial):
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    hid_dim = trial.suggest_int("hid_dim", 16, 64, step=16)
    num_layer = trial.suggest_int("num_layer", 1, 6)
    res = trial.suggest_categorical("res", ["add", "cat"])
    label_fn = trial.suggest_categorical("label_fn", ["dist", "zo", "ppr"])
    num_neg = trial.suggest_int("num_neg", 32, 256, step=32)
    max_dist = trial.suggest_int("max_dist", 2, 7, step=1)
    temperature = trial.suggest_categorical(
        "temperature", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0])
    return work(batch_size=batch_size,
                lr=lr,
                hid_dim=hid_dim,
                num_layer=num_layer,
                res=res,
                label_fn=label_fn,
                num_neg=num_neg,
                max_dist=max_dist,
                temperature=temperature,
                num_bin = 50)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    split_edge = None
    device = torch.device("cuda")
    
    if args.dataset.startswith("ogbl"):
        dataset = PygLinkPropPredDataset(args.dataset, root="data")
        num_node = dataset[0].num_nodes
        split_ogb(dataset)
        evaluator = Evaluator(name=args.dataset)
        evaluator.K = {"ogbl-ppa": 100, "ogbl-collab": 50, "ogbl-ddi": 20
                             }[args.dataset]
        # copied from SEAL-OGB
        def evaluate_hits(pred, y):
            pred = pred.flatten()
            mask = y > 0.5
            pos_val_pred = pred[mask]
            neg_val_pred = pred[torch.logical_not(mask)]
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_val_pred,
                'y_pred_neg': neg_val_pred,
            })[f'hits@{evaluator.K}']
            return valid_hits
        score_fns = [evaluate_hits]
    else:
        dat = load_dataset(args.dataset)
        num_node = torch.max(dat.edge_index).item() + 1
        split()
        score_fns = [metric.auroc]
    if args.test:
        params = {
            "temperature": 0.0,
            "batch_size": 96,#64,
            "max_dist": 1,
            "num_layer": 1,
            "lr": 3e-3,
            "num_neg": 1000,
            "res": "add",
            "label_fn": "dist",
            "hid_dim": 8,
        }
        '''
        {
            "temperature": 0.2,
            "batch_size": 16,#64,
            "max_dist": 3,
            "num_layer": 3,
            "lr": 3e-3,
            "num_neg": 16,
            "res": "add",
            "label_fn": "dist",
            "hid_dim": 64,
        }
        '''
        print(params)
        from torch_geometric.utils import degree
        work(**params, max_deg=2*int(degree(split_edge["train"]["edge"][0]).max().item()), to_un=False)

        
    else:
        study = optuna.create_study(f"sqlite:///{args.dataset}onehead.db",
                                    study_name=args.dataset,
                                    direction="maximize",
                                    load_if_exists=True)
        study.optimize(opt, 100)
