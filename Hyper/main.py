import torch
from torch_sparse import SparseTensor, cat as sp_cat
import argparse
from torch import Tensor
from model import HGNN, MLPNN, MLPNN_label, PZOHGNN, DegInit, RandomInit
from utils import neg_sample, sparse2tuplelist, tensorDataloader, tuplelist2sparse
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import optuna

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--kfold", type=int, default=5)
parser.add_argument("--num_neg", type=int, default=5)
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str)
parser.add_argument("--no_label", action="store_true")
args = parser.parse_args()

loaded_data = torch.load(f"data/{args.dataset}.pt")
inci = SparseTensor(rowptr=loaded_data[0],
                    col=loaded_data[1])  #(num_edge, num_node)

num_edge, num_node = inci.sparse_sizes()

split_idx = torch.randperm(num_edge)
split_masks = torch.zeros((args.kfold, num_edge), dtype=torch.bool)
split_size = num_edge // args.kfold
for i in range(args.kfold):
    split_masks[i, split_idx[i * split_size:(i + 1) * split_size]] = True


def opt(trial: optuna.Trial):
    kwargs = {}
    kwargs["batch_size"] = trial.suggest_int("batch_size", 16, 256, 16)
    kwargs["num_layer"] = trial.suggest_int("num_layer", 1, 4)
    kwargs["hid_dim"] = trial.suggest_int("hid_dim", 16, 64, 16)
    kwargs["lr"] = trial.suggest_categorical("lr",
                                             [1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2])
    kwargs["layerwise_label"] = trial.suggest_categorical("layerwise_label", [True, False])
    kwargs["use_act"] = trial.suggest_categorical("use_act", [True, False])    
    kwargs["share_label"] = trial.suggest_categorical("share_label", [True, False])    
    kwargs["share_lin"] = trial.suggest_categorical("share_lin", [True, False])    
    kwargs["scalefreq"] = trial.suggest_categorical("scalefreq", [True, False])    
    kwargs["dp0"] = trial.suggest_float("dp0", 0.0, 0.5, step=0.05)    
    return work(**kwargs)[0]


def idx2negidx(idx: Tensor):
    idx = idx * args.num_neg
    idx = idx.unsqueeze(0) + torch.arange(
        args.num_neg, dtype=idx.dtype, device=idx.device).unsqueeze_(-1)
    return idx.flatten()


def hasbit(a, b):
    return (a & (1 << b)) > 0


def buildMod(hid_dim: int, **kwargs):
    if args.model == "plabel":
        mod = MLPNN(inci.sparse_sizes()[1], hid_dim, **kwargs).to(dev)
    elif args.model == "label":
        mod = MLPNN_label(inci.sparse_sizes()[1], hid_dim, **kwargs).to(dev)
    else:
        raise NotImplementedError
    return mod

dev = torch.device("cuda")


def work(**kwargs):
    ret = []
    for i in range(args.kfold):
        mod = buildMod(**kwargs).to(dev)
        
        print(mod, flush=True)
        opt = Adam(mod.parameters(), kwargs["lr"])
        tst_inci: SparseTensor = inci[split_masks[i]]
        trn_inci = inci[torch.logical_not(split_masks[i])]
        edgeset = set(sparse2tuplelist(trn_inci))
        tst_neg = tuplelist2sparse(
            neg_sample(num_node, args.num_neg, edgeset,
                       sparse2tuplelist(tst_inci)), num_node)
        num_trn = trn_inci.sparse_sizes()[0]
        trn_idx = torch.arange(num_trn)
        trn_dl = tensorDataloader(trn_idx, kwargs["batch_size"], True)
        inci_mask = torch.ones(num_trn, dtype=torch.bool, device=dev)

        num_pos, num_neg = tst_inci.sparse_sizes()[0], tst_neg.sparse_sizes(
        )[0]
        tst_y = np.zeros((num_pos + num_neg, ))
        tst_y[:num_pos] = 1
        tst_inci = sp_cat((tst_inci, tst_neg), dim=0)
        best_score = 0
        tst_inci = tst_inci.cuda()
        ctrn_inci = trn_inci.cuda()
        val_idx = torch.arange(tst_inci.sparse_sizes()[0], device=tst_inci.device())
        val_dl = tensorDataloader(val_idx, kwargs["batch_size"], False)
        for epoch in range(200):
            mod.train()
            trn_neg = tuplelist2sparse(
                neg_sample(num_node, args.num_neg, edgeset,
                           sparse2tuplelist(trn_inci)), num_node).cuda()

            losss = []
            for idx in trn_dl:
                opt.zero_grad()
                inci_mask[idx] = False
                tmp_inci = ctrn_inci[inci_mask]
                inci_mask[idx] = True
                tar_inci = ctrn_inci[idx]
                neg_inci = trn_neg[idx2negidx(idx)]
                num_pos, num_neg = tar_inci.sparse_sizes(
                )[0], neg_inci.sparse_sizes()[0]
                tar_inci = sp_cat((tar_inci, neg_inci), dim=0)
                # print(tmp_inci.device(), tar_inci.device())
                pred = mod(tmp_inci, tar_inci)
                loss = -F.logsigmoid(pred[:num_pos]).mean() - F.logsigmoid(
                    -pred[num_pos:]).mean()
                loss.backward()
                opt.step()
                losss.append(loss)
            loss = np.average([_.item() for _ in losss])
            mod.eval()
            with torch.no_grad():
                pred = torch.cat([mod(ctrn_inci, tst_inci[idx]) > 0 for idx in val_dl]).cpu().numpy() 
            score = f1_score(tst_y, pred)
            print(f"epoch {epoch} loss {loss} f1 score {score:.4f}", flush=True)
            best_score = max(best_score, score)
        print(f"split {i} bestscore {best_score:.4f}")
        ret.append(best_score)
    return np.average(ret), np.std(ret)


if args.test:
    from best_params import bparams 
    best_params = bparams[args.model]["NDC-classes"]#[args.dataset]
    best_params["no_label"] =  args.no_label
    print(best_params)
    print("average", work(**best_params))
else:
    stu = optuna.create_study(f"sqlite:///{args.dataset}.{args.model}.db",
                              study_name=f"{args.dataset}.{args.model}",
                              direction="maximize",
                              load_if_exists=True)

    stu.optimize(opt)
