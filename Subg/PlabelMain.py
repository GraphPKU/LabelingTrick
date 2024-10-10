from email.mime import base
from impl import SubGDataset, train, metrics, utils, config, PlabelingModels
import datasets
import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import argparse
import torch.nn as nn
import functools
import numpy as np
import time
import random
import yaml

parser = argparse.ArgumentParser(description='')
# Dataset settings
parser.add_argument('--dataset', type=str, default='ppi_bp')
# Node feature settings.
# deg means use node degree. one means use homogeneous embeddings.
# nodeid means use pretrained node embeddings in ./Emb
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
parser.add_argument('--test', action='store_true')

parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')

parser.add_argument('--model', type=str, choices=["no", "zo", "pzo", "opzo"])

args = parser.parse_args()
config.set_device(args.device)


def set_seed(seed: int):
    print("seed ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


if args.use_seed:
    set_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

baseG: datasets.BaseGraph = datasets.load_dataset(args.dataset)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, output_channels = 0, 1
score_fn = None

if baseG.y.unique().shape[0] == 2:
    # binary classification task
    def loss_fn(x, y):
        return BCEWithLogitsLoss()(x.flatten(), y.flatten())

    baseG.y = baseG.y.to(torch.float)
    if baseG.y.ndim > 1:
        output_channels = baseG.y.shape[1]
    else:
        output_channels = 1
    score_fn = metrics.binaryf1
else:
    # multi-class classification task
    baseG.y = baseG.y.to(torch.int64)
    loss_fn = CrossEntropyLoss()
    output_channels = baseG.y.unique().shape[0]
    score_fn = metrics.microf1

loader_fn = SubGDataset.GDataloader
tloader_fn = SubGDataset.GDataloader


def split():
    '''
    load and split dataset.
    '''
    # initialize and split dataset
    global trn_node2subg, val_node2subg, tst_node2subg, trn_y, val_y, tst_y
    global max_deg, output_channels
    # initialize node features
    if args.use_deg:
        baseG.setDegreeFeature()
    elif args.use_one:
        baseG.setOneFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        raise NotImplementedError
    
    max_deg = torch.max(baseG.x)
    baseG.to(config.device)
    trn_node2subg = baseG.node2subg[baseG.mask == 0]
    trn_y = baseG.y[baseG.mask == 0]
    val_node2subg = baseG.node2subg[baseG.mask == 1]
    val_y = baseG.y[baseG.mask == 1]
    tst_node2subg = baseG.node2subg[baseG.mask == 2]
    tst_y = baseG.y[baseG.mask == 2]


def buildModel(hidden_dim, conv_layer, dropout, pool, z_ratio, aggr, gn,
               scalefreq, subsetsize=3):
    '''
    Build a GLASS model.
    Args:
        conv_layer: number of GLASSConv.
        pool: pooling function transfer node embeddings to subgraph embeddings.
        z_ratio: see GLASSConv in impl/model.py. Z_ratio in [0.5, 1].
        aggr: aggregation method. mean, sum, or gcn. 
    '''
    conv = PlabelingModels.EmbZGConv(hidden_dim,
                                     hidden_dim,
                                     conv_layer,
                                     max_deg=max_deg,
                                     activation=nn.ReLU(inplace=True),
                                     dropout=dropout,
                                     conv=functools.partial(
                                         PlabelingModels.SimpleConv,
                                         aggr=aggr,
                                         z_ratio=z_ratio,
                                         dropout=dropout),
                                     gn=gn,
                                     scalefreq=scalefreq)

    # use pretrained node embeddings.
    if args.use_nodeid:
        print("load ", f"./Emb/{args.dataset}_{hidden_dim}.pt")
        emb = torch.load(f"./Emb/{args.dataset}_{hidden_dim}.pt",
                         map_location=torch.device('cpu')).detach()
        conv.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)

    mlp = nn.Linear(hidden_dim, output_channels)
    if args.model == "zo":
        gnn = PlabelingModels.GLASS(conv, mlp, pool).to(config.device)
    elif args.model == "pzo":
        gnn = PlabelingModels.PGLASS(conv, mlp, pool).to(config.device)
    elif args.model == "opzo":
        gnn = PlabelingModels.OneHeadPGLASS(conv, mlp, pool, subsetsize).to(config.device)
    elif args.model == "no":
        gnn = PlabelingModels.noGLASS(conv, mlp, pool).to(config.device)
    else:
        raise NotImplementedError
    return gnn


def trainmod(optimizer, model, y, node2subg, batch_size, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    num_sample = y.shape[0]
    perm = torch.randperm(num_sample, device=y.device)
    for i in range(0, num_sample, batch_size):
        optimizer.zero_grad()
        tperm = perm[i:i + batch_size]
        pred = model(baseG.x, baseG.edge_index, baseG.edge_attr,
                     node2subg[tperm])
        #print(pred.shape)
        loss = loss_fn(pred, y[tperm])
        loss.backward()
        total_loss.append(loss)
        optimizer.step()
    total_loss = torch.stack(total_loss).mean().item()
    return total_loss


@torch.no_grad()
def testmod(model, y, node2subg, metrics, batch_size, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    num_sample = y.shape[0]
    for i in range(0, num_sample, batch_size):
        pred = model(baseG.x, baseG.edge_index, baseG.edge_attr,
                     node2subg[i:min(i + batch_size, num_sample)])
        preds.append(pred)
    pred = torch.cat(preds, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)


def test(pool="mean",
         aggr="mean",
         hidden_dim=64,
         conv_layer=8,
         dropout=0.3,
         lr=1e-3,
         z_ratio=0.8,
         batch_size=None,
         gn=True,
         scalefreq=False):
    '''
    Test a set of hyperparameters in a task.
    Args:
        z_ratio: see GLASSConv in impl/model.py. A hyperparameter of GLASS.
        resi: the lr reduce factor of ReduceLROnPlateau.
    '''
    outs = []
    t1 = time.time()
    # we set batch_size = tst_dataset.y.shape[0] // num_div.
    num_div = tst_y.shape[0] / batch_size
    # we use num_div to calculate the number of iteration per epoch and count the number of iteration.
    if args.dataset in ["density", "component", "cut_ratio", "coreness"]:
        num_div /= 5

    outs = []
    for repeat in range(args.repeat):
        if args.use_seed:
            set_seed((1 << repeat) - 1)
        print(f"repeat {repeat}")
        gnn = buildModel(hidden_dim, conv_layer, dropout, pool, z_ratio, aggr,
                         gn, scalefreq)

        optimizer = Adam(gnn.parameters(), lr=lr)
        val_score = 0
        tst_score = 0
        early_stop = 0
        trn_time = []
        for i in range(300):
            t1 = time.time()
            loss = trainmod(optimizer, gnn, trn_y, trn_node2subg, batch_size,
                            loss_fn)
            trn_time.append(time.time() - t1)

            if i >= 0:  #100 / num_div:
                score, _ = testmod(gnn,
                                   val_y,
                                   val_node2subg,
                                   score_fn,
                                   batch_size,
                                   loss_fn=loss_fn)

                if score > val_score:
                    early_stop = 0
                    val_score = score
                    score, _ = testmod(gnn,
                                       tst_y,
                                       tst_node2subg,
                                       score_fn,
                                       batch_size,
                                       loss_fn=loss_fn)
                    tst_score = score
                    print(
                        f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}",
                        flush=True)
                else:
                    early_stop += 1
                    print(f"iter {i} loss {loss:.4f} val {val_score:.4f}",
                          flush=True)
            if early_stop > 100 / num_div:
                break
        print(
            f"end: epoch {i+1}, train time {sum(trn_time):.2f} s, val {val_score:.3f}, tst {tst_score:.3f}",
            flush=True)
        outs.append(tst_score)
    print(f"average {np.average(outs):.3f} error {np.std(outs):.3f}")
    return np.average(outs) - np.std(outs)


print(args)
# read configuration
'''
with open(f"config/{args.dataset}.yml") as f:
    params = yaml.safe_load(f)

print("params", params, flush=True)
split()
test(**(params))
'''

import optuna


def obj(trial: optuna.Trial):
    aggr = trial.suggest_categorical("aggr", ["max", "mean", "sum"])
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    conv_layer = trial.suggest_int("conv_layer", 1, 4)
    dropout = trial.suggest_float("dropout", 0, 0.5, step=0.05)
    hidden_dim = 64
    lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    pool = trial.suggest_categorical("pool", ["max", "mean", "sum"])
    z_ratio = trial.suggest_float("z_ratio", 0.55, 1.0, step=0.05)
    gn = trial.suggest_categorical("gn", [True, False])
    scalefreq = trial.suggest_categorical("scalefreq", [True, False])
    return test(pool, aggr, hidden_dim, conv_layer, dropout, lr, z_ratio,
                batch_size, gn, scalefreq)


split()
print(args)
if args.test:
    from bestparams import params
    params = params[args.dataset]
    print(params)
    test(**(params))
else:
    stu = optuna.create_study(storage=f"sqlite:///{args.dataset}.db",
                              study_name=args.dataset,
                              load_if_exists=True,
                              direction="maximize")
    stu.optimize(obj, 100)