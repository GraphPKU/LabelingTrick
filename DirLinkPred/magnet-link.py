import os.path as osp
import argparse

import torch
import torch.nn as nn
from sklearn import metrics

from torch_geometric_signed_directed.utils import link_class_split, in_out_degree
from model import GIN_link_prediction_labelingtrick as MagNet_link_prediction
from torch_geometric_signed_directed.data import load_directed_real_data
from torch_geometric.utils import negative_sampling, coalesce
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, default='webkb/cornell'
)  # webkb/cornell, webkb/wisconsin, telegram/, cora_ml/, citeseer/
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--label', type=str, default="dizo")
parser.add_argument('--test', action="store_true")
parser.add_argument('--task', type=str, default="existence")
args = parser.parse_args()
print(args)

path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', args.dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = args.dataset.split('/')
data = load_directed_real_data(dataset=dataset_name[0],
                               root=path,
                               name=dataset_name[1]).to(device)

link_data = link_class_split(data,
                             prob_val=0.05,
                             prob_test=0.15,
                             splits=10,
                             task=args.task, #"existence",
                             device=device,
                             maintain_connect=False)
criterion = nn.NLLLoss()


def work(hid_dim, layer, lr, batch_size, **kwargs):
    def train(X_real,
              X_img,
              y,
              edge_index,
              edge_weight,
              query_edges,
              batch_size=64):
        model.train()
        neg_edge = query_edges[y == 1]
        qe = edge_index.t()
        qe = coalesce(edge_index).t()
        qe = qe[torch.randperm(qe.shape[0], device=qe.device)]
        neg_edge = neg_edge[torch.randperm(neg_edge.shape[0],
                                           device=neg_edge.device)]
        qlen = min(qe.shape[0], neg_edge.shape[0])
        for i in range(0, qlen - batch_size, batch_size):
            tqe = qe[i:i + batch_size]
            tei = torch.cat((qe[:i], qe[i + batch_size:]), dim=0).t()
            tneg = neg_edge[i:i + batch_size]
            ty = torch.zeros((tqe.shape[0] + tneg.shape[0]),
                             device=tqe.device,
                             dtype=torch.long)
            ty[tqe.shape[0]:] = 1
            X_real = in_out_degree(tei, X_real.shape[0]).to(device)
            X_img = X_real
            out = model(X_real, X_img, tei, torch.cat((tqe, tneg)), None)
            loss = criterion(out, ty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = metrics.accuracy_score(ty.cpu(), out.max(dim=1)[1].cpu())
        return loss.detach().item(), train_acc

    def test(X_real,
             X_img,
             y,
             edge_index,
             edge_weight,
             query_edges,
             batch_sizes=32):
        model.eval()
        with torch.no_grad():
            qlen = query_edges.shape[0]
            outs = []
            for i in range(0, qlen, batch_sizes):
                outs.append(
                    model(X_real,
                          X_img,
                          edge_index=edge_index,
                          query_edges=query_edges[i:i + batch_sizes],
                          edge_weight=None))
            out = torch.cat(outs)
        test_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
        return test_acc

    rec = []
    for split in list(link_data.keys()):
        model = MagNet_link_prediction(num_features=2,
                                       hidden=hid_dim,
                                       layer=layer,
                                       label=args.label,
                                       multpred=False,
                                       **kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        edge_index = link_data[split]['graph']
        edge_weight = link_data[split]['weights']
        query_edges = link_data[split]['train']['edges']
        y = link_data[split]['train']['label']
        X_real = in_out_degree(edge_index).to(device)
        X_img = X_real.clone()

        query_val_edges = link_data[split]['val']['edges']
        y_val = link_data[split]['val']['label']
        best_val = 0
        tst_score = 0
        for epoch in range(args.epochs):
            train_loss, train_acc = train(X_real, X_img, y, edge_index,
                                          edge_weight, query_edges, batch_size)
            val_acc = test(X_real, X_img, y_val, edge_index, edge_weight,
                           query_val_edges, batch_size)
            print(
                f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}, Val_Acc: {val_acc:.4f}',
                flush=True)
            if val_acc > best_val:
                best_val = val_acc
                query_test_edges = link_data[split]['test']['edges']
                y_test = link_data[split]['test']['label']
                tst_score = test(X_real, X_img, y_test, edge_index,
                                 edge_weight, query_test_edges)
                print(f'Split: {split:02d}, Test_Acc: {tst_score:.4f}')
        print(f'Split: {split:02d}, Final Test_Acc: {tst_score:.4f}')
        rec.append(best_val)

    print(f"average {np.average(rec)} pm {np.std(rec)}")
    return np.average(rec)


import optuna


def opt(trial: optuna.Trial):
    hid_dim = trial.suggest_int("hid_dim", 8, 64, 8)
    layer = trial.suggest_int("layer", 1, 4)
    lr = trial.suggest_categorical("lr", [3e-4, 1e-3, 3e-3, 1e-2])
    batch_size = trial.suggest_int("batch_size", 8, 128, 8)
    label_fn = trial.suggest_categorical("label_fn", ["lin", "emb"])
    layerwise_label = trial.suggest_categorical("layerwise_label", [True])
    ln = trial.suggest_categorical("ln", [True, False])
    res = trial.suggest_categorical("res", [True, False])
    cat_merge = trial.suggest_categorical("cat_merge", [True, False])
    return work(hid_dim=hid_dim, layer=layer,lr=lr,batch_size=batch_size,
                label_fn=label_fn,
                layerwise_label=layerwise_label,
                ln=ln,
                res=res,
                cat_merge=cat_merge)

if args.test:
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        import random
        random.seed(seed)
        np.random.seed(seed)
    from best_params import params
    set_seed(0)
    '''
    stu = optuna.load_study(storage=f"sqlite:///{args.dataset.replace('/', '.')}_{args.task}.db",
                            study_name=args.dataset.replace('/', '.'))
    
    print(stu.best_params)
    params = stu.best_params
    '''
    print(params)
    work(**(params))
else:
    stu = optuna.create_study(f"sqlite:///{args.dataset.replace('/', '.')}_{args.label}_{args.task}.db",
                            study_name=args.dataset.replace('/', '.'),
                            direction="maximize",
                            load_if_exists=True)
    stu.optimize(opt, 300)
