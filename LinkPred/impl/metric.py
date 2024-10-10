import numpy
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import numpy as np

def auroc(pred, y):
    pred = torch.sigmoid_(pred).cpu().numpy().flatten()
    y = y.cpu().numpy()
    return roc_auc_score(y, pred)


def ap(pred, y):
    pred = torch.sigmoid_(pred).cpu().numpy().flatten()
    y = y.cpu().numpy()
    return average_precision_score(y, pred)
