import subprocess
import torch
import torch_sparse
from torch_sparse import SparseTensor
from itertools import count
from typing import List, Set, Tuple
import numpy as np
from torch import Tensor
import numpy as np
subprocess.call("git clone https://github.com/arbenson/ScHoLP-Data.git", shell=True)
subprocess.call("mkdir data", shell=True)
dslist = ["DAWN", "email-Eu", "NDC-classes", "NDC-substances", 
    "threads-ask-ubuntu", "threads-math-sx", "tags-ask-ubuntu", "tags-math-sx"]
dslist2 = ["email-Enron", "contact-high-school", "congress-bills", "NDC-classes"]
for ds in dslist:
    subprocess.call(f"gunzip ScHoLP-Data/{ds}/{ds}-nverts.txt.gz", shell=True)
    subprocess.call(f"gunzip ScHoLP-Data/{ds}/{ds}-node-labels.txt.gz", shell=True)
    subprocess.call(f"gunzip ScHoLP-Data/{ds}/{ds}-simplices.txt.gz", shell=True)

def uniqueedge(rowptr: Tensor, col: Tensor):
    edges = []
    for i in range(len(rowptr)-1):
        tmp: np.ndarray =col[rowptr[i]: rowptr[i+1]].numpy()
        edges.append(tuple(sorted(tmp.tolist())))
    edges = list(set(edges))
    rowptr = torch.tensor([0] + [len(_) for _ in edges]).cumsum_(dim=0)
    col = torch.tensor([b for a in edges for b in a])
    return rowptr, col

for ds in dslist:
    with open(f"ScHoLP-Data/{ds}/{ds}-nverts.txt") as f:
        nverts = torch.tensor([0]+[int(_) for _ in f.readlines()])
    with open(f"ScHoLP-Data/{ds}/{ds}-simplices.txt") as f:
        edge = torch.tensor([int(_) for _ in f.readlines()])
    inci = SparseTensor(rowptr=torch.cumsum(nverts, dim=0), col=edge).coalesce()
    deg = inci.sum(dim=1)
    inci = inci[deg>1.5]
    deg = inci.sum(dim=0)
    mask = deg>0
    inci = inci[:, mask]
    rowptr, col = uniqueedge(*(inci.csr()[:-1]))
    torch.save((rowptr, col),f"data/{ds}.pt")
