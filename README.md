This repository contains the official code for the paper [Neural Common Neighbor with Completion for Link Prediction](https://arxiv.org/pdf/2302.00890.pdf).

**Environment**

Tested Combination:
torch 1.13.0 + pyg 2.2.0 + ogb 1.3.5

**Experiments**

Each directory contain a different code for experiments.

| name        | experiments                 | 
|-------------|-----------------------------|
| DirLinkPred | Directed Link Prediction    |
| Hyper       | Hyper Link Prediction       |
| LinkPred    | Undirected link prediction  |
| SubG        | SubGraph Prediction         |