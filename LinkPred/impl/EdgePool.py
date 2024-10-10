from typing import Final
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import cosine_similarity


class EdgeInnerProduct(nn.Module):
    pool: Final[str]

    def __init__(self, hid_dim: int, pool: str, ln: bool, out_dim: int = 1):
        super().__init__()
        if pool not in ["sum", "max", "mean", "cos", "nn"]:
            raise NotImplementedError
        self.pool = pool
        self.mlp1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim) if pool == "nn" else nn.Identity(),
            nn.LayerNorm(hid_dim, elementwise_affine=False)
            if ln else nn.Identity(),
            nn.ReLU(inplace=True) if pool == "nn" else nn.Identity())
        self.mlp2 = nn.Sequential(nn.Linear(hid_dim,
                                            out_dim)) if pool == "nn" else None

    def forward(self, emb: Tensor):
        emb = self.mlp1(emb)
        if self.pool == "cos":
            ret = cosine_similarity(emb[:, 0], emb[:, 1])
        elif self.pool == "nn":
            ret = self.mlp2(emb.sum(dim=1)).squeeze(-1)
        else:
            ret = emb.prod(dim=1)
            if self.pool == "sum":
                ret = ret.sum(dim=-1)
            elif self.pool == "max":
                ret = ret.max(dim=-1)[0]
            elif self.pool == "mean":
                ret = ret.mean(dim=-1)
        return ret


class TwoSetPool(nn.Module):
    def __init__(self, hid_dim: int, dp: float, ln: bool, out_dim: int = 1):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim, elementwise_affine=False)
            if ln else nn.Identity(), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlp2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim, elementwise_affine=False)
            if ln else nn.Identity(), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlp3 = nn.Linear(hid_dim, out_dim)

    def forward(self, emb: Tensor):
        emb = self.mlp2(self.mlp1(emb).sum(dim=1)).sum(dim=1)
        return self.mlp3(emb).squeeze(-1)