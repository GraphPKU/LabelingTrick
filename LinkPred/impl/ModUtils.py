import torch.nn as nn
from torch import Tensor
import torch

class ResMod(nn.Module):

    def __init__(self, mod: nn.Module) -> None:
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod(x) + x


class Resblock(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        norm: bool,  # whether to use norm after it.
        dropout: float,
        act=nn.ReLU(inplace=True)):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.act = act
        self.norm = norm
        self.lin3 = nn.Identity() if in_dim == out_dim else nn.Linear(
            in_dim, out_dim)
        self.dp = nn.Dropout(
            p=dropout, inplace=True) if dropout > 0.01 else nn.Identity()

    def forward(self, s: Tensor):
        id = self.lin3(s)
        s = self.act(self.dp(self.lin1(s)))
        return self.lin2(s) + id


class MLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 num_layers: int,
                 tail_act: bool = False,
                 ln: bool = False,
                 act: nn.Module = nn.ReLU(inplace=True),
                 gn: bool = False,
                 res: bool = False,
                 dropout: float = 0.0,
                 **kwargs) -> None:
        super().__init__()
        if gn:
            raise NotImplementedError
        mods = []

        def lin_fn(a, b, c):
            return nn.Linear(a, c)

        if res:
            assert num_layers % 2 == 0
            num_layers = num_layers // 2

            def lin_fn(a, b, c):
                return Resblock(a, b, c, dropout, act)

        for layer in range(num_layers):
            if layer > 0:
                in_dim = hid_dim
            if layer == num_layers - 1:
                hid_dim = out_dim
            mods.append(lin_fn(in_dim, in_dim, hid_dim))
            if layer < num_layers - 1 or tail_act:
                if ln:
                    mods.append(nn.LayerNorm(hid_dim))
                if dropout > 0.01:
                    mods.append(nn.Dropout(p=dropout, inplace=True))
                mods.append(act)
        self.mods = nn.Sequential(*tuple(mods))

    def forward(self, x: Tensor) -> Tensor:
        x = self.mods(x)
        return x


class BiasMod(nn.Module):

    def __init__(self, hid_dim) -> None:
        super().__init__()
        self.register_parameter("bias", nn.Parameter(torch.zeros(
            (1, hid_dim))))
        bound = hid_dim**-0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x + self.bias
