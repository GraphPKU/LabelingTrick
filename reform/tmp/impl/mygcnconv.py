from torch_geometric.nn import GCNConv
import torch.nn as nn

class myGCNConv(GCNConv):
    def __init__(self, in_channels: int, out_channels: int, improved: bool = False, cached: bool = False, add_self_loops: bool = True, normalize: bool = True, bias: bool = True, **kwargs):
        super().__init__(in_channels, out_channels, improved, cached, add_self_loops, normalize, bias, **kwargs)
        self.lin = nn.Sequential(nn.Identity())