import torch
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Planetoid
import scipy.io as sio
import scipy.sparse as ssp
from torch_geometric.data import Data


def load_dataset(name: str) -> Data:
    if name in ["Cora", "CiteSeer", "PubMed"]:
        return Planetoid(root="./data", name=name)[0]
    elif name in ["USAir", "NS", "PB", "Yeast", "Router", "Wikipedia", "PB", "Power", "Celegans", "Ecoli", "arxiv"]:
        data_dir = 'data/{}.mat'.format(name)
        data = sio.loadmat(data_dir)
        net = data['net']
        row, col, _ = ssp.find(net)
        ei = torch.stack((torch.from_numpy(row.flatten()).to(torch.long), torch.from_numpy(col.flatten()).to(torch.long)))
        ei = to_undirected(ei)
        ret = Data(x=None, edge_index=ei)
        return ret
    raise NotImplementedError(f"{name} dataset is not included")
