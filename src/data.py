import torch as th
from torch.distributions.binomial import Binomial
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

"""
Author: Josue N Rivera
"""

class DatasetManager():

    def __init__(self, 
                 dt:Dataset = MNIST("src/data/", 
                              transform=transforms.ToTensor(), 
                              download=True)) -> None:
        
        self.dt = dt
        self.data = dt.data
        self.targets = self.dt.targets
        self.indexes = {}
        self.unique_targets, indxs, self.count = th.unique(self.targets, sorted=True, return_inverse=True, return_counts=True)
        self.max_per_lab = th.min(self.count)

        for lab in self.unique_targets:
            self.indexes[lab.item()] = th.nonzero(indxs == lab, as_tuple=True)[0]

    def get_dataset(self, size=500, distribution=None, p=0.5):
        
        m = Binomial(len(self.unique_targets)-1, p) if type(distribution) == type(None) else distribution
        lab_perm = th.randperm(len(self.unique_targets))
        inv_perm = th.zeros_like(lab_perm)
        for i in range(len(self.unique_targets)):
            inv_perm[i] = th.nonzero(lab_perm == i, as_tuple=True)[0]

        prob_distribution = th.exp(m.log_prob(inv_perm))

        og_sample = m.sample((size,)).to(th.int)
        sampled_targets = lab_perm[og_sample]

        dtl = []
        targets = []
        for lab in self.unique_targets:
            n = sum(lab==sampled_targets)

            dtl.append(self.get_data(lab, n=n))
            targets.append(th.ones((n,), dtype=th.long) * lab)

        return DatasetSubset(th.cat(dtl, dim=0), th.cat(targets, dim=0)), prob_distribution

    def get_data(self, target: th.Tensor, n: int = -1): # return shuffled data given target
        n = self.max_per_lab if n <= 0 else n

        choices = self.indexes[target.item()][th.randint(self.count[target], (n, ))]
        return self.data[choices].to(th.float)

class DatasetSubset(Dataset):
    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets
        self.n = len(self.targets)
    
    def __len__(self):
        return self.n

    def __getitem__(self, index) -> tuple:
        return self.data[index], self.targets[index]

if __name__ == "__main__":
    dtManager = DatasetManager()
    dt, _ = dtManager.get_dataset()