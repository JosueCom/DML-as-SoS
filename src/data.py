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
                 dt:Dataset = MNIST("src/data/", transform=transforms.ToTensor(), download=True)) -> None:
        
        self.dt = dt
        self.data = dt.data
        self.labels = self.dt.targets
        self.indexes = {}
        self.unique_labels, indxs, self.count = th.unique(self.labels, sorted=True, return_inverse=True, return_counts=True)
        self.max_per_lab = th.min(self.count)

        for lab in self.unique_labels:
            self.indexes[lab] = th.nonzero(indxs == lab, as_tuple=True)[0]

    def get_dataset(self, size=500, distribution=None, p=0.5):
        
        m = Binomial(len(self.unique_labels)-1, p) if type(distribution) != type(Binomial) else distribution
        lab_perm = th.randperm(len(self.unique_labels))
        inv_perm = th.zeros_like(lab_perm)
        for i in range(len(self.unique_labels)):
            inv_perm[i] = th.nonzero(lab_perm == i, as_tuple=True)[0]

        prob_distribution = th.exp(m.log_prob(inv_perm))

        og_sample = m.sample((size,)).to(th.int)
        labels = lab_perm[og_sample]

        dtl = []
        for lab in self.unique_labels:
            dtl.append(self.get_data(lab, n=sum(lab==labels)))

        return DatasetSubset(th.cat(dtl, dim=0), labels), prob_distribution

    def get_data(self, label, n=-1): # return shuffle data given label
        n = self.max_per_lab if n <= 0 else n

        choices = self.indexes[label][th.randint(self.count[label], (n, ))]
        return self.data[choices]

class DatasetSubset(Dataset):
    def __init__(self, data, labels) -> None:
        self.data = data
        self.labels = labels
        self.n = len(self.labels)
    
    def __len__(self):
        return self.n

    def __getitem__(self, index) -> tuple:
        return self.data[index], self.labels[index]

if __name__ == "__main__":
    dtload = DatasetManager()
    dtload.get_dataset()