from typing import List
from pyvis.network import Network

import torch as th
from torch.utils.data import DataLoader

from .company import Company

"""
Author: Josue N Rivera
"""

class SoS():

    def __init__(self, companies:List[Company], notebook=True) -> None:

        self.companies = companies
        self.vis_net = Network(notebook=True, directed=True, cdn_resources='remote')

        for i in range(len(self.companies)):
            self.vis_net.add_node(i, 
                                  label = self.companies[i].name,
                                  title = str(self.companies[i]))

    def set_partners(self, partners:list):

        self.vis_net = Network(notebook=True, directed=True, cdn_resources='remote')

        for i in range(len(self.companies)):
            self.companies[i].partners = partners[i]
            self.vis_net.add_node(i, 
                                  label = self.companies[i].name,
                                  title = str(self.companies[i]))

        for i in range(len(self.companies)):
            apis = [self.companies[j].request_parameters for j in partners[i]]
            self.companies[i].set_partners(partners[i], apis)

            for j in partners[i]:
                self.vis_net.add_edge(i, j)

    def train(self) -> None:
        for company in self.companies:
            company.train()

    def merge_parameters(self) -> None:
        for company in self.companies:
            company.merge_parameters()
        
        for company in self.companies:
            company.update_parameters()

    def validate(self, test_loader:DataLoader):
        return [company.validate(test_loader) for company in self.companies]

    def avg_validate(self, test_loader:DataLoader):
        scores = self.validate(test_loader)
        return sum(scores)/float(len(scores))

    def get_names(self) -> List[str]:
        return [company.name for company in self.companies]

    def get_distribution(self) -> th.Tensor:
        return [company.distribution for company in self.companies]
