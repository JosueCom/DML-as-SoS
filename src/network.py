from typing import List

import torch as th

from company import Company

"""
Author: Josue N Rivera
"""

class SoS():

    def __init__(self, companies:List[Company]) -> None:

        self.companies = companies

    def set_partners(self):
        pass

    def validate(self, dataset):
        pass

    def get_names(self) -> List[str]:
        return [company.name for company in self.companies]

    def get_distribution(self) -> th.Tensor:
        return th.tensor([company.distribution for company in self.companies])
