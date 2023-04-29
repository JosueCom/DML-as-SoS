import torch as th
from torch import nn

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.distributions.uniform import Uniform
from torch.utils.data import DataLoader

from src import DatasetManager, Company, SoS
from src.utils import d3ls
from src.data.logger import Logger

"""
Author: Josue N Rivera
"""

## hyper-paramaters
ntrial = 1
ncompanies = 4
ncycle = 2
epochs_per_train = 3
train_size = 100
val_size = 100

## logger
d3ls_scores = []
accuracies = []
if __name__ == '__main__':
    logger = Logger(
        log_keys=['d3ls_scores', 'accuracies', 'avg accuracy', 'final accuracy', 'distributions', 'adjacency'],
        config={
            "number of trials": ntrial,
            'number of companies': ncompanies,
            "number of training-sharing cycles": ncycle,
            "number of epochs in indpendent training": epochs_per_train,
            "training dataset size": train_size,
            "validation dataset size": val_size,
            "save": {
                "path": "src\\data\\",
                "format": "MONTH DAY YEAR (START_HOUR;START_MINUTE - END_HOUR;END_MINUTE) - TYPE",
                "logs": True,
                "printless": False,
                "progress rate": 5
            }
        }
    ) 

    # Validation dataset
    logger.print("Loading dataset")
    mnist_test = MNIST("src/data/", 
                            transform=transforms.ToTensor(), 
                            download=True, train=False)
    dtManager = DatasetManager(mnist_test)
    dt, _ = dtManager.get_dataset(size=val_size, distribution=Uniform(0, 10))
    loader_test = DataLoader(dt, 100, shuffle=True, num_workers=3)

    # Training dataset
    mnist_train = MNIST("src/data/", 
                transform=transforms.ToTensor(),
                download=True)
    dtManager = DatasetManager(mnist_train)
    logger.print("Done loading dataset")

    class Model(nn.Module):
        def __init__(self) -> None:
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x) -> th.Tensor:
            x = self.conv1(x.unsqueeze(1))
            x = th.relu(x)
            x = self.conv2(x)
            x = th.relu(x)
            x = th.max_pool2d(x, 2)
            x = th.flatten(x, 1)
            x = self.fc1(x)
            x = th.relu(x)
            x = self.fc2(x)
            output = th.log_softmax(x, dim=1)
            return output

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    bernoulli_prob_mask = 0.5*(th.ones((ncompanies, ncompanies)) - th.eye(ncompanies))

    for trial in range(ntrial):
        
        logger.print(f"---- Trial {trial+1}/{ntrial} ----")
        companies = []

        for i in range(ncompanies):
            dt, distribution = dtManager.get_dataset(size = train_size)    
            companies.append(
                Company("Company " + str(i+1), 
                        dataset = dt,
                        distribution = distribution,
                        shared_model = Model().to(device),
                        device = device,
                        epochs = epochs_per_train,
                        criterion = nn.NLLLoss
                ))

        sos = SoS(companies)
        logger.log('distributions', sos.get_distribution())

        partners = th.bernoulli(bernoulli_prob_mask).to(th.int).tolist()
        sos.set_partners(partners)
        logger.log('adjacency', partners)

        score = 0.0
        for company in sos.companies:
            score += sum([d3ls(company.distribution, sos.companies[i].distribution).item() for i in company.partners])

        score /= ncompanies*(ncompanies - 1)
        
        logger.log('d3ls_scores', score)

        avg_validation = []

        for cycle in range(ncycle):
            logger.print(f"Cycle {cycle}: Started")

            # independent training
            sos.train()

            # validation test
            avg_validation.append(sos.avg_validate(loader_test))

            # merge neigbors' parameters
            sos.merge_parameters()

        logger.print(f"Cycle {ncycle}: Started")
        sos.train()
        avg_validation.append(sos.avg_validate(loader_test))

        logger.log('accuracies', avg_validation)
        logger.log('avg accuracy', sum(avg_validation)/float(len(avg_validation)))
        logger.log('final accuracy', avg_validation[-1])

    logger.close()