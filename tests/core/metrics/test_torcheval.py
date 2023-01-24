from dataclasses import dataclass
from typing import Any

import pytest

from fastNLP import Trainer
from fastNLP.envs.imports import _NEED_IMPORT_TORCH, _module_available
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.utils import magic_argv_env_context, skip_no_cuda

if _NEED_IMPORT_TORCH:
    import torch
    from torch.optim import SGD
    from torch.utils.data import DataLoader, Dataset
    if _module_available('torcheval'):
        from torcheval.metrics import MulticlassAccuracy

    class DataSet(Dataset):

        def __init__(self, num_sample=100, num_features=2):
            g = torch.Generator()
            g.manual_seed(1000)
            self.data = torch.randn(num_sample, num_features, generator=g)
            self.y = self.data.argmax(dim=-1)

        def __getitem__(self, item):
            return {'x': self.data[item], 'y': self.y[item]}

        def __len__(self):
            return len(self.data)


@dataclass
class Config:
    num_labels: int = 2
    feature_dimension: int = 2
    seed: int = 0
    batch_size: int = 4
    shuffle: bool = True


@dataclass
class TrainerParameters:
    model: Any = None
    optimizers: Any = None
    train_dataloader: Any = None
    evaluate_dataloader: Any = None


@pytest.fixture(scope='module', params=[1], autouse=True)
def trainer_params(request):
    trainer_params = TrainerParameters()
    trainer_params.model = TorchNormalModel_Classification_1(
        num_labels=Config.num_labels,
        feature_dimension=Config.feature_dimension)
    trainer_params.optimizers = SGD(
        trainer_params.model.parameters(), lr=0.001)
    dataset = DataSet(10, num_features=Config.feature_dimension)
    _dataloader = DataLoader(
        dataset=dataset, batch_size=Config.batch_size, shuffle=True)
    trainer_params.train_dataloader = _dataloader
    trainer_params.evaluate_dataloader = _dataloader
    return trainer_params


@pytest.mark.torch
@pytest.mark.parametrize('device', ['cpu', 'cuda', 1])
@pytest.mark.skipif(not _module_available('torcheval'), reason='No torcheval')
@magic_argv_env_context
def test_1(trainer_params: TrainerParameters, device):
    skip_no_cuda(device)
    trainer = Trainer(
        model=trainer_params.model,
        train_dataloader=trainer_params.train_dataloader,
        evaluate_dataloaders=trainer_params.evaluate_dataloader,
        optimizers=trainer_params.optimizers,
        n_epochs=1,
        device=device,
        driver='torch',
        metric={'acc': MulticlassAccuracy()})
    trainer.run()
