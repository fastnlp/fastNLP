from typing import Any
from dataclasses import dataclass

import pytest

from fastNLP import Metric, Accuracy
from tests.helpers.utils import magic_argv_env_context
from fastNLP import Trainer, Evaluator
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    from torch.utils.data import DataLoader
    from torch.optim import SGD
    import torch.distributed as dist
    import torch

from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchArgMaxDataset


@dataclass
class ArgMaxDatasetConfig:
    num_labels: int = 10
    feature_dimension: int = 10
    data_num: int = 20
    seed: int = 0

    batch_size: int = 4
    shuffle: bool = True


@dataclass
class TrainerParameters:
    model: Any = None
    optimizers: Any = None
    train_dataloader: Any = None
    evaluate_dataloaders: Any = None
    input_mapping: Any = None
    output_mapping: Any = None
    metrics: Any = None
    more_metrics: Any = None


@pytest.fixture(scope="module", params=[0], autouse=True)
def model_and_optimizers(request):
    trainer_params = TrainerParameters()

    trainer_params.model = TorchNormalModel_Classification_1(
        num_labels=ArgMaxDatasetConfig.num_labels,
        feature_dimension=ArgMaxDatasetConfig.feature_dimension
    )
    trainer_params.optimizers = SGD(trainer_params.model.parameters(), lr=0.001)
    dataset = TorchArgMaxDataset(
        feature_dimension=ArgMaxDatasetConfig.feature_dimension,
        data_num=ArgMaxDatasetConfig.data_num,
        seed=ArgMaxDatasetConfig.seed
    )
    _dataloader = DataLoader(
        dataset=dataset,
        batch_size=ArgMaxDatasetConfig.batch_size,
        shuffle=True
    )

    class LossMetric(Metric):
        def __init__(self):
            super().__init__()
            self.register_element('loss')

        def update(self, loss):
            self.loss += loss.item()

        def get_metric(self) -> dict:
            return self.loss.item()

    trainer_params.train_dataloader = _dataloader
    trainer_params.evaluate_dataloaders = _dataloader
    trainer_params.metrics = {'loss': LossMetric()}

    trainer_params.more_metrics = {"acc": Accuracy()}

    return trainer_params


@pytest.mark.torch
@pytest.mark.parametrize('device', ['cpu', [0, 1]])
@magic_argv_env_context
def test_run( model_and_optimizers: TrainerParameters, device):

    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip(f"No cuda for device:{device}")
    n_epochs = 2
    for progress_bar in ['rich', 'auto', None, 'raw', 'tqdm']:
        trainer = Trainer(
            model=model_and_optimizers.model,
            driver='torch',
            device=device,
            optimizers=model_and_optimizers.optimizers,
            train_dataloader=model_and_optimizers.train_dataloader,
            evaluate_dataloaders=model_and_optimizers.evaluate_dataloaders,
            input_mapping=model_and_optimizers.input_mapping,
            output_mapping=model_and_optimizers.output_mapping,
            metrics=model_and_optimizers.metrics,
            n_epochs=n_epochs,
            callbacks=None,
            progress_bar=progress_bar,
            output_from_new_proc="all",
            evaluate_fn='train_step',
            larger_better=False
        )

        trainer.run()

        evaluator = Evaluator(model=model_and_optimizers.model, dataloaders=model_and_optimizers.train_dataloader,
                              driver=trainer.driver, metrics=model_and_optimizers.metrics,
                              progress_bar=progress_bar, evaluate_fn='train_step')
        evaluator.run()

    if dist.is_initialized():
        dist.destroy_process_group()





