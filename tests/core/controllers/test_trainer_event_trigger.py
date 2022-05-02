import pytest
from typing import Any
from dataclasses import dataclass
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import torch.distributed as dist

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.callbacks.callback_events import Events
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchNormalDataset_Classification
from tests.helpers.callbacks.helper_callbacks import RecordTrainerEventTriggerCallback
from tests.helpers.utils import magic_argv_env_context, Capturing


@dataclass
class NormalClassificationTrainTorchConfig:
    num_labels: int = 2
    feature_dimension: int = 3
    each_label_data: int = 100
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


@pytest.fixture(scope="module", autouse=True)
def model_and_optimizers():
    trainer_params = TrainerParameters()

    trainer_params.model = TorchNormalModel_Classification_1(
        num_labels=NormalClassificationTrainTorchConfig.num_labels,
        feature_dimension=NormalClassificationTrainTorchConfig.feature_dimension
    )
    trainer_params.optimizers = SGD(trainer_params.model.parameters(), lr=0.001)
    dataset = TorchNormalDataset_Classification(
        num_labels=NormalClassificationTrainTorchConfig.num_labels,
        feature_dimension=NormalClassificationTrainTorchConfig.feature_dimension,
        each_label_data=NormalClassificationTrainTorchConfig.each_label_data,
        seed=NormalClassificationTrainTorchConfig.seed
    )
    _dataloader = DataLoader(
        dataset=dataset,
        batch_size=NormalClassificationTrainTorchConfig.batch_size,
        shuffle=True
    )
    trainer_params.train_dataloader = _dataloader
    trainer_params.evaluate_dataloaders = _dataloader
    trainer_params.metrics = {"acc": Accuracy()}

    return trainer_params


@pytest.mark.parametrize("driver,device", [("torch", "cpu")])  # , ("torch", 6), ("torch", [6, 7])
@pytest.mark.parametrize("callbacks", [[RecordTrainerEventTriggerCallback()]])
@pytest.mark.torch
@magic_argv_env_context
def test_trainer_event_trigger(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        callbacks,
        n_epochs=2,
):

    with pytest.raises(Exception):
        with Capturing() as output:
            trainer = Trainer(
                model=model_and_optimizers.model,
                driver=driver,
                device=device,
                optimizers=model_and_optimizers.optimizers,
                train_dataloader=model_and_optimizers.train_dataloader,
                evaluate_dataloaders=model_and_optimizers.evaluate_dataloaders,
                input_mapping=model_and_optimizers.input_mapping,
                output_mapping=model_and_optimizers.output_mapping,
                metrics=model_and_optimizers.metrics,

                n_epochs=n_epochs,
                callbacks=callbacks
            )

            trainer.run()

            if dist.is_initialized():
                dist.destroy_process_group()

        for name, member in Events.__members__.items():
            assert member.value in output[0]




