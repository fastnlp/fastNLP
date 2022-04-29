"""
注意这一文件中的测试函数都应当是在 `test_trainer_w_evaluator_torch.py` 中已经测试过的测试函数的基础上加上 metrics 和 evaluator 修改而成；
"""
import pytest
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.distributed as dist
from dataclasses import dataclass
from typing import Any
from torchmetrics import Accuracy

from fastNLP.core.controllers.trainer import Trainer
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchNormalDataset_Classification, TorchArgMaxDataset
from tests.helpers.callbacks.helper_callbacks import RecordLossCallback, RecordMetricCallback
from tests.helpers.utils import magic_argv_env_context


@dataclass
class NormalClassificationTrainTorchConfig:
    num_labels: int = 2
    feature_dimension: int = 3
    each_label_data: int = 100
    seed: int = 0

    batch_size: int = 4
    shuffle: bool = True


@dataclass
class ArgMaxDatasetConfig:
    num_labels: int = 10
    feature_dimension: int = 10
    data_num: int = 100
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


@pytest.fixture(scope="module", params=[1], autouse=True)
def model_and_optimizers(request):
    trainer_params = TrainerParameters()

    if request.param == 0:
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

    elif request.param == 1:
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
        trainer_params.train_dataloader = _dataloader
        trainer_params.evaluate_dataloaders = _dataloader
        trainer_params.metrics = {"acc": Accuracy()}

    return trainer_params


# 测试一下普通的情况；
@pytest.mark.parametrize("driver,device", [("torch", "cpu"), ("torch", 1), ("torch", [0, 1])])  #  ("torch", "cpu"), ("torch", 1), ("torch", [0, 1])
@pytest.mark.parametrize("callbacks", [[RecordMetricCallback(monitor="acc", metric_threshold=0.2, larger_better=True)]])
@pytest.mark.parametrize("evaluate_every", [-3, -1, 100])
@magic_argv_env_context
def test_trainer_torch_with_evaluator(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        callbacks,
        evaluate_every,
        n_epochs=10,
):
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
        evaluate_every=evaluate_every,

        n_epochs=n_epochs,
        callbacks=callbacks,
        output_from_new_proc="all"
    )

    trainer.run()

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize("driver,device", [("torch", [0, 1]), ("torch", 1)])  # ("torch", [0, 1]),("torch", 1)
@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("accumulation_steps", [1, 3])
@magic_argv_env_context
def test_trainer_torch_with_evaluator_fp16_accumulation_steps(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        fp16,
        accumulation_steps,
        n_epochs=6,
):
    callbacks = [RecordMetricCallback(monitor="acc", metric_threshold=0.1, larger_better=True)]
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
        callbacks=callbacks,
        fp16=fp16,
        accumulation_steps=accumulation_steps,
        output_from_new_proc="all"

    )

    trainer.run()

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize("driver,device", [("torch", 1)])  # ("torch", [0, 1]),("torch", 1)
@magic_argv_env_context
def test_trainer_validate_every(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        n_epochs=6,
):

    def validate_every(trainer):
        if trainer.global_forward_batches % 10 == 0:
            print(trainer)
            print("\nfastNLP test validate every.\n")
            print(trainer.global_forward_batches)
            return True

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
        output_from_new_proc="all",
        evaluate_every=validate_every
    )

    trainer.run()

    if dist.is_initialized():
        dist.destroy_process_group()



