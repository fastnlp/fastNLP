"""
注意这一文件中的测试函数都应当是在 `test_trainer_w_evaluator_torch.py` 中已经测试过的测试函数的基础上加上 metrics 和 evaluator 修改而成；
"""
import pytest

from dataclasses import dataclass
from typing import Any

from fastNLP.core.controllers.trainer import Trainer
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchNormalDataset_Classification, TorchArgMaxDataset
from tests.helpers.callbacks.helper_callbacks import RecordLossCallback, RecordMetricCallback
from tests.helpers.utils import magic_argv_env_context
from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    from torch.optim import SGD
    from torch.utils.data import DataLoader
    import torch.distributed as dist
    from torchmetrics import Accuracy


@dataclass
class NormalClassificationTrainTorchConfig:
    num_labels: int = 2
    feature_dimension: int = 3
    each_label_data: int = 10
    seed: int = 0

    batch_size: int = 4
    shuffle: bool = True


@dataclass
class ArgMaxDatasetConfig:
    num_labels: int = 4
    feature_dimension: int = 4
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
@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", "cpu"), ("torch", 1),
                                           ("torch", [0, 1])])  # ("torch", "cpu"), ("torch", 1), ("torch", [0, 1])
@pytest.mark.parametrize("evaluate_every", [-3, -1, 2])
@magic_argv_env_context
def test_trainer_torch_with_evaluator(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        evaluate_every,
        n_epochs=4,
):
    callbacks = [RecordMetricCallback(monitor="acc", metric_threshold=0.2, larger_better=True)]
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


@pytest.mark.torch
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


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", 'cpu')])  # ("torch", [0, 1]),("torch", 1)
@magic_argv_env_context
def test_trainer_validate_every(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        n_epochs=6,
):
    def validate_every(trainer):
        if trainer.global_forward_batches % 10 == 0:
            print("\nfastNLP test validate every.\n")
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


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", 'cpu')])  # ("torch", [0, 1]),("torch", 1)
@magic_argv_env_context
def test_trainer_on(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        n_epochs=2,
):
    from fastNLP import Event
    @Trainer.on(Event.on_before_backward())
    def before_backend(trainer, outputs):
        pass

    @Trainer.on(Event.on_before_backward())
    def before_backend_2(*args):
        pass

    trainer = Trainer(
        model=model_and_optimizers.model,
        driver=driver,
        device=device,
        optimizers=model_and_optimizers.optimizers,
        train_dataloader=model_and_optimizers.train_dataloader,
        evaluate_dataloaders={"dl": model_and_optimizers.evaluate_dataloaders},
        input_mapping=model_and_optimizers.input_mapping,
        output_mapping=model_and_optimizers.output_mapping,
        metrics=model_and_optimizers.metrics,
        n_epochs=n_epochs,
        output_from_new_proc="all",
        evaluate_every=-1
    )

    trainer.run()


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", 'cpu'), ("torch", 0)])  # ("torch", [0, 1]),("torch", 1)
@magic_argv_env_context
def test_trainer_specific_params_1(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        n_epochs=2,
):
    """
    测试一些特殊的参数是否能够正确地传递；
    """
    trainer = Trainer(
        model=model_and_optimizers.model,
        driver=driver,
        device=device,
        optimizers=model_and_optimizers.optimizers,
        train_dataloader=model_and_optimizers.train_dataloader,
        evaluate_dataloaders={"dl": model_and_optimizers.evaluate_dataloaders},
        input_mapping=model_and_optimizers.input_mapping,
        output_mapping=model_and_optimizers.output_mapping,
        metrics=model_and_optimizers.metrics,
        n_epochs=n_epochs,
        output_from_new_proc="all",
        evaluate_every=-1,

        model_wo_auto_param_call=True,
        torch_kwargs={
            "non_blocking": False,
            "set_grad_to_none": True
        }

    )

    assert trainer.driver.non_blocking is False
    assert trainer.driver.wo_auto_param_call is True

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", [0, 1])])  # ("torch", [0, 1]),("torch", 1)
@magic_argv_env_context
def test_trainer_specific_params_2(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        n_epochs=2,
):
    """
    测试一些特殊的参数是否能够正确地传递；
    """
    trainer = Trainer(
        model=model_and_optimizers.model,
        driver=driver,
        device=device,
        optimizers=model_and_optimizers.optimizers,
        train_dataloader=model_and_optimizers.train_dataloader,
        evaluate_dataloaders={"dl": model_and_optimizers.evaluate_dataloaders},
        input_mapping=model_and_optimizers.input_mapping,
        output_mapping=model_and_optimizers.output_mapping,
        metrics=model_and_optimizers.metrics,
        n_epochs=n_epochs,
        output_from_new_proc="all",
        evaluate_every=-1,

        model_wo_auto_param_call=True,
        torch_kwargs={
            "ddp_kwargs": {
                "broadcast_buffers": True,
                "find_unused_parameters": True
            },
            "non_blocking": False,
        }

    )

    assert trainer.driver.non_blocking is False
    assert trainer.driver.wo_auto_param_call is True
    assert trainer.driver.output_from_new_proc == "all"

    _ddp_kwargs = trainer.driver._ddp_kwargs
    assert _ddp_kwargs.get("broadcast_buffers") is True
    assert _ddp_kwargs.get("find_unused_parameters") is True

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", 1), ("torch", [0, 1])])  # ("torch", [0, 1]),("torch", 1)
@pytest.mark.parametrize("overfit_batches,num_train_batch_per_epoch", [(-1, -1), (0, -1), (3, 10), (6, -1)])
@magic_argv_env_context
def test_trainer_w_evaluator_overfit_torch(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        overfit_batches,
        num_train_batch_per_epoch
):
    """
    测试一些特殊的参数是否能够正确地传递；
    """
    trainer = Trainer(
        model=model_and_optimizers.model,
        driver=driver,
        device=device,
        overfit_batches=overfit_batches,
        optimizers=model_and_optimizers.optimizers,
        train_dataloader=model_and_optimizers.train_dataloader,
        evaluate_dataloaders={"dl": model_and_optimizers.evaluate_dataloaders},
        input_mapping=model_and_optimizers.input_mapping,
        output_mapping=model_and_optimizers.output_mapping,
        metrics=model_and_optimizers.metrics,
        n_epochs=2,
        output_from_new_proc="all",
        evaluate_every=-1,

        torch_kwargs={
            "non_blocking": False,
            "set_grad_to_none": True
        }

    )

    trainer.run(num_train_batch_per_epoch=num_train_batch_per_epoch)

    if dist.is_initialized():
        dist.destroy_process_group()