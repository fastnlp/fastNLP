"""
测试 more_evaluate_callback
（1）能不能正确 evaluate ;
 (2) 能不能保存 topk 并load进来进行训练

"""
import pytest



import os
import pytest
from typing import Any
from dataclasses import dataclass

from pathlib import Path
import re

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.envs import FASTNLP_LAUNCH_TIME, FASTNLP_DISTRIBUTED_CHECK

from tests.helpers.utils import magic_argv_env_context
from fastNLP.envs.distributed import rank_zero_rm
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchArgMaxDataset
from torchmetrics import Accuracy
from fastNLP.core.metrics import Metric
from fastNLP.core.log import logger
from fastNLP.core.callbacks import MoreEvaluateCallback
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    from torch.utils.data import DataLoader
    from torch.optim import SGD
    import torch.distributed as dist

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
@pytest.mark.parametrize("driver,device", [("torch", "cpu"), ("torch", [0, 1]), ("torch", 1)])  # ("torch", "cpu"), ("torch", [0, 1]), ("torch", 1)
@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("only_state_dict", [True, False])
@magic_argv_env_context
def test_model_more_evaluate_callback_1(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        version,
        only_state_dict
):
    try:
        path = Path.cwd().joinpath(f"test_model_checkpoint")
        path.mkdir(exist_ok=True, parents=True)

        if version == 0:
            callbacks = [
                MoreEvaluateCallback(dataloaders=model_and_optimizers.evaluate_dataloaders,
                                     metrics=model_and_optimizers.more_metrics,
                                     evaluate_every=-1,
                                     folder=path, topk=-1,
                                     topk_monitor='acc', only_state_dict=only_state_dict, save_object='model')
            ]
        elif version == 1:
            callbacks = [
                MoreEvaluateCallback(dataloaders=model_and_optimizers.evaluate_dataloaders,
                                     metrics=model_and_optimizers.more_metrics,
                                     evaluate_every=None, watch_monitor='loss', watch_monitor_larger_better=False,
                                     folder=path, topk=1, topk_monitor='acc', only_state_dict=only_state_dict,
                                     save_object='model')
            ]
        n_epochs = 5
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
            output_from_new_proc="all",
            evaluate_fn='train_step'
        )

        trainer.run()

        all_saved_model_paths = {w.name: w for w in path.joinpath(os.environ[FASTNLP_LAUNCH_TIME]).iterdir()}
        # 检查生成保存模型文件的数量是不是正确的；
        if version == 0:
            assert len(all_saved_model_paths) == n_epochs
        elif version == 1:
            assert len(all_saved_model_paths) == 1

        for folder in all_saved_model_paths:
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
                n_epochs=2,
                output_from_new_proc="all",
                evaluate_fn='train_step'
            )
            folder = path.joinpath(os.environ[FASTNLP_LAUNCH_TIME]).joinpath(folder)
            trainer.load_model(folder, only_state_dict=only_state_dict)

            trainer.run()
            trainer.driver.barrier()
    finally:
        rank_zero_rm(path)

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", "cpu"), ("torch", [0, 1]), ("torch", 0)])  # ("torch", "cpu"), ("torch", [0, 1]), ("torch", 1)
@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("only_state_dict", [True, False])
@magic_argv_env_context
def test_trainer_checkpoint_callback_1(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        version,
        only_state_dict
):
    try:
        path = Path.cwd().joinpath(f"test_model_checkpoint")
        path.mkdir(exist_ok=True, parents=True)

        if version == 0:
            callbacks = [
                MoreEvaluateCallback(dataloaders=model_and_optimizers.evaluate_dataloaders,
                                     metrics=model_and_optimizers.more_metrics,
                                     evaluate_every=-1,
                                     folder=path, topk=-1,
                                     topk_monitor='acc', only_state_dict=only_state_dict, save_object='trainer')
            ]
        elif version == 1:
            callbacks = [
                MoreEvaluateCallback(dataloaders=model_and_optimizers.evaluate_dataloaders,
                                     metrics=model_and_optimizers.more_metrics,
                                     evaluate_every=None, watch_monitor='loss', watch_monitor_larger_better=False,
                                     folder=path, topk=1, topk_monitor='acc', only_state_dict=only_state_dict,
                                     save_object='trainer')
            ]
        n_epochs = 5
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
            output_from_new_proc="all",
            evaluate_fn='train_step'
        )

        trainer.run()

        all_saved_model_paths = {w.name: w for w in path.joinpath(os.environ[FASTNLP_LAUNCH_TIME]).iterdir()}
        # 检查生成保存模型文件的数量是不是正确的；
        if version == 0:
            assert len(all_saved_model_paths) == n_epochs
        elif version == 1:
            assert len(all_saved_model_paths) == 1

        for folder in all_saved_model_paths:
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
                n_epochs=7,
                output_from_new_proc="all",
                evaluate_fn='train_step'
            )
            folder = path.joinpath(os.environ[FASTNLP_LAUNCH_TIME]).joinpath(folder)
            trainer.load_checkpoint(folder, only_state_dict=only_state_dict)

            trainer.run()
            trainer.driver.barrier()

    finally:
        rank_zero_rm(path)

    if dist.is_initialized():
        dist.destroy_process_group()
