import os
from dataclasses import dataclass
from pathlib import Path
import subprocess

import pytest

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW

if _NEED_IMPORT_ONEFLOW:
    from oneflow.optim import Adam
    from oneflow.utils.data import DataLoader

from tests.helpers.datasets.oneflow_data import OneflowArgMaxDataset
from tests.helpers.models.oneflow_model import \
    OneflowNormalModel_Classification_1
from tests.helpers.utils import magic_argv_env_context, skip_no_cuda


@dataclass
class TrainOneflowConfig:
    num_labels: int = 3
    feature_dimension: int = 3

    batch_size: int = 2
    shuffle: bool = True
    evaluate_every = 2


@pytest.mark.parametrize('device', ['cpu', 1])
@pytest.mark.parametrize('callbacks', [[]])
@pytest.mark.oneflow
@magic_argv_env_context
def test_trainer_oneflow(
    device,
    callbacks,
    n_epochs=2,
):
    skip_no_cuda(device)
    model = OneflowNormalModel_Classification_1(
        num_labels=TrainOneflowConfig.num_labels,
        feature_dimension=TrainOneflowConfig.feature_dimension)
    optimizers = Adam(params=model.parameters(), lr=0.0001)
    train_dataloader = DataLoader(
        dataset=OneflowArgMaxDataset(20, TrainOneflowConfig.feature_dimension),
        batch_size=TrainOneflowConfig.batch_size,
        shuffle=True)
    val_dataloader = DataLoader(
        dataset=OneflowArgMaxDataset(12, TrainOneflowConfig.feature_dimension),
        batch_size=TrainOneflowConfig.batch_size,
        shuffle=True)
    train_dataloader = train_dataloader
    evaluate_dataloaders = val_dataloader
    evaluate_every = TrainOneflowConfig.evaluate_every
    metrics = {'acc': Accuracy()}
    trainer = Trainer(
        model=model,
        driver='oneflow',
        device=device,
        optimizers=optimizers,
        train_dataloader=train_dataloader,
        evaluate_dataloaders=evaluate_dataloaders,
        evaluate_every=evaluate_every,
        input_mapping=None,
        output_mapping=None,
        metrics=metrics,
        n_epochs=n_epochs,
        callbacks=callbacks,
    )
    trainer.run()


@pytest.mark.oneflow
def test_distributed_launch_outside_1():
    r"""测试用户自己不初始化 ddp，使用 python -m oneflow.distributed.launch
    启动"""
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'python',
        '-m',
        'oneflow.distributed.launch',
        '--nproc_per_node',
        '2',
        f"{path.joinpath('_test_trainer_oneflow.py')}",
    ]
    subprocess.check_call(command, env=os.environ)


@pytest.mark.oneflow
def test_distributed_launch_outside_2():
    r"""测试用户自己初始化 ddp，使用 python -m oneflow.distributed.launch
    启动"""
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'python', '-m', 'oneflow.distributed.launch', '--nproc_per_node', '2',
        f"{path.joinpath('_test_trainer_oneflow.py')}", '-w'
    ]
    subprocess.check_call(command, env=os.environ)
