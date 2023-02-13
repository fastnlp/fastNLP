import os
import subprocess
from pathlib import Path
from dataclasses import dataclass

import pytest

from fastNLP.core.callbacks.progress_callback import RichCallback
from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.drivers.torch_driver.utils import _create_default_config
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.envs.imports import _NEED_IMPORT_TORCH, _NEED_IMPORT_DEEPSPEED

if _NEED_IMPORT_TORCH:
    from torch.optim import Adam
    from torch.utils.data import DataLoader

if _NEED_IMPORT_DEEPSPEED:
    import deepspeed
    from deepspeed import comm

from tests.helpers.datasets.torch_data import TorchArgMaxDataset
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.utils import magic_argv_env_context, skip_no_cuda


@dataclass
class TrainDeepSpeedConfig:
    num_labels: int = 3
    feature_dimension: int = 3

    batch_size: int = 2
    shuffle: bool = True
    evaluate_every = 2


@pytest.mark.deepspeed
@pytest.mark.parametrize('device', [[0, 1]])
@pytest.mark.parametrize('callbacks', [[RichCallback(5)]])
@pytest.mark.parametrize('strategy', ['deepspeed', 'deepspeed_stage_1'])
@pytest.mark.parametrize('config', [None, _create_default_config(stage=1)])
@magic_argv_env_context
def test_trainer_deepspeed(
    device,
    callbacks,
    strategy,
    config,
    n_epochs=2,
):
    skip_no_cuda()
    model = TorchNormalModel_Classification_1(
        num_labels=TrainDeepSpeedConfig.num_labels,
        feature_dimension=TrainDeepSpeedConfig.feature_dimension)
    optimizers = Adam(params=model.parameters(), lr=0.0001)
    train_dataloader = DataLoader(
        dataset=TorchArgMaxDataset(TrainDeepSpeedConfig.feature_dimension, 20),
        batch_size=TrainDeepSpeedConfig.batch_size,
        shuffle=True)
    val_dataloader = DataLoader(
        dataset=TorchArgMaxDataset(TrainDeepSpeedConfig.feature_dimension, 12),
        batch_size=TrainDeepSpeedConfig.batch_size,
        shuffle=True)
    train_dataloader = train_dataloader
    evaluate_dataloaders = val_dataloader
    evaluate_every = TrainDeepSpeedConfig.evaluate_every
    metrics = {'acc': Accuracy()}
    if config is not None:
        config[
            'train_micro_batch_size_per_gpu'] = TrainDeepSpeedConfig.batch_size
    trainer = Trainer(
        model=model,
        driver='deepspeed',
        device=device,
        optimizers=optimizers,
        train_dataloader=train_dataloader,
        evaluate_dataloaders=evaluate_dataloaders,
        evaluate_every=evaluate_every,
        metrics=metrics,
        output_mapping={'preds': 'pred'},
        n_epochs=n_epochs,
        callbacks=callbacks,
        deepspeed_kwargs={
            'strategy': strategy,
            'config': config,
            'train_dataloader': train_dataloader,
        })
    trainer.run()

    if comm.is_initialized():
        comm.barrier()
        comm.destroy_process_group()
        deepspeed.utils.groups._WORLD_GROUP = None


@pytest.mark.deepspeed
def test_distributed_launch_1():
    """测试用户自己不初始化 ddp，直接启动."""
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'python', f"{path.joinpath('_test_trainer_deepspeed.py')}", '-d', '0',
        '1'
    ]
    subprocess.check_call(command, env=os.environ)


@pytest.mark.deepspeed
def test_distributed_launch_2():
    """测试用户自己不初始化 ddp，但是使用 deepspeed 启动；"""
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'deepspeed',
        f"{path.joinpath('_test_trainer_deepspeed.py')}",
    ]
    subprocess.check_call(command, env=os.environ)


@pytest.mark.deepspeed
def test_distributed_launch_outside_1():
    """测试用户自己初始化 ddp，通过 deepspeed 启动."""
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'deepspeed',
        f"{path.joinpath('_test_trainer_deepspeed_outside.py')}",
    ]
    subprocess.check_call(command, env=os.environ)
