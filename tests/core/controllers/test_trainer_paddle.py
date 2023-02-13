import os
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from fastNLP.core.callbacks.progress_callback import RichCallback
from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.envs.env import USER_CUDA_VISIBLE_DEVICES
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE

if _NEED_IMPORT_PADDLE:
    from paddle.optimizer import Adam
    from paddle.io import DataLoader

from tests.helpers.datasets.paddle_data import PaddleArgMaxDataset
from tests.helpers.models.paddle_model import \
    PaddleNormalModel_Classification_1
from tests.helpers.utils import magic_argv_env_context, skip_no_cuda


@dataclass
class TrainPaddleConfig:
    num_labels: int = 3
    feature_dimension: int = 3

    batch_size: int = 2
    shuffle: bool = True
    evaluate_every = 2


@pytest.mark.parametrize('device', ['cpu', 0, [0, 1]])
@pytest.mark.parametrize('callbacks', [[RichCallback(5)]])
@pytest.mark.paddledist
@pytest.mark.paddle
@magic_argv_env_context
def test_trainer_paddle(
    device,
    callbacks,
    n_epochs=2,
):
    skip_no_cuda(device)
    if isinstance(device,
                  List) and USER_CUDA_VISIBLE_DEVICES not in os.environ:
        pytest.skip('Skip test fleet if FASTNLP_BACKEND is not set to paddle.')
    model = PaddleNormalModel_Classification_1(
        num_labels=TrainPaddleConfig.num_labels,
        feature_dimension=TrainPaddleConfig.feature_dimension)
    optimizers = Adam(parameters=model.parameters(), learning_rate=0.0001)
    train_dataloader = DataLoader(
        dataset=PaddleArgMaxDataset(20, TrainPaddleConfig.feature_dimension),
        batch_size=TrainPaddleConfig.batch_size,
        shuffle=True)
    val_dataloader = DataLoader(
        dataset=PaddleArgMaxDataset(12, TrainPaddleConfig.feature_dimension),
        batch_size=TrainPaddleConfig.batch_size,
        shuffle=True)
    train_dataloader = train_dataloader
    evaluate_dataloaders = val_dataloader
    evaluate_every = TrainPaddleConfig.evaluate_every
    metrics = {'acc': Accuracy(backend='paddle')}
    trainer = Trainer(
        model=model,
        driver='paddle',
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


@pytest.mark.paddle
def test_trainer_paddle_distributed():
    """测试多卡的 paddle."""
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'pytest', f"{path.joinpath('test_trainer_paddle.py')}", '-m',
        'paddledist'
    ]
    env = deepcopy(os.environ)
    env['FASTNLP_BACKEND'] = 'paddle'
    subprocess.check_call(command, env=env)


@pytest.mark.paddle
def test_distributed_launch_1():
    """测试用户自己不初始化 ddp，直接启动."""
    pytest.skip()
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'python', f"{path.joinpath('_test_trainer_fleet.py')}", '-d', '0', '1'
    ]
    env = deepcopy(os.environ)
    env['FASTNLP_BACKEND'] = 'paddle'
    subprocess.check_call(command, env=env)


@pytest.mark.paddle
def test_distributed_launch_2():
    """测试用户自己不初始化 ddp，使用 python -m paddle.distributed.launch 启动."""
    # TODO 下面两个测试会出现显卡没有占用但使用率100%的问题
    # 似乎是测试文件中 labels 数目有问题，但出现一次问题就要重启机器
    # 暂时跳过这几个测试
    pytest.skip()
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'python',
        '-m',
        'paddle.distributed.launch',
        '--devices',
        '0,1',
        f"{path.joinpath('_test_trainer_fleet.py')}",
    ]
    env = deepcopy(os.environ)
    env['FASTNLP_BACKEND'] = 'paddle'
    subprocess.check_call(command, env=env)


@pytest.mark.paddle
def test_distributed_launch_outside_1():
    """测试用户自己不初始化 ddp，使用 python -m paddle.distributed.launch 启动."""
    skip_no_cuda()
    pytest.skip()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'python',
        '-m',
        'paddle.distributed.launch',
        '--devices',
        '0,1',
        f"{path.joinpath('_test_trainer_fleet_outside.py')}",
    ]
    env = deepcopy(os.environ)
    env['FASTNLP_BACKEND'] = 'paddle'
    subprocess.check_call(command, env=env)
