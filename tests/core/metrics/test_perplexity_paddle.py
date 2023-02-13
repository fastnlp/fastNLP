import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np
import pytest
from fastNLP import Instance, Perplexity
from fastNLP.core.dataset import DataSet
from fastNLP.core.metrics.metric import Metric
from fastNLP.core.drivers.paddle_driver.fleet_launcher import FleetLauncher
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE

from tests.helpers.utils import skip_no_cuda

if _NEED_IMPORT_PADDLE:
    import paddle
    import paddle.nn.functional as F
    import paddle.distributed as dist


def _test(local_rank: int,
          world_size: int,
          device,
          dataset: DataSet,
          metric_class: Type[Metric],
          metric_kwargs: Dict[str, Any],
          atol: float = 1e-8) -> None:
    # metric 应该是每个进程有自己的一个 instance，所以在 _test 里面实例化
    metric = metric_class(**metric_kwargs)
    # dataset 也类似（每个进程有自己的一个）
    dataset = deepcopy(dataset)
    metric.to(device)
    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是
    # 一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, len(dataset), world_size):
        pred = dataset[i]['pred']
        pred = F.softmax(pred, axis=2)
        target = dataset[i]['target']
        target[0, 6] = -100
        target[0, 7] = -101
        metric.update(pred, target)
    results = metric.get_metric()
    np.testing.assert_almost_equal(results['perplexity'], 5.677935, decimal=6)


@pytest.mark.paddledist
class TestPerplexity:

    @classmethod
    def setup_class(cls):
        devices = [0, 1]
        output_from_new_proc = 'all'

        launcher = FleetLauncher(
            devices=devices, output_from_new_proc=output_from_new_proc)
        cls.local_rank = int(os.getenv('PADDLE_RANK_IN_NODE', '0'))
        if cls.local_rank == 0:
            launcher = FleetLauncher(devices, output_from_new_proc)
            launcher.launch()
        dist.fleet.init(is_collective=True)
        dist.barrier()

    @pytest.mark.parametrize('metric_kwargs', [{'backend': 'paddle'}])
    def test_v1(self, metric_kwargs: Dict[str, Any]) -> None:
        if sys.platform == 'win32':
            pytest.skip('DDP not supported on windows')
        np.random.seed(22)
        dataset = DataSet([
            Instance(
                pred=paddle.to_tensor(np.random.rand(2, 8, 5)),
                target=paddle.to_tensor(np.random.randint(5, size=(2, 8)))),
            Instance(
                pred=paddle.to_tensor(np.random.rand(2, 8, 5)),
                target=paddle.to_tensor(np.random.randint(5, size=(2, 8)))),
            Instance(
                pred=paddle.to_tensor(np.random.rand(2, 8, 5)),
                target=paddle.to_tensor(np.random.randint(5, size=(2, 8)))),
            Instance(
                pred=paddle.to_tensor(np.random.rand(2, 8, 5)),
                target=paddle.to_tensor(np.random.randint(5, size=(2, 8)))),
        ])
        metric_kwargs['ignore_labels'] = [-100, -101]
        world_size = int(os.environ['PADDLE_TRAINERS_NUM'])

        metric_kwargs['aggregate_when_get_metric'] = True
        _test(
            local_rank=self.local_rank,
            world_size=world_size,
            device=self.local_rank,
            dataset=dataset,
            metric_class=Perplexity,
            metric_kwargs=metric_kwargs)


@pytest.mark.paddle
@pytest.mark.parametrize('metric_kwargs', [{'backend': 'paddle'}])
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_perplexity_paddle(device, metric_kwargs):
    skip_no_cuda(device)
    # 用 numpy 的种子是因为种子似乎会受到影响
    # 在上层批量执行时会导致结果出错
    np.random.seed(22)
    dataset = DataSet([
        Instance(
            pred=paddle.to_tensor(np.random.rand(2, 8, 5)),
            target=paddle.to_tensor(np.random.randint(5, size=(2, 8)))),
        Instance(
            pred=paddle.to_tensor(np.random.rand(2, 8, 5)),
            target=paddle.to_tensor(np.random.randint(5, size=(2, 8)))),
        Instance(
            pred=paddle.to_tensor(np.random.rand(2, 8, 5)),
            target=paddle.to_tensor(np.random.randint(5, size=(2, 8)))),
        Instance(
            pred=paddle.to_tensor(np.random.rand(2, 8, 5)),
            target=paddle.to_tensor(np.random.randint(5, size=(2, 8)))),
    ])
    metric_kwargs['ignore_labels'] = [-100, -101]
    metric_kwargs['aggregate_when_get_metric'] = False
    _test(
        local_rank=0,
        world_size=1,
        device=device,
        dataset=dataset,
        metric_class=Perplexity,
        metric_kwargs=metric_kwargs,
    )


@pytest.mark.paddle
def test_perplexity_dist():
    """测试上面的测试函数."""
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'pytest', f"{path.joinpath('test_perplexity_paddle.py')}", '-m',
        'paddledist'
    ]
    env = deepcopy(os.environ)
    env['FASTNLP_BACKEND'] = 'paddle'
    subprocess.check_call(command, env=env)
