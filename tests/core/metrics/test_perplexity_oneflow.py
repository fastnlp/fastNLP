import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Type

path = os.path.abspath(__file__)
folders = path.split(os.sep)
for folder in list(folders[::-1]):
    if 'fastnlp' not in folder.lower():
        folders.pop(-1)
    else:
        break
path = os.sep.join(folders)
sys.path.extend([path, os.path.join(path, 'fastNLP')])

import numpy as np
import pytest
from fastNLP import Instance, Perplexity
from fastNLP.core.dataset import DataSet
from fastNLP.core.metrics.metric import Metric
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW

from tests.helpers.utils import skip_no_cuda

if _NEED_IMPORT_ONEFLOW:
    import oneflow
    import oneflow.nn.functional as F


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
    dataset = copy.deepcopy(dataset)
    metric.to(device)
    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是
    # 一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, len(dataset), world_size):
        pred = dataset[i]['pred']
        pred = F.softmax(pred, dim=2)
        target = dataset[i]['target']
        target[0, 6] = -100
        target[0, 7] = -101
        metric.update(pred, target)
    results = metric.get_metric()
    np.testing.assert_almost_equal(results['perplexity'], 4.715723, decimal=6)


@pytest.mark.oneflowdist
@pytest.mark.parametrize('metric_kwargs', [{'backend': 'oneflow'}])
class TestPerplexity:

    def test_v1(self, metric_kwargs: Dict[str, Any]) -> None:
        global pool
        dataset = DataSet([
            Instance(
                pred=oneflow.rand((2, 8, 5),
                                  generator=oneflow.manual_seed(22)),
                target=oneflow.randint(
                    5, (2, 8), generator=oneflow.manual_seed(14))),
            Instance(
                pred=oneflow.rand((2, 8, 5),
                                  generator=oneflow.manual_seed(22)),
                target=oneflow.randint(
                    5, (2, 8), generator=oneflow.manual_seed(14))),
            Instance(
                pred=oneflow.rand((2, 8, 5),
                                  generator=oneflow.manual_seed(22)),
                target=oneflow.randint(
                    5, (2, 8), generator=oneflow.manual_seed(14))),
            Instance(
                pred=oneflow.rand((2, 8, 5),
                                  generator=oneflow.manual_seed(22)),
                target=oneflow.randint(
                    5, (2, 8), generator=oneflow.manual_seed(14))),
        ])
        metric_kwargs['ignore_labels'] = [-100, -101]
        if sys.platform == 'win32':
            pytest.skip('DDP not supported on windows')
        metric_kwargs['aggregate_when_get_metric'] = True
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        _test(
            local_rank=local_rank,
            world_size=world_size,
            device=local_rank,
            dataset=dataset,
            metric_class=Perplexity,
            metric_kwargs=metric_kwargs)


@pytest.mark.oneflow
@pytest.mark.parametrize('metric_kwargs', [{'backend': 'oneflow'}])
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_perplexity_paddle(device, metric_kwargs):
    skip_no_cuda(device)
    dataset = DataSet([
        Instance(
            pred=oneflow.rand((2, 8, 5), generator=oneflow.manual_seed(22)),
            target=oneflow.randint(
                5, (2, 8), generator=oneflow.manual_seed(14))),
        Instance(
            pred=oneflow.rand((2, 8, 5), generator=oneflow.manual_seed(22)),
            target=oneflow.randint(
                5, (2, 8), generator=oneflow.manual_seed(14))),
        Instance(
            pred=oneflow.rand((2, 8, 5), generator=oneflow.manual_seed(22)),
            target=oneflow.randint(
                5, (2, 8), generator=oneflow.manual_seed(14))),
        Instance(
            pred=oneflow.rand((2, 8, 5), generator=oneflow.manual_seed(22)),
            target=oneflow.randint(
                5, (2, 8), generator=oneflow.manual_seed(14))),
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


@pytest.mark.oneflow
def test_perplexity_dist():
    r"""分布式的测试"""
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'python',
        '-m',
        'oneflow.distributed.launch',
        '--nproc_per_node',
        '2',
        f"{path.joinpath('test_perplexity_oneflow.py')}",
    ]
    subprocess.check_call(command, env=os.environ)


if __name__ == '__main__':

    pytest.main([f'{__file__}', '-m', 'oneflowdist'])
