import copy
import sys
from functools import partial
from typing import Any, Dict, Type

import numpy as np
import pytest
from fastNLP import Instance, Perplexity
from fastNLP.core.dataset import DataSet
from fastNLP.core.metrics.metric import Metric
from fastNLP.envs.imports import _NEED_IMPORT_TORCH

from .utils import find_free_network_port, setup_ddp
from tests.helpers.utils import skip_no_cuda

if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed
    import torch.nn.functional as F
    from torch.multiprocessing import Pool, set_start_method
else:
    from fastNLP.core.utils.dummy_class import DummyClass as set_start_method

set_start_method('spawn', force=True)

NUM_PROCESSES = 2
pool = None


def _test(local_rank: int,
          world_size: int,
          device: 'torch.device',
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
    np.testing.assert_almost_equal(results['perplexity'], 5.33002, decimal=6)


@pytest.fixture(scope='class', autouse=True)
def pre_process():
    skip_no_cuda()
    global pool
    pool = Pool(processes=NUM_PROCESSES)
    master_port = find_free_network_port()
    pool.starmap(setup_ddp, [(rank, NUM_PROCESSES, master_port)
                             for rank in range(NUM_PROCESSES)])
    yield
    pool.close()
    pool.join()


@pytest.mark.torch
@pytest.mark.parametrize('is_ddp', [True, False])
@pytest.mark.parametrize('metric_class', [Perplexity])
@pytest.mark.parametrize('metric_kwargs', [{'backend': 'torch'}])
class TestPerplexity:

    def test_v1(self, is_ddp: bool, metric_class: Type['Metric'],
                metric_kwargs: Dict[str, Any]) -> None:
        global pool
        dataset = DataSet([
            Instance(
                pred=torch.rand(2, 8, 5, generator=torch.manual_seed(22)),
                target=torch.randint(
                    5, (2, 8), generator=torch.manual_seed(22)),
            ),
            Instance(
                pred=torch.rand(2, 8, 5, generator=torch.manual_seed(18)),
                target=torch.randint(
                    5, (2, 8), generator=torch.manual_seed(18)),
            ),
            Instance(
                pred=torch.rand(2, 8, 5, generator=torch.manual_seed(16)),
                target=torch.randint(
                    5, (2, 8), generator=torch.manual_seed(16)),
            ),
            Instance(
                pred=torch.rand(2, 8, 5, generator=torch.manual_seed(14)),
                target=torch.randint(
                    5, (2, 8), generator=torch.manual_seed(14)),
            )
        ])
        metric_kwargs['ignore_labels'] = [-100, -101]
        if is_ddp:
            if sys.platform == 'win32':
                pytest.skip('DDP not supported on windows')
            metric_kwargs['aggregate_when_get_metric'] = True
            processes = NUM_PROCESSES
            pool.starmap(
                partial(
                    _test,
                    dataset=dataset,
                    metric_class=metric_class,
                    metric_kwargs=metric_kwargs,
                ), [(rank, processes, torch.device(f'cuda:{rank}'))
                    for rank in range(processes)])

        else:
            device = torch.device('cuda' if (
                torch.cuda.is_available() and torch.cuda.device_count() > 0
            ) else 'cpu')
            metric_kwargs['aggregate_when_get_metric'] = False
            _test(
                local_rank=0,
                world_size=1,
                device=device,
                dataset=dataset,
                metric_class=metric_class,
                metric_kwargs=metric_kwargs,
            )


@pytest.mark.torch
@pytest.mark.parametrize(['pred_shape', 'tgt_high', 'tgt_shape'],
                         [[[2, 8], 5,
                           (2, 8)], [[2, 8, 5], 5,
                                     (2, 8, 4)], [[2, 10, 5], 5, (2, 8)]])
def test_input_shape(pred_shape, tgt_high, tgt_shape):
    dataset = {
        'pred':
        torch.rand(*pred_shape, generator=torch.manual_seed(22)),
        'target':
        torch.randint(tgt_high, tgt_shape, generator=torch.manual_seed(22))
    }
    metric = Perplexity(backend='torch')
    with pytest.raises(ValueError):
        metric.update(dataset['pred'], dataset['target'])
