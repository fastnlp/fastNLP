import os
import sys
from typing import Dict, List, Any, Callable, Type, Union
from functools import partial
import copy

import socket
import pytest
import numpy as np

from fastNLP.core.dataset import DataSet
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.core.metrics.metric import Metric
from .utils import find_free_network_port, setup_ddp, _assert_allclose
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed
    from torch.multiprocessing import Pool, set_start_method
else:
    from fastNLP.core.utils.dummy_class import DummyClass as set_start_method
try:
    from sklearn.metrics import accuracy_score as sklearn_accuracy
except:
    pass

set_start_method("spawn", force=True)


NUM_PROCESSES = 2
pool = None


def _test(local_rank: int,
          world_size: int,
          device: "torch.device",
          dataset: DataSet,
          metric_class: Type[Metric],
          metric_kwargs: Dict[str, Any],
          sklearn_metric: Callable,
          atol: float = 1e-8) -> None:
    # metric 应该是每个进程有自己的一个 instance，所以在 _test 里面实例化
    metric = metric_class(**metric_kwargs)
    # dataset 也类似（每个进程有自己的一个）
    dataset = copy.deepcopy(dataset)
    metric.to(device)
    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, len(dataset), world_size):
        pred, tg = torch.tensor(dataset[i]['pred']).to(device), torch.tensor(dataset[i]['target']).to(device)
        metric.update(pred, tg)

        # my_result = metric.get_metric()
        # using_predict, using_target = dataset[: i + world_size]['pred'], dataset[: i + world_size]['target']
        # sklearn_result = sklearn_metric(using_predict, using_target)
        # _assert_allclose(my_result, sklearn_result, atol=atol)

    my_result = metric.get_metric()
    my_result = my_result['acc']
    using_predict, using_target = [], []
    for i in range(len(dataset)):
        using_predict.append(dataset[i]['pred'])
        using_target.append(dataset[i]['target'])
    using_target, using_predict = np.array(using_target), np.array(using_predict)
    sklearn_result = sklearn_metric(using_predict, using_target)
    _assert_allclose(my_result, sklearn_result, atol=atol)


@pytest.fixture(scope='class', autouse=True)
def pre_process():
    global pool
    pool = Pool(processes=NUM_PROCESSES)
    master_port = find_free_network_port()
    pool.starmap(setup_ddp, [(rank, NUM_PROCESSES, master_port) for rank in range(NUM_PROCESSES)])
    yield
    pool.close()
    pool.join()


@pytest.mark.torch
@pytest.mark.parametrize('dataset', [
    DataSet({'pred': np.random.randint(low=0, high=1, size=(36, 32)),
             'target': np.random.randint(low=0, high=1, size=(36, 32))}),
    DataSet({'pred': np.random.randint(low=0, high=1, size=(360, 32)),
             'target': np.random.randint(low=0, high=1, size=(360, 32))})
])
@pytest.mark.parametrize('is_ddp', [True, False])
@pytest.mark.parametrize('metric_class', [Accuracy])
@pytest.mark.parametrize('metric_kwargs', [{'backend': 'auto'}])
class TestAccuracy:

    def test_v1(self, is_ddp: bool, dataset: DataSet, metric_class: Type['Metric'],
                metric_kwargs: Dict[str, Any]) -> None:
        global pool
        if is_ddp:
            if sys.platform == "win32":
                pytest.skip("DDP not supported on windows")
            metric_kwargs['aggregate_when_get_metric'] = True
            processes = NUM_PROCESSES
            pool.starmap(
                partial(
                    _test,
                    dataset=dataset,
                    metric_class=metric_class,
                    metric_kwargs=metric_kwargs,
                    sklearn_metric=sklearn_accuracy,
                ),
                [(rank, processes, torch.device(f'cuda:{rank}')) for rank in range(processes)]
            )
        else:
            device = torch.device(
                "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
            metric_kwargs['aggregate_when_get_metric'] = False
            _test(
                local_rank=0,
                world_size=1,
                device=device,
                dataset=dataset,
                metric_class=metric_class,
                metric_kwargs=metric_kwargs,
                sklearn_metric=sklearn_accuracy
            )
