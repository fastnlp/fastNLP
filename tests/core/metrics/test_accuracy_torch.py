import os
import sys
from typing import Dict, List, Any, Callable, Type, Union
from functools import partial
import copy

import socket
import pytest
import numpy as np
import torch
import torch.distributed
from torch.multiprocessing import Pool, set_start_method
from sklearn.metrics import accuracy_score as sklearn_accuracy

from fastNLP.core.dataset import DataSet
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.core.metrics.metric import Metric

set_start_method("spawn", force=True)


NUM_PROCESSES = 2
pool = None


def setup_ddp(rank: int, world_size: int, master_port: int) -> None:
    """Setup ddp environment."""

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    print(torch.cuda.device_count())
    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real master node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def _assert_allclose(my_result: Union[float, np.ndarray], sklearn_result: Union[float, np.ndarray],
                     atol: float = 1e-8) -> None:
    """
    测试对比结果，这里不用非得是必须数组且维度对应，一些其他情况例如 np.allclose(np.array([[1e10, ], ]), 1e10+1) 也是 True
    :param my_result: 可以不限设备等
    :param sklearn_result:
    :param atol:
    :return:
    """
    assert np.allclose(a=my_result, b=sklearn_result, atol=atol)


def _test(local_rank: int,
          world_size: int,
          device: torch.device,
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
