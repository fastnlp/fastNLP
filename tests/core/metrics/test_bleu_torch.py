import os
import sys
from typing import Dict, List, Any, Callable, Type, Union
from functools import partial
import copy

import socket
import pytest
import numpy as np

from fastNLP import BLEU
from fastNLP.core.dataset import DataSet
from fastNLP.core.metrics.metric import Metric
from .utils import find_free_network_port, setup_ddp
from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed
    from torch.multiprocessing import Pool, set_start_method
else:
    from fastNLP.core.utils.dummy_class import DummyClass as set_start_method

set_start_method("spawn", force=True)

NUM_PROCESSES = 2
pool = None


def _test(local_rank: int,
          world_size: int,
          device: "torch.device",
          dataset: DataSet,
          metric_class: Type[Metric],
          metric_kwargs: Dict[str, Any],
          atol: float = 1e-8) -> None:
    # metric 应该是每个进程有自己的一个 instance，所以在 _test 里面实例化
    metric = metric_class(**metric_kwargs)
    # dataset 也类似（每个进程有自己的一个）
    dataset = copy.deepcopy(dataset)
    metric.to(device)
    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, len(dataset), world_size):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    if local_rank == 0:
        np.testing.assert_almost_equal(my_result["bleu"], 0.4181)


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
    DataSet({
        "predictions": ['There is a big tree near the park here',
                        'The sun rises from the northeast with sunshine',
                        'I was late for work today for the rainy',
                        'the cat is on the mat', ],
        "references": [['A big tree is growing near the park here'],
                       ["A fierce sun rises in the northeast with sunshine"],
                       ['I went to work too late today for the rainy'],
                       ['a cat is on the mat', 'one cat is in the mat', 'cat is in mat', 'a cat is on a blue mat'], ]
    })
    # DataSet({'predictions': ['the cat is on the mat'],
    #          'references': [['there is a cat on the mat', 'a cat is on the mat']]}),
])
@pytest.mark.parametrize('is_ddp', [1,2,3])
@pytest.mark.parametrize('metric_class', [BLEU])
@pytest.mark.parametrize('metric_kwargs', [{'backend': 'torch'}])
@pytest.mark.parametrize('smooth', [False])
class TestBLEU:

    @pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.device_count()>1), reason='no cuda')
    def test_v1(self, is_ddp: bool, dataset: DataSet, metric_class: Type['Metric'],
                metric_kwargs: Dict[str, Any], smooth: bool) -> None:
        global pool
        if is_ddp == 1:

            if sys.platform == "win32":
                pytest.skip("DDP not supported on windows")
            metric_kwargs['aggregate_when_get_metric'] = True
            metric_kwargs["smooth"] = smooth
            processes = NUM_PROCESSES
            pool.starmap(
                partial(
                    _test,
                    dataset=dataset,
                    metric_class=metric_class,
                    metric_kwargs=metric_kwargs,
                ),
                [(rank, processes, torch.device(f'cuda:{rank}')) for rank in range(processes)]
            )
        elif is_ddp ==2:
            if sys.platform == "win32":
                pytest.skip("DDP not supported on windows")
            metric_kwargs['aggregate_when_get_metric'] = True
            metric_kwargs["smooth"] = smooth
            processes = NUM_PROCESSES
            pool.starmap(
                partial(
                    _test,
                    dataset=dataset,
                    metric_class=metric_class,
                    metric_kwargs=metric_kwargs,
                ),
                [(0, processes, torch.device(f'cuda:{0}')), (1, processes, torch.device("cpu"))]
            )
        else:
            device = torch.device(
                "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
            metric_kwargs['aggregate_when_get_metric'] = False
            metric_kwargs["smooth"] = smooth
            _test(
                local_rank=0,
                world_size=1,
                device=device,
                dataset=dataset,
                metric_class=metric_class,
                metric_kwargs=metric_kwargs,
            )