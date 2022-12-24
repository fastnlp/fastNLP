import os
import sys
from typing import Dict, List, Any, Callable, Type, Union
from functools import partial
import copy

import socket
import pytest
import numpy as np

from fastNLP import ROUGE
from fastNLP.core.dataset import DataSet
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
    results = metric.get_metric()
    if local_rank == 0:
        if metric_kwargs["use_stemmer"]:
            np.testing.assert_almost_equal(results['rouge1_fmeasure'], 0.7495700120925903)
            np.testing.assert_almost_equal(results['rouge2_fmeasure'], 0.5278186202049255)
            np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.6954764127731323)
        else:
            np.testing.assert_almost_equal(results['rouge1_fmeasure'], 0.6382868885993958)
            np.testing.assert_almost_equal(results['rouge2_fmeasure'], 0.4007352888584137)
            np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.6105090975761414)


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
        "predictions": ['the cat is on the mats', 'There is a big trees near the park here',
                        'The sun rises from the northeast with sunshine', 'I was later for work today for the rainy'],
        "references": [['a cat is on the mating'], ['A big tree is growing near the parks here'],
                       ["A fierce sunning rises in the northeast with sunshine"],
                       ['I go to working too later today for the rainy']]
    })
    # DataSet({'predictions': ['the cat is on the mat'],
    #          'references': [['there is a cat on the mat', 'a cat is on the mat']]}),
])
@pytest.mark.parametrize('is_ddp', [True,False])
@pytest.mark.parametrize('metric_class', [ROUGE])
@pytest.mark.parametrize('metric_kwargs', [{'backend': 'torch'}])
@pytest.mark.parametrize('use_stemmer', [True, False])
class TestROUGE:
    def test_v1(self, is_ddp: bool, dataset: DataSet, metric_class: Type['Metric'],
                metric_kwargs: Dict[str, Any], use_stemmer: bool) -> None:
        global pool
        if is_ddp:
            if sys.platform == "win32":
                pytest.skip("DDP not supported on windows")
            metric_kwargs['aggregate_when_get_metric'] = True
            metric_kwargs["use_stemmer"] = use_stemmer
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

        else:
            device = torch.device(
                "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
            metric_kwargs['aggregate_when_get_metric'] = False
            metric_kwargs["use_stemmer"] = use_stemmer
            _test(
                local_rank=0,
                world_size=1,
                device=device,
                dataset=dataset,
                metric_class=metric_class,
                metric_kwargs=metric_kwargs,
            )
