import copy
from typing import Any, Dict, Type

import numpy as np
import pytest
from fastNLP import Instance, Perplexity
from fastNLP.core.dataset import DataSet
from fastNLP.core.metrics.metric import Metric
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    import jittor as jt


def _test(local_rank: int,
          world_size: int,
          dataset: DataSet,
          metric_class: Type[Metric],
          metric_kwargs: Dict[str, Any],
          atol: float = 1e-8) -> None:
    # metric 应该是每个进程有自己的一个 instance，所以在 _test 里面实例化
    metric = metric_class(**metric_kwargs)
    # dataset 也类似（每个进程有自己的一个）
    dataset = copy.deepcopy(dataset)
    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是
    # 一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, len(dataset), world_size):
        pred = dataset[i]['pred']
        pred = jt.nn.softmax(pred, dim=2)
        target = dataset[i]['target']
        target[0, 6] = -100
        target[0, 7] = -101
        metric.update(pred, target)
    results = metric.get_metric()
    np.testing.assert_almost_equal(results['perplexity'], 5.677934, decimal=6)


@pytest.mark.jittor
@pytest.mark.parametrize('metric_kwargs', [{'backend': 'jittor'}])
def test_perplexity_jittor(metric_kwargs):
    # 用 numpy 的种子是因为 jittor 的种子似乎会受到影响
    # 在上层批量执行时会导致结果出错
    np.random.seed(22)
    dataset = DataSet([
        Instance(
            pred=jt.Var(np.random.rand(2, 8, 5)),
            target=jt.Var(np.random.randint(5, size=(2, 8)))),
        Instance(
            pred=jt.Var(np.random.rand(2, 8, 5)),
            target=jt.Var(np.random.randint(5, size=(2, 8)))),
        Instance(
            pred=jt.Var(np.random.rand(2, 8, 5)),
            target=jt.Var(np.random.randint(5, size=(2, 8)))),
        Instance(
            pred=jt.Var(np.random.rand(2, 8, 5)),
            target=jt.Var(np.random.randint(5, size=(2, 8)))),
    ])
    metric_kwargs['ignore_labels'] = [-100, -101]
    metric_kwargs['aggregate_when_get_metric'] = False
    _test(
        local_rank=0,
        world_size=1,
        dataset=dataset,
        metric_class=Perplexity,
        metric_kwargs=metric_kwargs,
    )
