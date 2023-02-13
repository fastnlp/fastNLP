import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np
import pytest

path = os.path.abspath(__file__)
folders = path.split(os.sep)
for folder in list(folders[::-1]):
    if 'fastnlp' not in folder.lower():
        folders.pop(-1)
    else:
        break
path = os.sep.join(folders)
sys.path.extend([path, os.path.join(path, 'fastNLP')])

from fastNLP import DataSet, Metric
from fastNLP.core.metrics.element import Element
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW
from tests.helpers.utils import skip_no_cuda
from tests.core.metrics.utils import DemoMetric, DemoMetric2

if _NEED_IMPORT_ONEFLOW:
    import oneflow


def _test(
    local_rank: int,
    world_size: int,
    device,
    dataset: DataSet,
    metric_class: Type[Metric],
    metric_kwargs: Dict[str, Any],
) -> None:
    # metric 应该是每个进程有自己的一个 instance，所以在 _test 里面实例化
    metric = metric_class(**metric_kwargs)
    # dataset 也类似（每个进程有自己的一个）
    dataset = copy.deepcopy(dataset)
    metric.to(device)
    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, len(dataset), world_size):
        a = oneflow.tensor(dataset[i]['a'])
        metric.update(a.to(device=device))
    my_result = metric.get_metric()
    if local_rank == 0:
        np.testing.assert_almost_equal(my_result['a'], [10, 14, 18, 22])


@pytest.mark.oneflowdist
def test_list_ddp() -> None:
    # 该测试的目的
    skip_no_cuda()
    global pool
    metric_kwargs = {}
    metric_kwargs['aggregate_when_get_metric'] = True
    metric_kwargs['backend'] = 'oneflow'
    dataset = DataSet(
        {'a': [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]})
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    _test(
        local_rank=local_rank,
        world_size=world_size,
        device=local_rank,
        dataset=dataset,
        metric_class=DemoMetric2,
        metric_kwargs=metric_kwargs)


@pytest.mark.oneflow
def test_oneflow_return_element():
    # 测试的目的是为了方式直接返回了Element对象
    metric = DemoMetric('oneflow')
    metric.update(a=1)
    res = metric.get_metric()
    metric.reset()
    assert res['a'] == 1


@pytest.mark.oneflow
def test_getter_setter():
    ele = Element(
        name='a',
        value=[0, 1, 4, 2, 3],
        aggregate_method='sum',
        backend='oneflow')
    x = ele[2]
    assert x == 4
    ele[4] = 100
    assert ele.to_list() == [0, 1, 4, 2, 100]
    ele2 = Element(
        name='b', value=0, aggregate_method='sum', backend='oneflow')
    with pytest.raises(TypeError):
        ele2[0]
    with pytest.raises(TypeError):
        ele2[0] = 3


@pytest.mark.oneflow
def test_element_dist():
    r"""分布式的测试"""
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'python',
        '-m',
        'oneflow.distributed.launch',
        '--nproc_per_node',
        '2',
        f"{path.joinpath('test_element_oneflow.py')}",
    ]
    subprocess.check_call(command, env=os.environ)


if __name__ == '__main__':

    pytest.main([f'{__file__}', '-m', 'oneflowdist'])
