import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np
import pytest

from fastNLP import DataSet, Metric
from fastNLP.core import apply_to_collection
from fastNLP.core.metrics.backend import AutoBackend
from fastNLP.core.metrics.element import Element
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.core.drivers.paddle_driver.fleet_launcher import FleetLauncher
from tests.helpers.utils import skip_no_cuda
from .utils import DemoMetric, DemoMetric2

if _NEED_IMPORT_PADDLE:
    import paddle
    import paddle.distributed as dist


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
    dataset = deepcopy(dataset)
    metric.to(device)
    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, len(dataset), world_size):
        a = paddle.to_tensor(dataset[i]['a'])
        metric.update(a)
    my_result = metric.get_metric()
    if local_rank == 0:
        np.testing.assert_almost_equal(my_result['a'], [10, 14, 18, 22])


@pytest.mark.paddledist
class TestElementFleet:

    @classmethod
    def setup_class(cls):
        skip_no_cuda()
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

    def test_list_ddp(self) -> None:
        # 该测试的目的
        global pool
        metric_kwargs = {}
        metric_kwargs['aggregate_when_get_metric'] = True
        metric_kwargs['backend'] = 'paddle'
        world_size = int(os.environ['PADDLE_TRAINERS_NUM'])
        dataset = DataSet(
            {'a': [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]})
        _test(
            local_rank=self.local_rank,
            world_size=world_size,
            device=self.local_rank,
            dataset=dataset,
            metric_class=DemoMetric2,
            metric_kwargs=metric_kwargs)


@pytest.mark.paddle
def test_paddle_return_element():
    # 测试的目的是为了方式直接返回了Element对象
    metric = DemoMetric('paddle')
    metric.update(a=1)
    res = metric.get_metric()
    metric.reset()
    assert res['a'] == 1


@pytest.mark.paddle
def test_apply_to_collection():
    # 测试apply_to_collection这个函数是否运行正常
    backend = AutoBackend('paddle')
    element_float = Element(
        name='element1', value=3.3, aggregate_method='sum', backend=backend)
    element_Element = Element(
        name='element2',
        value=[0, 1, 2, 3],
        aggregate_method='sum',
        backend=backend)
    results = {'element1': element_float, 'element2': element_Element}
    outputs = apply_to_collection(
        results,
        dtype=Element,
        function=lambda x: x.get_scalar()
        if x.value.ndim == 1 and x.value.shape[0] == 1 else x.to_list(),
        include_none=False)

    assert type(outputs['element1']) == float
    assert type(outputs['element2']) == list


@pytest.mark.paddle
def test_getter_setter():
    ele = Element(
        name='a',
        value=[0, 1, 4, 2, 3],
        aggregate_method='sum',
        backend='paddle')
    x = ele[2]
    assert x == 4
    ele[4] = 100
    assert ele.to_list() == [0, 1, 4, 2, 100]
    ele2 = Element(name='b', value=0, aggregate_method='sum', backend='paddle')
    with pytest.raises(TypeError):
        ele2[0]
    with pytest.raises(TypeError):
        ele2[0] = 3


@pytest.mark.paddle
def test_element_dist():
    """测试上面的测试函数."""
    skip_no_cuda()
    path = Path(os.path.abspath(__file__)).parent
    command = [
        'pytest', f"{path.joinpath('test_element_paddle.py')}", '-m',
        'paddledist'
    ]
    env = deepcopy(os.environ)
    env['FASTNLP_BACKEND'] = 'paddle'
    subprocess.check_call(command, env=env)
