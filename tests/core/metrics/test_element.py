from typing import List

import pytest

from fastNLP.core import apply_to_collection
from fastNLP.core.metrics.backend import AutoBackend
from fastNLP.core.metrics.element import Element
from fastNLP import Metric


class DemoMetric(Metric):
    def __init__(self, backend):
        super(DemoMetric, self).__init__(backend=backend)
        self.register_element('a', 0, aggregate_method='sum')

    def update(self, a):
        self.a += a

    def get_metric(self) -> dict:
        return {'a': self.a}


@pytest.mark.torch
def test_torch_return_element():
    # 测试的目的是为了方式直接返回了Element对象
    metric = DemoMetric('torch')
    metric.update(a=1)
    res = metric.get_metric()
    metric.reset()
    assert res['a'] == 1


@pytest.mark.paddle
def test_paddle_return_element():
    # 测试的目的是为了方式直接返回了Element对象
    metric = DemoMetric('paddle')
    metric.update(a=1)
    res = metric.get_metric()
    metric.reset()
    assert res['a'] == 1

@pytest.mark.oneflow
def test_oneflow_return_element():
    # 测试的目的是为了方式直接返回了Element对象
    metric = DemoMetric('oneflow')
    metric.update(a=1)
    res = metric.get_metric()
    metric.reset()
    assert res['a'] == 1

@pytest.mark.jittor
def test_jittor_return_element():
    # 测试的目的是为了方式直接返回了Element对象
    metric = DemoMetric('jittor')
    metric.update(a=1)
    res = metric.get_metric()
    metric.reset()
    assert res['a'] == 1

@pytest.mark.oneflow
def test_apply_to_collection():
    # 测试apply_to_collection这个函数是否运行正常
    backend = AutoBackend("paddle")
    element_float = Element(name="element1", value=3, aggregate_method="sum",backend=backend)
    element_Element = Element(name="element2", value=[0,1,2,3],aggregate_method="sum",backend=backend)
    results = {"element1":element_float, "element2":element_Element}
    outputs = apply_to_collection(results, dtype=Element,
                                  function=lambda x:x.get_scalar() if x.value.ndim==1 and x.value.shape[0]==1 else x.to_list(),
                                  include_none=False)
    print(results)
    print(outputs)
