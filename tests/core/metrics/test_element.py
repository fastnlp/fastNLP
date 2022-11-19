import pytest

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