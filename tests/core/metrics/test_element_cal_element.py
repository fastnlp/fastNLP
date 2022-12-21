import numpy as np
import pytest

from fastNLP.core.metrics import Metric
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from .utils import find_free_network_port, setup_ddp

if _NEED_IMPORT_TORCH:
    import torch


class MyMetric(Metric):
    def __init__(self):
        super(MyMetric, self).__init__()
        self.register_element(name="t1", value=0)
        self.register_element(name="t2", value=0)
        self.register_element(name="t3", value=0)

    def update(self, pred):
        self.t1 = len(pred)
        self.t2 = len(pred)
        temp = self.t1 + self.t2
        self.t3 = temp
        self.t1 = self.t3 / self.t2

    def get_metric(self) -> dict:
        return {"t1": self.t1.get_scalar(), "t2": self.t2.get_scalar(), "t3": self.t3.get_scalar()}


class MyMetric2(Metric):
    def __init__(self):
        super(MyMetric2, self).__init__()
        self.register_element(name="a", value=[0, 0, 0, 0])
        self.register_element(name="b", value=[0, 0, 0, 0])
        self.register_element(name="c", value=[0, 0, 0, 0])

    def update(self, pred):
        self.a += pred
        self.b += pred
        self.c += pred
        self.a /= 2
        self.b -= self.c
        self.c *= 2
        self.b += 2
        for i in range(4):
            self.b[i] += i

    def get_metric(self) -> dict:
        return {"a": self.a.to_list(), "b": self.b.to_list(), "c": self.c.to_list()}


class TestElemnt:

    @pytest.mark.torch
    def test_case_v1(self):
        pred = torch.tensor([1, 1, 1, 1])
        metric = MyMetric()
        metric.update(pred)
        res = metric.get_metric()
        print(res)

    @pytest.mark.torch
    def test_case_v2(self):
        pred = torch.tensor([16, 8, 4, 2])
        metric = MyMetric2()
        metric.update(pred)
        res = metric.get_metric()
        np.testing.assert_almost_equal(res["a"], [8, 4, 2, 1])
        np.testing.assert_almost_equal(res["b"], [2, 3, 4, 5])
        np.testing.assert_almost_equal(res["c"], [32, 16, 8, 4])
