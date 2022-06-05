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


class TestElemnt:

    @pytest.mark.torch
    def test_case_v1(self):
        pred = torch.tensor([1, 1, 1, 1])
        metric = MyMetric()
        metric.update(pred)
        res = metric.get_metric()
        print(res)