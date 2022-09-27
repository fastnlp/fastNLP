import pytest
from fastNLP import Metric


class DemoMetric(Metric):

    def __init__(self):
        super().__init__(backend='torch')
        self.count = 0
        self.register_element('count', 0)

    def evaluate(self):
        self.count += 1
        print(self.count)


class DemoMetric1(Metric):

    def __init__(self):
        super().__init__(backend='torch')
        self.register_element('count', 0)
        self.count = 2

    def evaluate(self):
        self.count += 1
        return self.count


@pytest.mark.torch
class TestMetric:

    def test_v1(self):
        with pytest.raises(RuntimeError):
            dmtr = DemoMetric()
            dmtr.evaluate()

    def test_v2(self):
        dmtr = DemoMetric1()
        assert 3 == dmtr.evaluate()
