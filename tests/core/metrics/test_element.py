from fastNLP.core.metrics.metric import Metric

from collections import defaultdict
from functools import partial

import unittest


class MyMetric(Metric):

    def __init__(self, backend='auto',
                 aggregate_when_get_metric: bool = False):
        super(MyMetric, self).__init__(backend=backend, aggregate_when_get_metric=aggregate_when_get_metric)

        self.tp = defaultdict(partial(self.register_element, aggregate_method='sum'))

    def update(self, item):
        self.tp['1'] += item


class TestMetric(unittest.TestCase):

    def test_va1(self):
        my = MyMetric()
        my.update(1)
        print(my.tp['1'])
