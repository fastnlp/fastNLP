import unittest
from fastNLP.core.metrics.utils import func_post_proc


class Metric:
    def accumulate(self, x, y):
        return x, y

    def compute(self, x, y):
        return x, y


class TestMetricUtil(unittest.TestCase):
    def test_func_post_proc(self):
        metric = Metric()
        metric = func_post_proc(metric, lambda o: {'x': o[0], 'y': o[1]}, method_name='accumulate')
        self.assertDictEqual({'x': 1, 'y': 2}, metric.accumulate(x=1, y=2))

        func_post_proc(metric, lambda o: {'1': o['x'], '2': o['y']}, method_name='accumulate')
        self.assertDictEqual({'1': 1, '2': 2}, metric.accumulate(x=1, y=2))

        metric = func_post_proc(metric, lambda o: {'x': o[0], 'y': o[1]}, method_name='update')
        self.assertDictEqual({'x': 1, 'y': 2}, metric.update(x=1, y=2))

        func_post_proc(metric, lambda o: {'1': o['x'], '2': o['y']}, method_name='update')
        self.assertDictEqual({'1': 1, '2': 2}, metric.update(x=1, y=2))

    def test_check_accumulate_post_special_local_variable(self):
        metric = Metric()
        self.assertFalse(hasattr(metric, '__wrapped_by_fn__'))
        metric = func_post_proc(metric, lambda o: {'x': o[0], 'y': o[1]}, method_name='update')
        self.assertTrue(hasattr(metric, '__wrapped_by_fn__'))
