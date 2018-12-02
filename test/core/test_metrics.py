
import unittest

class TestOptim(unittest.TestCase):
    def test_AccuracyMetric(self):
        from fastNLP.core.metrics import AccuracyMetric
        import torch
        import numpy as np

        # (1) only input, targets passed
        output_dict = {"input": torch.zeros(4, 3)}
        target_dict = {'target': torch.zeros(4)}
        metric = AccuracyMetric()

        metric(output_dict=output_dict, target_dict=target_dict)
        print(metric.get_metric())

