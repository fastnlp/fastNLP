
import unittest

from fastNLP.core.metrics import AccuracyMetric
import torch
import numpy as np

class TestAccuracyMetric(unittest.TestCase):
    def test_AccuracyMetric1(self):
        # (1) only input, targets passed
        output_dict = {"pred": torch.zeros(4, 3)}
        target_dict = {'target': torch.zeros(4)}
        metric = AccuracyMetric()

        metric(output_dict=output_dict, target_dict=target_dict)
        print(metric.get_metric())

    def test_AccuracyMetric2(self):
        # (2) with corrupted size
        output_dict = {"pred": torch.zeros(4, 3, 2)}
        target_dict = {'target': torch.zeros(4)}
        metric = AccuracyMetric()

        metric(output_dict=output_dict, target_dict=target_dict)
        print(metric.get_metric())

    def test_AccuracyMetric3(self):
        # (3) with check=False , the second batch is corrupted size
        metric = AccuracyMetric()
        output_dict = {"pred": torch.zeros(4, 3, 2)}
        target_dict = {'target': torch.zeros(4, 3)}
        metric(output_dict=output_dict, target_dict=target_dict)

        output_dict = {"pred": torch.zeros(4, 3, 2)}
        target_dict = {'target': torch.zeros(4)}
        metric(output_dict=output_dict, target_dict=target_dict)

        print(metric.get_metric())

    def test_AccuracyMetric4(self):
        # (4) with check=True , the second batch is corrupted size
        metric = AccuracyMetric()
        output_dict = {"pred": torch.zeros(4, 3, 2)}
        target_dict = {'target': torch.zeros(4, 3)}
        metric(output_dict=output_dict, target_dict=target_dict)

        output_dict = {"pred": torch.zeros(4, 3, 2)}
        target_dict = {'target': torch.zeros(4)}
        metric(output_dict=output_dict, target_dict=target_dict, check=True)

        print(metric.get_metric())

    def test_AccuaryMetric5(self):
        # (5) check reset
        metric = AccuracyMetric()
        output_dict = {"pred": torch.zeros(4, 3, 2)}
        target_dict = {'target': torch.zeros(4, 3)}
        metric(output_dict=output_dict, target_dict=target_dict)
        self.assertDictEqual(metric.get_metric(), {'acc': 1})

        output_dict = {"pred": torch.zeros(4, 3, 2)}
        target_dict = {'target': torch.zeros(4, 3)+1}
        metric(output_dict=output_dict, target_dict=target_dict)
        self.assertDictEqual(metric.get_metric(), {'acc':0})

    def test_AccuaryMetric6(self):
        # (6) check numpy array is not acceptable
        metric = AccuracyMetric()
        output_dict = {"pred": np.zeros((4, 3, 2))}
        target_dict = {'target': np.zeros((4, 3))}
        metric(output_dict=output_dict, target_dict=target_dict)
        self.assertDictEqual(metric.get_metric(), {'acc': 1})