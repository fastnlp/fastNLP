
import unittest

from fastNLP.core.metrics import AccuracyMetric
import torch
import numpy as np

class TestAccuracyMetric(unittest.TestCase):
    def test_AccuracyMetric1(self):
        # (1) only input, targets passed
        pred_dict = {"pred": torch.zeros(4, 3)}
        target_dict = {'target': torch.zeros(4)}
        metric = AccuracyMetric()

        metric(pred_dict=pred_dict, target_dict=target_dict,  )
        print(metric.get_metric())
    #
    def test_AccuracyMetric2(self):
        # (2) with corrupted size
        try:
            pred_dict = {"pred": torch.zeros(4, 3, 2)}
            target_dict = {'target': torch.zeros(4)}
            metric = AccuracyMetric()

            metric(pred_dict=pred_dict, target_dict=target_dict,  )
            print(metric.get_metric())
        except Exception as e:
            print(e)
            return
        self.assertTrue(True, False), "No exception catches."
    #
    def test_AccuracyMetric3(self):
        # (3) the second batch is corrupted size
        try:
            metric = AccuracyMetric()
            pred_dict = {"pred": torch.zeros(4, 3, 2)}
            target_dict = {'target': torch.zeros(4, 3)}
            metric(pred_dict=pred_dict, target_dict=target_dict)

            pred_dict = {"pred": torch.zeros(4, 3, 2)}
            target_dict = {'target': torch.zeros(4)}
            metric(pred_dict=pred_dict, target_dict=target_dict)

            print(metric.get_metric())
        except Exception as e:
            print(e)
            return
        self.assertTrue(True, False), "No exception catches."

    #
    def test_AccuaryMetric4(self):
        # (5) check reset
        metric = AccuracyMetric()
        pred_dict = {"pred": torch.zeros(4, 3, 2)}
        target_dict = {'target': torch.zeros(4, 3)}
        metric(pred_dict=pred_dict, target_dict=target_dict)
        self.assertDictEqual(metric.get_metric(), {'acc': 1})

        pred_dict = {"pred": torch.zeros(4, 3, 2)}
        target_dict = {'target': torch.zeros(4, 3)+1}
        metric(pred_dict=pred_dict, target_dict=target_dict)
        self.assertDictEqual(metric.get_metric(), {'acc':0})

    def test_AccuaryMetric5(self):
        # (5) check reset
        metric = AccuracyMetric()
        pred_dict = {"pred": torch.zeros(4, 3, 2)}
        target_dict = {'target': torch.zeros(4, 3)}
        metric(pred_dict=pred_dict, target_dict=target_dict)
        self.assertDictEqual(metric.get_metric(reset=False), {'acc': 1})

        pred_dict = {"pred": torch.zeros(4, 3, 2)}
        target_dict = {'target': torch.zeros(4, 3)+1}
        metric(pred_dict=pred_dict, target_dict=target_dict)
        self.assertDictEqual(metric.get_metric(), {'acc':0.5})

    #
    def test_AccuaryMetric6(self):
        # (6) check numpy array is not acceptable
        try:
            metric = AccuracyMetric()
            pred_dict = {"pred": np.zeros((4, 3, 2))}
            target_dict = {'target': np.zeros((4, 3))}
            metric(pred_dict=pred_dict, target_dict=target_dict)
        except Exception as e:
            print(e)
            return
        self.assertTrue(True, False), "No exception catches."

    def test_AccuaryMetric7(self):
        # (7) check map, match
        metric = AccuracyMetric(pred='predictions', target='targets')
        pred_dict = {"predictions": torch.zeros(4, 3, 2)}
        target_dict = {'targets': torch.zeros(4, 3)}
        metric(pred_dict=pred_dict, target_dict=target_dict)
        self.assertDictEqual(metric.get_metric(), {'acc': 1})

    def test_AccuaryMetric8(self):
        # (8) check map, does not match. use stop_fast_param to stop fast param map
        try:
            metric = AccuracyMetric(pred='predictions', target='targets')
            pred_dict = {"prediction": torch.zeros(4, 3, 2), "stop_fast_param":1}
            target_dict = {'targets': torch.zeros(4, 3)}
            metric(pred_dict=pred_dict, target_dict=target_dict,  )
            self.assertDictEqual(metric.get_metric(), {'acc': 1})
        except Exception as e:
            print(e)
            return
        self.assertTrue(True, False), "No exception catches."

    def test_AccuaryMetric9(self):
        # (9) check map, include unused
        try:
            metric = AccuracyMetric(pred='prediction', target='targets')
            pred_dict = {"prediction": torch.zeros(4, 3, 2), 'unused':1}
            target_dict = {'targets': torch.zeros(4, 3)}
            metric(pred_dict=pred_dict, target_dict=target_dict)
            self.assertDictEqual(metric.get_metric(), {'acc': 1})
        except Exception as e:
            print(e)
            return
        self.assertTrue(True, False), "No exception catches."

    def test_AccuaryMetric10(self):
        # (10) check _fast_metric
        try:
            metric = AccuracyMetric()
            pred_dict = {"predictions": torch.zeros(4, 3, 2), "masks": torch.zeros(4, 3)}
            target_dict = {'targets': torch.zeros(4, 3)}
            metric(pred_dict=pred_dict, target_dict=target_dict)
            self.assertDictEqual(metric.get_metric(), {'acc': 1})
        except Exception as e:
            print(e)
            return
        self.assertTrue(True, False), "No exception catches."
