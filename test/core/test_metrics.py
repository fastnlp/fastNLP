import os
import sys

sys.path = [os.path.join(os.path.dirname(__file__), '..')] + sys.path

from fastNLP.core import metrics
# from sklearn import metrics as skmetrics
import unittest
from numpy import random
from fastNLP.core.metrics import SeqLabelEvaluator
import torch


def generate_fake_label(low, high, size):
    return random.randint(low, high, size), random.randint(low, high, size)


class TestEvaluator(unittest.TestCase):
    def test_a(self):
        evaluator = SeqLabelEvaluator()
        pred = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        truth = [{"truth": torch.LongTensor([1, 2, 3, 3, 3])}, {"truth": torch.LongTensor([1, 2, 3, 3, 4])}]
        ans = evaluator(pred, truth)
        print(ans)

    def test_b(self):
        evaluator = SeqLabelEvaluator()
        pred = [[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 4, 5, 0, 0]]
        truth = [{"truth": torch.LongTensor([1, 2, 3, 3, 3, 0, 0])}, {"truth": torch.LongTensor([1, 2, 3, 3, 4, 0, 0])}]
        ans = evaluator(pred, truth)
        print(ans)


class TestMetrics(unittest.TestCase):
    delta = 1e-5
    # test for binary, multiclass, multilabel
    data_types = [((1000,), 2), ((1000,), 10), ((1000, 10), 2)]
    fake_data = [generate_fake_label(0, high, shape) for shape, high in data_types]

    def test_accuracy_score(self):
        for y_true, y_pred in self.fake_data:
            for normalize in [True, False]:
                for sample_weight in [None, random.rand(y_true.shape[0])]:
                    test = metrics.accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)
                    # ans = skmetrics.accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)
                    # self.assertAlmostEqual(test, ans, delta=self.delta)

    def test_recall_score(self):
        for y_true, y_pred in self.fake_data:
            # print(y_true.shape)
            labels = list(range(y_true.shape[1])) if len(y_true.shape) >= 2 else None
            test = metrics.recall_score(y_true, y_pred, labels=labels, average=None)
            if not isinstance(test, list):
                test = list(test)
            # ans = skmetrics.recall_score(y_true, y_pred,labels=labels, average=None)
            # ans = list(ans)
            # for a, b in zip(test, ans):
            #     # print('{}, {}'.format(a, b))
            #     self.assertAlmostEqual(a, b, delta=self.delta)
        # test binary
        y_true, y_pred = generate_fake_label(0, 2, 1000)
        test = metrics.recall_score(y_true, y_pred)
        # ans = skmetrics.recall_score(y_true, y_pred)
        # self.assertAlmostEqual(ans, test, delta=self.delta)

    def test_precision_score(self):
        for y_true, y_pred in self.fake_data:
            # print(y_true.shape)
            labels = list(range(y_true.shape[1])) if len(y_true.shape) >= 2 else None
            test = metrics.precision_score(y_true, y_pred, labels=labels, average=None)
            # ans = skmetrics.precision_score(y_true, y_pred,labels=labels, average=None)
            # ans, test = list(ans), list(test)
            # for a, b in zip(test, ans):
            #     # print('{}, {}'.format(a, b))
            #     self.assertAlmostEqual(a, b, delta=self.delta)
        # test binary
        y_true, y_pred = generate_fake_label(0, 2, 1000)
        test = metrics.precision_score(y_true, y_pred)
        # ans = skmetrics.precision_score(y_true, y_pred)
        # self.assertAlmostEqual(ans, test, delta=self.delta)

    def test_f1_score(self):
        for y_true, y_pred in self.fake_data:
            # print(y_true.shape)
            labels = list(range(y_true.shape[1])) if len(y_true.shape) >= 2 else None
            test = metrics.f1_score(y_true, y_pred, labels=labels, average=None)
            # ans = skmetrics.f1_score(y_true, y_pred,labels=labels, average=None)
            # ans, test = list(ans), list(test)
            # for a, b in zip(test, ans):
            #     # print('{}, {}'.format(a, b))
            #     self.assertAlmostEqual(a, b, delta=self.delta)
        # test binary
        y_true, y_pred = generate_fake_label(0, 2, 1000)
        test = metrics.f1_score(y_true, y_pred)
        # ans = skmetrics.f1_score(y_true, y_pred)
        # self.assertAlmostEqual(ans, test, delta=self.delta)


if __name__ == '__main__':
    unittest.main()
