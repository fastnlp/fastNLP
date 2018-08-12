import sys, os
sys.path = [os.path.join(os.path.dirname(__file__), '..')] + sys.path

from fastNLP.core import metrics
from sklearn import metrics as skmetrics
import unittest
import numpy as np
from numpy import random

def generate_fake_label(low, high, size):
    return random.randint(low, high, size), random.randint(low, high, size)

class TestMetrics(unittest.TestCase):
    delta = 1e-5
    # test for binary, multiclass, multilabel
    data_types = [((1000,), 2), ((1000,), 10), ((1000, 10), 2)]
    fake_data = [generate_fake_label(0, high, shape) for shape, high in data_types]
    def test_accuracy_score(self):
        for y_true, y_pred in self.fake_data:
            for normalize in [True, False]:
                for sample_weight in [None, random.rand(y_true.shape[0])]:
                    ans = skmetrics.accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)
                    test = metrics.accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)
                    self.assertAlmostEqual(test, ans, delta=self.delta)
    
    def test_recall_score(self):
        for y_true, y_pred in self.fake_data:
            # print(y_true.shape)
            labels = list(range(y_true.shape[1])) if len(y_true.shape) >= 2 else None
            ans = skmetrics.recall_score(y_true, y_pred,labels=labels, average=None)
            test = metrics.recall_score(y_true, y_pred, labels=labels, average=None)
            ans = list(ans)
            if not isinstance(test, list):
                test = list(test)
            for a, b in zip(test, ans):
                # print('{}, {}'.format(a, b))
                self.assertAlmostEqual(a, b, delta=self.delta)
        # test binary
        y_true, y_pred = generate_fake_label(0, 2, 1000)
        ans = skmetrics.recall_score(y_true, y_pred)
        test = metrics.recall_score(y_true, y_pred)
        self.assertAlmostEqual(ans, test, delta=self.delta)

    def test_precision_score(self):
        for y_true, y_pred in self.fake_data:
            # print(y_true.shape)
            labels = list(range(y_true.shape[1])) if len(y_true.shape) >= 2 else None
            ans = skmetrics.precision_score(y_true, y_pred,labels=labels, average=None)
            test = metrics.precision_score(y_true, y_pred, labels=labels, average=None)
            ans, test = list(ans), list(test)
            for a, b in zip(test, ans):
                # print('{}, {}'.format(a, b))
                self.assertAlmostEqual(a, b, delta=self.delta)
        # test binary
        y_true, y_pred = generate_fake_label(0, 2, 1000)
        ans = skmetrics.precision_score(y_true, y_pred)
        test = metrics.precision_score(y_true, y_pred)
        self.assertAlmostEqual(ans, test, delta=self.delta)
    
    def test_precision_score(self):
        for y_true, y_pred in self.fake_data:
            # print(y_true.shape)
            labels = list(range(y_true.shape[1])) if len(y_true.shape) >= 2 else None
            ans = skmetrics.precision_score(y_true, y_pred,labels=labels, average=None)
            test = metrics.precision_score(y_true, y_pred, labels=labels, average=None)
            ans, test = list(ans), list(test)
            for a, b in zip(test, ans):
                # print('{}, {}'.format(a, b))
                self.assertAlmostEqual(a, b, delta=self.delta)
        # test binary
        y_true, y_pred = generate_fake_label(0, 2, 1000)
        ans = skmetrics.precision_score(y_true, y_pred)
        test = metrics.precision_score(y_true, y_pred)
        self.assertAlmostEqual(ans, test, delta=self.delta)

    def test_f1_score(self):
        for y_true, y_pred in self.fake_data:
            # print(y_true.shape)
            labels = list(range(y_true.shape[1])) if len(y_true.shape) >= 2 else None
            ans = skmetrics.f1_score(y_true, y_pred,labels=labels, average=None)
            test = metrics.f1_score(y_true, y_pred, labels=labels, average=None)
            ans, test = list(ans), list(test)
            for a, b in zip(test, ans):
                # print('{}, {}'.format(a, b))
                self.assertAlmostEqual(a, b, delta=self.delta)
        # test binary
        y_true, y_pred = generate_fake_label(0, 2, 1000)
        ans = skmetrics.f1_score(y_true, y_pred)
        test = metrics.f1_score(y_true, y_pred)
        self.assertAlmostEqual(ans, test, delta=self.delta)

if __name__ == '__main__':
    unittest.main()