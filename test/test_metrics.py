import sys, os
sys.path = [os.path.abspath('..')] + sys.path

from fastNLP.action.metrics import accuracy_score
from sklearn import metrics as M
import unittest
import numpy as np
from numpy import random

def generate_fake_label(low, high, size):
    return random.randint(low, high, size), random.randint(low, high, size)

class TestMetrics(unittest.TestCase):
    delta = 1e-5
    def test_accuracy_score(self):
        for shape, high_bound in [((1000,), 2), ((1000,), 10), ((1000, 10), 2)]:
            # test for binary, multiclass, multilabel
            y_true, y_pred = generate_fake_label(0, high_bound, shape)
            for normalize in [True, False]:
                for sample_weight in [None, random.rand(shape[0])]:
                    test = accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)
                    ans = M.accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)
                    self.assertAlmostEqual(test, ans, delta=self.delta)


if __name__ == '__main__':
    unittest.main()