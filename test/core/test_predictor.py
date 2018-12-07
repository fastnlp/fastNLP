import unittest

import numpy as np
import torch

from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.core.predictor import Predictor
from fastNLP.modules.encoder.linear import Linear


def prepare_fake_dataset():
    mean = np.array([-3, -3])
    cov = np.array([[1, 0], [0, 1]])
    class_A = np.random.multivariate_normal(mean, cov, size=(1000,))

    mean = np.array([3, 3])
    cov = np.array([[1, 0], [0, 1]])
    class_B = np.random.multivariate_normal(mean, cov, size=(1000,))

    data_set = DataSet([Instance(x=[float(item[0]), float(item[1])], y=[0.0]) for item in class_A] +
                       [Instance(x=[float(item[0]), float(item[1])], y=[1.0]) for item in class_B])
    return data_set


class TestPredictor(unittest.TestCase):
    def test(self):
        predictor = Predictor()
        model = Linear(2, 1)
        data = prepare_fake_dataset()
        data.set_input("x")
        ans = predictor.predict(model, data)
        self.assertEqual(len(ans), 2000)
        self.assertTrue(isinstance(ans[0], torch.Tensor))
