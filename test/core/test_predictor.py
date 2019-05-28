import unittest
from collections import defaultdict

import numpy as np
import torch

from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.core.predictor import Predictor


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


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return {"predict": self.linear(x)}


class TestPredictor(unittest.TestCase):
    def test_simple(self):
        model = LinearModel()
        predictor = Predictor(model)
        data = prepare_fake_dataset()
        data.set_input("x")
        ans = predictor.predict(data)
        self.assertTrue(isinstance(ans, defaultdict))
        self.assertTrue("predict" in ans)
        self.assertTrue(isinstance(ans["predict"], list))

    def test_sequence(self):
        # test sequence input/output
        pass
